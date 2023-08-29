import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from depth.models.builder import LOSSES

def masked_shift_and_scale(depth_preds, depth_gt, mask_valid):
    depth_preds_nan = depth_preds.clone()
    depth_gt_nan = depth_gt.clone()
    depth_preds_nan[~mask_valid] = np.nan
    depth_gt_nan[~mask_valid] = np.nan

    mask_diff = mask_valid.reshape(mask_valid.size()[:2] + (-1,)).sum(-1, keepdims=True) + 1

    t_gt = depth_gt_nan.reshape(depth_gt_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
    t_gt[torch.isnan(t_gt)] = 0
    diff_gt = torch.abs(depth_gt - t_gt)
    diff_gt[~mask_valid] = 0
    s_gt = (diff_gt.reshape(diff_gt.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
    depth_gt_aligned = (depth_gt - t_gt) / (s_gt + 1e-6)


    t_pred = depth_preds_nan.reshape(depth_preds_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
    t_pred[torch.isnan(t_pred)] = 0
    diff_pred = torch.abs(depth_preds - t_pred)
    diff_pred[~mask_valid] = 0
    s_pred = (diff_pred.reshape(diff_pred.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
    depth_pred_aligned = (depth_preds - t_pred) / (s_pred + 1e-6)

    return depth_pred_aligned, depth_gt_aligned

def masked_l1_loss(preds, target, mask_valid,dense=False):
    element_wise_loss = abs(preds - target)
    element_wise_loss[~mask_valid] = 0
    if dense is False:
        return element_wise_loss.sum() / mask_valid.sum()
    else: # not average
        return element_wise_loss


def get_contexts_dr( level, depth_gt, mask_valid):
    batch_norm_context = []
    for mask_index in range(depth_gt.shape[0]): #process each img in the batch
        depth_map = depth_gt[mask_index]
        valid_map = mask_valid[mask_index]

        if depth_map[valid_map].numel() == 0: #if there is no valid pixel
            map_context_list = [valid_map for _ in range(2 ** (level) - 1)]
        else:
            valid_values = depth_map[valid_map]
            max_d = valid_values.max()
            min_d = valid_values.min()
            bin_size_list = [(1 / 2) ** (i) for i in range(level)]
            bin_size_list.reverse()
            map_context_list = []
            for bin_size in bin_size_list:
                for i in range(int(1 / bin_size)):
                    mask_new = (depth_map >= min_d + (max_d - min_d) * i * bin_size) & (
                            depth_map < min_d + (max_d - min_d) * (i + 1) * bin_size + 1e-30)
                    mask_new = mask_new & valid_map
                    map_context_list.append(mask_new)
                    
        map_context_list = torch.stack(map_context_list, dim=0)
        batch_norm_context.append(map_context_list)
    batch_norm_context = torch.stack(batch_norm_context, dim=0).swapdims(0, 1)

    return batch_norm_context



def get_contexts_dp( level, depth_gt, mask_valid):

        depth_gt_nan=depth_gt.clone()
        depth_gt_nan[~mask_valid] = np.nan
        depth_gt_nan=depth_gt_nan.view(depth_gt_nan.shape[0], depth_gt_nan.shape[1], -1)

        bin_size_list = [(1 / 2) ** (i) for i in range(level)]
        bin_size_list.reverse()


        batch_norm_context=[]
        for bin_size in bin_size_list:
            num_bins=int(1/bin_size)

            for bin_index in range(num_bins):

                min_bin=depth_gt_nan.nanquantile(bin_index*bin_size,dim=-1).unsqueeze(-1).unsqueeze(-1)
                max_bin=depth_gt_nan.nanquantile((bin_index+1) * bin_size, dim=-1).unsqueeze(-1).unsqueeze(-1)

                new_mask_valid=mask_valid
                new_mask_valid=new_mask_valid &  (depth_gt>=min_bin)
                new_mask_valid = new_mask_valid & (depth_gt < max_bin)
                batch_norm_context.append(new_mask_valid)
        batch_norm_context = torch.stack(batch_norm_context, dim=0)
        return batch_norm_context



def init_temp_masks_ds(level,image_size, device):
        size=image_size
        bin_size_list = [(1 / 2) ** (i) for i in range(level)]
        bin_size_list.reverse()

        map_level_list = []
        for bin_size in bin_size_list:  # e.g. 1/8
            for h in range(int(1 / bin_size)):
                for w in range(int(1 / bin_size)):
                    mask_new=torch.zeros(1,1,size[0],size[1], device=device)
                    mask_new[:,:, int(h * bin_size * size[0]):int((h + 1) * bin_size * size[0]),
                    int(w * bin_size * size[1]):int((w + 1) * bin_size * size[1])] = 1
                    mask_new = mask_new> 0
                    map_level_list.append(mask_new)
        batch_norm_context=torch.stack(map_level_list,dim=0)
        return batch_norm_context


def get_contexts_ds( level, mask_valid):
    templete_contexts=init_temp_masks_ds(level,mask_valid.shape[-2:], device=mask_valid.device)

    batch_norm_context = mask_valid.unsqueeze(0)
    batch_norm_context = batch_norm_context.repeat(templete_contexts.shape[0], 1, 1, 1, 1)
    batch_norm_context = batch_norm_context & templete_contexts

    return batch_norm_context


class SSIMAE(nn.Module):
    #modified from omnidata github https://github.com/EPFL-VILAB/omnidata
    def __init__(self):
        super().__init__()

    def forward(self, depth_preds, depth_gt, mask_valid,dense):
        depth_pred_aligned, depth_gt_aligned = masked_shift_and_scale(depth_preds, depth_gt, mask_valid) #normalize the depth maps
        ssi_mae_loss = masked_l1_loss(depth_pred_aligned, depth_gt_aligned, mask_valid,dense)
        return ssi_mae_loss



def compute_hdn_loss(SSI_LOSS,depth_preds,depth_gt,mask_valid_list,mask_valid):
    hdn_loss_level = SSI_LOSS( #batch computation
        depth_preds.unsqueeze(0).repeat(mask_valid_list.shape[0], 1, 1, 1, 1).reshape(-1,
                                                                                      *depth_preds.shape[
                                                                                       -3:]),
        depth_gt.unsqueeze(0).repeat(mask_valid_list.shape[0], 1, 1, 1, 1).reshape(-1,
                                                                                   *depth_gt.shape[
                                                                                    -3:]),
        mask_valid_list.reshape(-1, *mask_valid_list.shape[-3:]), dense=True)

    hdn_loss_level_list = hdn_loss_level.reshape(*mask_valid_list.shape)
    hdn_loss_level_list = hdn_loss_level_list.sum(dim=0)  # summed loss generated by  different contexts  for all locations
    mask_valid_list_times = mask_valid_list.sum(dim=0)  # the number of  contexts for each locations
    valid_locations = (mask_valid_list_times != 0)  # valid locations
    hdn_loss_level_list[valid_locations] = hdn_loss_level_list[valid_locations] / mask_valid_list_times[
        valid_locations]  # mean loss in each location
    hdn_loss = hdn_loss_level_list.sum() / mask_valid.sum()  # average the losses of all locations
    return hdn_loss

@LOSSES.register_module()
class HDNLoss(nn.Module):
    """HDNLoss wrapper.
    Args:
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 loss_weight=1.0,
                 level=3,
                 ltype='dr',
                 num_merged=25,
                 valid_thr=1e-3):
        super(HDNLoss, self).__init__()
        self.loss_weight = loss_weight
        self.SSI_LOSS=SSIMAE() 
        self.valid_thr = valid_thr
        self.level = level
        self.ltype = ltype
        self.num_merged = num_merged

    def forward(self,
                input,
                target,
                sam=None):
        """Forward function."""
        mask_valid = target > self.valid_thr
        
        if self.ltype == 'dr':
            mask_valid_list_dr = get_contexts_dr(self.level, target, mask_valid)  # get  contexts by hdn_dr
            loss = compute_hdn_loss(self.SSI_LOSS, input, target, mask_valid_list_dr, mask_valid)
        elif self.ltype == 'dp':
            mask_valid_list_dp = get_contexts_dp(self.level, target, mask_valid)  # get  contexts by hdn_dp
            loss = compute_hdn_loss(self.SSI_LOSS, input, target, mask_valid_list_dp, mask_valid)
        elif self.ltype == 'ds':
            mask_valid_list_ds = get_contexts_ds(self.level, mask_valid)  # get  contexts by hdn_ds
            loss = compute_hdn_loss(self.SSI_LOSS, input, target, mask_valid_list_ds, mask_valid)
        elif self.ltype == 'sam':
            # TODO implement HDN-SAM loss
            sam = sam.permute(3, 0, 1, 2).unsqueeze(2)
            loss = compute_hdn_loss(self.SSI_LOSS, input, target, sam, mask_valid)
        elif self.ltype == 'sam-pair': # random merge pair mask as mask_valid_list
            # random choice 2 masks in sam and merge it
            idx1, idx2 = np.random.choice(sam.shape[-1], size=(2, self.num_merged))
            new_masks = sam[..., idx1] | sam[..., idx2]
            all_masks = torch.cat([sam, new_masks], dim=-1)
            all_mask_valid_list = all_masks.permute(3, 0, 1, 2).unsqueeze(2)
            loss = compute_hdn_loss(self.SSI_LOSS, input, target, all_mask_valid_list, mask_valid)
        else:
            raise NotImplementedError
        return self.loss_weight*loss

if __name__ == '__main__':
    batch_size=2
    valid_thr = 1e-1

    depth_preds=torch.rand(2,1,384,512) #predicted depth maps
    depth_gt=torch.rand(2,1,384,512) #ground truth depth maps
    # mask_valid=depth_gt>valid_thr #valid pixels
    

    # SSI_LOSS=SSIMAE() #ssi loss function

    # mask_valid_list_dr = get_contexts_dr(3, depth_gt, mask_valid)  # get  contexts by hdn_dr
    # mask_valid_list_dp = get_contexts_dp(3, depth_gt, mask_valid)  # get  contexts by hdn_dp
    # mask_valid_list_ds = get_contexts_ds(3, mask_valid)  # get  contexts by hdn_ds


    # loss_hdn_dr = compute_hdn_loss(SSI_LOSS, depth_preds, depth_gt, mask_valid_list_dr, mask_valid)
    # loss_hdn_dp = compute_hdn_loss(SSI_LOSS, depth_preds, depth_gt, mask_valid_list_dp, mask_valid)
    # loss_hdn_ds = compute_hdn_loss(SSI_LOSS, depth_preds, depth_gt, mask_valid_list_ds, mask_valid)
    # print("original implementation: ",loss_hdn_ds,loss_hdn_dp,loss_hdn_dr)
    
    hdn = HDNLoss(valid_thr=valid_thr) #HDN loss function
    hdn_loss=hdn(depth_preds,depth_gt)
    print("HDNLoss warpper: ",hdn_loss)

