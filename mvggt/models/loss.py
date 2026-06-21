import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
import math

from ..utils.geometry import homogenize_points, se3_inverse, depth_edge
from ..utils.alignment import align_points_scale

from datasets import __HIGH_QUALITY_DATASETS__, __MIDDLE_QUALITY_DATASETS__

# ---------------------------------------------------------------------------
# Some functions from MoGe
# ---------------------------------------------------------------------------

def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)

def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))

# ---------------------------------------------------------------------------
# PointLoss: Scale-invariant Local Pointmap
# ---------------------------------------------------------------------------

class PointLoss(nn.Module):
    def __init__(self, local_align_res=4096, train_conf=False, expected_dist_thresh=0.02):
        super().__init__()
        self.local_align_res = local_align_res
        self.criteria_local = nn.L1Loss(reduction='none')

        self.train_conf = train_conf
        if self.train_conf:
            self.prepare_segformer()
            self.conf_loss_fn = torch.nn.BCEWithLogitsLoss()
            self.expected_dist_thresh = expected_dist_thresh

    def prepare_segformer(self):
        from mvggt.models.segformer.model import EncoderDecoder
        self.segformer = EncoderDecoder()
        self.segformer.load_state_dict(torch.load('ckpts/segformer.b0.512x512.ade.160k.pth', map_location=torch.device('cpu'), weights_only=False)['state_dict'])
        self.segformer = self.segformer.cuda()

    def predict_sky_mask(self, imgs):
        with torch.no_grad():
            output = self.segformer.inference_(imgs)
            output = output == 2
        return output

    def prepare_ROE(self, pts, mask, target_size=4096):
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]

            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
                # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
            else:
                valid_pts = torch.ones((target_size, C), device=valid_pts.device)

            output.append(valid_pts)

        return torch.stack(output, dim=0)
    
    def noraml_loss(self, points, gt_points, mask):
        not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

        leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

        loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

        loss = loss.mean() / (4 * max(points.shape[-3:-1]))

        return loss

    def forward(self, pred, gt):
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        details = dict()
        final_loss = 0.0

        B, N, H, W, _ = pred_local_pts.shape

        weights_ = gt_local_pts[..., 2]
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)

        # alignment
        with torch.no_grad():
            xyz_pred_local = self.prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_gt_local = self.prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_weights_local = self.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]

            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            S_opt_local[S_opt_local <= 0] *= -1

        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts

        # local point loss
        local_pts_loss = self.criteria_local(aligned_local_pts[valid_masks].float(), gt_local_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

        # conf loss
        if self.train_conf:
            pred_conf = pred['conf']

            # probability loss
            valid = local_pts_loss.detach().mean(-1, keepdims=True) < self.expected_dist_thresh
            local_conf_loss = self.conf_loss_fn(pred_conf[valid_masks], valid.float())

            sky_mask = self.predict_sky_mask(gt['imgs'].reshape(B*N, 3, H, W)).reshape(B, N, H, W)
            sky_mask[valid_masks] = False
            if sky_mask.sum() == 0:
                sky_mask_loss = 0.0 * aligned_local_pts.mean()
            else:
                sky_mask_loss = self.conf_loss_fn(pred_conf[sky_mask], torch.zeros_like(pred_conf[sky_mask]))
            
            final_loss += 0.05 * (local_conf_loss + sky_mask_loss)
            details['local_conf_loss'] = (local_conf_loss + sky_mask_loss)

        final_loss += local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()

        # normal loss
        normal_batch_id = [i for i in range(len(gt['dataset_names'])) if gt['dataset_names'][i] in __HIGH_QUALITY_DATASETS__ + __MIDDLE_QUALITY_DATASETS__]
        if len(normal_batch_id) == 0:
            normal_loss =  0.0 * aligned_local_pts.mean()
        else:
            normal_loss = self.noraml_loss(aligned_local_pts[normal_batch_id], gt_local_pts[normal_batch_id], valid_masks[normal_batch_id])
            final_loss += normal_loss.mean()
        details['normal_loss'] = normal_loss.mean()

        # [Optional] Global Point Loss
        if 'global_points' in pred and pred['global_points'] is not None:
            gt_pts = gt['global_points']

            pred_global_pts = pred['global_points'] * S_opt_local.view(B, 1, 1, 1, 1)
            global_pts_loss = self.criteria_local(pred_global_pts[valid_masks].float(), gt_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

            final_loss += global_pts_loss.mean()
            details['global_pts_loss'] = global_pts_loss.mean()

        return final_loss, details, S_opt_local

# ---------------------------------------------------------------------------
# CameraLoss: Affine-invariant Camera Pose
# ---------------------------------------------------------------------------

class CameraLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:  
            R_err: rotation angular error 
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err.mean()         # [0, 3.14]
    
    def forward(self, pred, gt, scale):
        pred_pose = pred['camera_poses']
        gt_pose = gt['camera_poses']

        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
        pred_pose_align[..., :3, 3] *=  scale.view(B, 1, 1)
        
        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)
        
        pred_w2c_exp = pred_w2c.unsqueeze(2)
        pred_pose_exp = pred_pose_align.unsqueeze(1)
        
        gt_w2c_exp = gt_w2c.unsqueeze(2)
        gt_pose_exp = gt_pose.unsqueeze(1)
        
        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]
        
        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)
        
        rot_loss = self.rot_ang_loss(
            R_pred.reshape(-1, 3, 3), 
            R_gt.reshape(-1, 3, 3)
        )
        
        total_loss = self.alpha * trans_loss + rot_loss

        return total_loss, dict(trans_loss=trans_loss, rot_loss=rot_loss)

# ---------------------------------------------------------------------------
# Final Loss
# ---------------------------------------------------------------------------

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps=1e-6,
):
    """
    Compute the DICE loss with various strategies.
    Args:
        inputs: A float tensor of shape (B, V, H, W).
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs_sig = inputs.sigmoid()
    
    # Per-view loss calculation
    inputs_flat = inputs_sig.flatten(2)  # (B, V, H*W)
    targets_flat = targets.flatten(2)  # (B, V, H*W)
    numerator = 2 * (inputs_flat * targets_flat).sum(-1)  # (B, V)
    denominator = inputs_flat.sum(-1) + targets_flat.sum(-1)  # (B, V)
    per_view_loss = 1 - (numerator + eps) / (denominator + eps)  # (B, V)

    weights = torch.ones_like(per_view_loss)
    no_target_mask = (targets_flat.sum(-1) == 0)
    if no_target_mask.any():
        num_no_target_per_item = no_target_mask.sum(dim=1, keepdim=True)
        weights_for_no_target = 1.0 / torch.clamp(num_no_target_per_item, min=1)
        weights = torch.where(no_target_mask, weights_for_no_target, weights)
    per_view_loss = per_view_loss * weights
    
    final_loss = per_view_loss.mean()
    
    return final_loss


def iou_score(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps=1e-6,
):
    """
    Compute the IoU score
    Args:
        inputs: A float tensor of shape (B, V, H, W).
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    # binary
    inputs = (inputs > 0.5).float() # (B, V, H, W)
    # flatten H, W
    inputs = inputs.flatten(2)  # (B, V, H*W)
    targets = targets.flatten(2)  # (B, V, H*W)
    numerator = (inputs * targets).sum(-1)  # (B, V)
    denominator = inputs.sum(-1) + targets.sum(-1) - numerator  # (B, V)
    score = (numerator + eps) / (denominator + eps)  # (B, V)
    score = score * (targets.sum(-1) > 0).float() # (B, V)
    score = torch.where((targets.sum(-1) > 0).float().sum(-1) > 0, score.sum(-1) / (targets.sum(-1) > 0).float().sum(-1), torch.zeros_like(score[:, 0])) # (B)
    return score.mean()


def iou_score_global(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps=1e-6,
):
    """
    Compute the IoU score
    Args:
        inputs: A float tensor of arbitrary shape..
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid() # (B, V, H, W)
    # binary
    inputs = (inputs > 0.5).float() # (B, V, H, W)
    # flatten
    inputs = inputs.flatten(1) # (B, V*H*W)
    targets = targets.flatten(1) # (B, V*H*W)
    numerator = (inputs * targets).sum(-1)  # (B)
    denominator = inputs.sum(-1) + targets.sum(-1) - numerator  # (B)
    score = (numerator + eps) / (denominator + eps)  # (B)
    return score.mean()


def iou_score_global_per_sample(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps=1e-6,
):
    """
    Computes global IoU for each sample in the batch.
    Returns a tensor of shape (B,).
    """
    inputs = inputs.sigmoid()
    inputs = (inputs > 0.5).float()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1) - numerator
    score = (numerator + eps) / (denominator + eps)
    return score

def iou_score_per_view(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps=1e-6,
):
    """
    Computes IoU for each view for each sample in the batch.
    Returns a tensor of shape (B, V).
    """
    inputs = inputs.sigmoid()
    inputs = (inputs > 0.5).float()  # (B, V, H, W)
    # flatten H, W
    inputs_flat = inputs.flatten(2)  # (B, V, H*W)
    targets_flat = targets.flatten(2)  # (B, V, H*W)
    numerator = (inputs_flat * targets_flat).sum(-1)  # (B, V)
    denominator = inputs_flat.sum(-1) + targets_flat.sum(-1) - numerator  # (B, V)
    score_per_view = (numerator + eps) / (denominator + eps)  # (B, V)
    return score_per_view

class ReferringMaskLoss(nn.Module):
    def __init__(
        self,
        weight_dict=None,
        layer_weight=0.5,
        contrastive_temperature=0.1,
        contrastive_mode=None,
        contrastive_mask_ratio_threshold=0.2,
        contrastive_mask_ratio_metric="view_presence",
        contrastive_loss_version="ori",
        contrastive_pos_normalize=True,
    ):
        super().__init__()
        self.weight_dict = weight_dict if weight_dict is not None else {'loss_mask': 1, 'loss_dice': 1}
        self.layer_weight = layer_weight
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_mode = contrastive_mode
        self.contrastive_mask_ratio_threshold = contrastive_mask_ratio_threshold
        self.contrastive_mask_ratio_metric = contrastive_mask_ratio_metric
        self.contrastive_loss_version = contrastive_loss_version
        self.contrastive_pos_normalize = contrastive_pos_normalize

    @staticmethod
    def _metadata_to_str_list(values):
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().tolist()
        return [str(value) for value in values]

    @staticmethod
    def _metadata_to_int_list(values):
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().tolist()
        output = []
        for value in values:
            if isinstance(value, torch.Tensor):
                value = value.item()
            output.append(int(value))
        return output

    def _pool_text_features(self, text_features, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones(text_features.shape[:2], device=text_features.device)
        mask = attention_mask.to(device=text_features.device, dtype=text_features.dtype)
        denom = mask.sum(dim=1, keepdim=True)
        pooled = (text_features * mask.unsqueeze(-1)).sum(dim=1) / denom.clamp_min(1.0)
        valid = denom.squeeze(1) > 0
        return pooled, valid

    def _pool_anchor_patch_features(self, patch_features, gt_masks, patch_shape):
        B, V, _, _ = gt_masks.shape
        patch_h, patch_w = patch_shape
        mask_weights = F.interpolate(
            gt_masks.float().reshape(B * V, 1, *gt_masks.shape[-2:]),
            size=(patch_h, patch_w),
            mode='area',
        ).reshape(B, V, patch_h * patch_w)

        patch_features = patch_features.float()
        weight_sum = mask_weights.sum(dim=(1, 2))
        pooled = (patch_features * mask_weights.unsqueeze(-1)).sum(dim=(1, 2))
        pooled = pooled / weight_sum.clamp_min(1e-6).unsqueeze(-1)
        valid = weight_sum > 1e-6
        return pooled, valid

    def _multi_pos_cross_entropy(self, logits, pos_mask, neg_mask):
        losses = []
        pos_counts = []
        neg_counts = []
        for anchor_idx in range(logits.shape[0]):
            pos_logits = logits[anchor_idx][pos_mask[anchor_idx]]
            neg_logits = logits[anchor_idx][neg_mask[anchor_idx]]
            if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                continue

            diff = neg_logits[:, None] - pos_logits[None, :]
            diff = diff.reshape(-1)
            diff = torch.cat([diff, logits.new_zeros(1)], dim=0)
            losses.append(torch.logsumexp(diff, dim=0))
            pos_counts.append(float(pos_logits.numel()))
            neg_counts.append(float(neg_logits.numel()))

        if len(losses) == 0:
            zero = logits.sum() * 0.0
            return zero, logits.new_tensor(0.0), logits.new_tensor(0.0), logits.new_tensor(0.0)

        loss = torch.stack(losses).mean()
        valid_anchor_rate = logits.new_tensor(len(losses) / logits.shape[0])
        mean_pos_count = logits.new_tensor(sum(pos_counts) / len(pos_counts))
        mean_neg_count = logits.new_tensor(sum(neg_counts) / len(neg_counts))
        return loss, valid_anchor_rate, mean_pos_count, mean_neg_count

    def _scene_object_contrastive_loss_for_anchor(self, logits, pos_mask, neg_mask):
        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            return None

        if self.contrastive_loss_version == "ori":
            diff = neg_logits[:, None] - pos_logits[None, :]
            diff = diff.reshape(-1)
            diff = torch.cat([diff, logits.new_zeros(1)], dim=0)
            return torch.logsumexp(diff, dim=0)

        if self.contrastive_loss_version == "unbiased":
            stable_logits = logits - logits.max().detach()
            exp_logits = torch.exp(stable_logits)
            if self.contrastive_pos_normalize:
                pos_weights = pos_mask.to(dtype=logits.dtype)
                pos_weights = pos_weights / pos_weights.sum().clamp_min(1.0)
                exp_logits_input = (exp_logits * pos_weights).sum() + exp_logits[neg_mask].sum()
            else:
                exp_logits_input = exp_logits[pos_mask | neg_mask].sum()
            log_prob = stable_logits - torch.log(exp_logits_input.clamp_min(1e-12))
            return -(log_prob[pos_mask].sum() / pos_mask.sum().clamp_min(1))

        if self.contrastive_loss_version == "set_infonce":
            stable_logits = logits - logits.max().detach()
            log_pos = torch.logsumexp(stable_logits[pos_mask], dim=0)
            log_all = torch.logsumexp(stable_logits[pos_mask | neg_mask], dim=0)
            return -(log_pos - log_all)

        raise ValueError(f"Unknown contrastive_loss_version: {self.contrastive_loss_version}")

    def _cross_text_mask_ratios(self, instance_maps, scene_ids, object_ids):
        B, V, H, W = instance_maps.shape
        ratios = instance_maps.new_zeros((B, B), dtype=torch.float32)
        same_scene = torch.zeros((B, B), device=instance_maps.device, dtype=torch.bool)

        for anchor_idx in range(B):
            for text_idx in range(B):
                if scene_ids[anchor_idx] != scene_ids[text_idx]:
                    continue
                same_scene[anchor_idx, text_idx] = True
                object_instance_id = int(object_ids[text_idx]) + 1
                object_mask = instance_maps[anchor_idx] == object_instance_id
                if self.contrastive_mask_ratio_metric == "view_presence":
                    ratios[anchor_idx, text_idx] = object_mask.flatten(1).any(dim=1).float().mean()
                elif self.contrastive_mask_ratio_metric == "pixel_area":
                    ratios[anchor_idx, text_idx] = object_mask.float().mean()
                else:
                    raise ValueError(
                        f"Unknown contrastive_mask_ratio_metric: {self.contrastive_mask_ratio_metric}"
                    )

        return ratios, same_scene

    def _patch_anchor_text_mask_ratio_loss(self, pred, gt_masks, instance_maps, text_info):
        required_keys = (
            'contrastive_patch_features',
            'contrastive_patch_shape',
            'contrastive_text_features',
            'contrastive_attention_mask',
        )
        if instance_maps is None or text_info is None or any(key not in pred or pred[key] is None for key in required_keys):
            zero = gt_masks.float().sum() * 0.0
            return zero, {
                'contrastive_valid_anchor_rate': zero.detach(),
                'contrastive_mean_pos_count': zero.detach(),
                'contrastive_mean_neg_count': zero.detach(),
                'contrastive_mean_pos_mask_ratio': zero.detach(),
                'contrastive_mean_neg_mask_ratio': zero.detach(),
            }

        patch_features = pred['contrastive_patch_features']
        patch_shape = pred['contrastive_patch_shape']
        text_features = pred['contrastive_text_features']
        attention_mask = pred['contrastive_attention_mask']

        anchor_features, valid_anchor = self._pool_anchor_patch_features(patch_features, gt_masks, patch_shape)
        text_features, valid_text = self._pool_text_features(text_features, attention_mask)
        anchor_features = F.normalize(anchor_features, dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)
        logits = anchor_features @ text_features.T
        logits = logits / self.contrastive_temperature

        B = logits.shape[0]
        scene_ids = self._metadata_to_str_list(text_info.get('scene_id'))
        object_ids = self._metadata_to_int_list(text_info.get('object_id'))
        if scene_ids is None or object_ids is None or len(scene_ids) != B or len(object_ids) != B:
            zero = logits.sum() * 0.0
            return zero, {
                'contrastive_valid_anchor_rate': zero.detach(),
                'contrastive_mean_pos_count': zero.detach(),
                'contrastive_mean_neg_count': zero.detach(),
                'contrastive_mean_pos_mask_ratio': zero.detach(),
                'contrastive_mean_neg_mask_ratio': zero.detach(),
            }

        ratios, same_scene = self._cross_text_mask_ratios(instance_maps.long(), scene_ids, object_ids)
        valid_pairs = valid_anchor[:, None] & valid_text[None, :] & same_scene
        pos_mask = (ratios > self.contrastive_mask_ratio_threshold) & valid_pairs
        neg_mask = (ratios < self.contrastive_mask_ratio_threshold) & valid_pairs

        loss, valid_anchor_rate, mean_pos_count, mean_neg_count = self._multi_pos_cross_entropy(
            logits,
            pos_mask,
            neg_mask,
        )
        zero = logits.sum() * 0.0
        pos_ratios = ratios[pos_mask]
        neg_ratios = ratios[neg_mask]
        return loss, {
            'contrastive_valid_anchor_rate': valid_anchor_rate.detach(),
            'contrastive_mean_pos_count': mean_pos_count.detach(),
            'contrastive_mean_neg_count': mean_neg_count.detach(),
            'contrastive_mean_pos_mask_ratio': (pos_ratios.mean() if pos_ratios.numel() > 0 else zero).detach(),
            'contrastive_mean_neg_mask_ratio': (neg_ratios.mean() if neg_ratios.numel() > 0 else zero).detach(),
        }

    @staticmethod
    def _ragged_item(values, idx):
        if values is None:
            return None
        if torch.is_tensor(values):
            return values[idx]
        return values[idx]

    def _patch_anchor_scene_object_text_loss(self, pred, gt_masks, text_info):
        required_keys = (
            'contrastive_patch_features',
            'contrastive_patch_shape',
            'contrastive_scene_text_features',
        )
        zero = gt_masks.float().sum() * 0.0
        if text_info is None or any(key not in pred or pred[key] is None for key in required_keys):
            return zero, {
                'contrastive_valid_anchor_rate': zero.detach(),
                'contrastive_mean_pos_count': zero.detach(),
                'contrastive_mean_neg_count': zero.detach(),
                'contrastive_mean_scene_text_count': zero.detach(),
                'contrastive_mean_pos_rank': zero.detach(),
                'contrastive_pos_neg_pair_accuracy': zero.detach(),
            }

        scene_text_object_ids = text_info.get('scene_text_object_ids')
        object_ids = self._metadata_to_int_list(text_info.get('object_id'))
        if scene_text_object_ids is None or object_ids is None:
            return zero, {
                'contrastive_valid_anchor_rate': zero.detach(),
                'contrastive_mean_pos_count': zero.detach(),
                'contrastive_mean_neg_count': zero.detach(),
                'contrastive_mean_scene_text_count': zero.detach(),
                'contrastive_mean_pos_rank': zero.detach(),
                'contrastive_pos_neg_pair_accuracy': zero.detach(),
            }

        patch_features = pred['contrastive_patch_features']
        patch_shape = pred['contrastive_patch_shape']
        scene_text_features = pred['contrastive_scene_text_features']
        anchor_features, valid_anchor = self._pool_anchor_patch_features(patch_features, gt_masks, patch_shape)
        anchor_features = F.normalize(anchor_features, dim=-1)

        losses = []
        pos_counts = []
        neg_counts = []
        scene_text_counts = []
        pos_ranks = []
        pair_accuracies = []
        for anchor_idx in range(anchor_features.shape[0]):
            if not bool(valid_anchor[anchor_idx].item()):
                continue

            text_features = self._ragged_item(scene_text_features, anchor_idx)
            text_object_ids = self._ragged_item(scene_text_object_ids, anchor_idx)
            if text_features is None or text_object_ids is None:
                continue

            text_features = text_features.to(device=anchor_features.device, dtype=anchor_features.dtype)
            text_object_ids = text_object_ids.to(device=anchor_features.device, dtype=torch.long)
            if text_features.numel() == 0 or text_object_ids.numel() == 0:
                continue

            target_object_id = int(object_ids[anchor_idx])
            text_features = F.normalize(text_features.float(), dim=-1)
            logits = anchor_features[anchor_idx].float() @ text_features.T
            logits = logits / self.contrastive_temperature

            pos_mask = text_object_ids == target_object_id
            neg_mask = text_object_ids != target_object_id
            pos_logits = logits[pos_mask]
            neg_logits = logits[neg_mask]
            if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                continue

            pos_rank = 1.0 + (logits[None, :] > pos_logits[:, None]).float().sum(dim=1)
            pos_neg_pair_accuracy = (pos_logits[:, None] > neg_logits[None, :]).float().mean()
            contrastive_loss = self._scene_object_contrastive_loss_for_anchor(logits, pos_mask, neg_mask)
            if contrastive_loss is None:
                continue
            losses.append(contrastive_loss)
            pos_counts.append(float(pos_logits.numel()))
            neg_counts.append(float(neg_logits.numel()))
            scene_text_counts.append(float(text_features.shape[0]))
            pos_ranks.append(float(pos_rank.mean().item()))
            pair_accuracies.append(float(pos_neg_pair_accuracy.item()))

        if len(losses) == 0:
            return zero, {
                'contrastive_valid_anchor_rate': zero.detach(),
                'contrastive_mean_pos_count': zero.detach(),
                'contrastive_mean_neg_count': zero.detach(),
                'contrastive_mean_scene_text_count': zero.detach(),
                'contrastive_mean_pos_rank': zero.detach(),
                'contrastive_pos_neg_pair_accuracy': zero.detach(),
            }

        loss = torch.stack(losses).mean()
        return loss, {
            'contrastive_valid_anchor_rate': loss.new_tensor(len(losses) / anchor_features.shape[0]).detach(),
            'contrastive_mean_pos_count': loss.new_tensor(sum(pos_counts) / len(pos_counts)).detach(),
            'contrastive_mean_neg_count': loss.new_tensor(sum(neg_counts) / len(neg_counts)).detach(),
            'contrastive_mean_scene_text_count': loss.new_tensor(sum(scene_text_counts) / len(scene_text_counts)).detach(),
            'contrastive_mean_pos_rank': loss.new_tensor(sum(pos_ranks) / len(pos_ranks)).detach(),
            'contrastive_pos_neg_pair_accuracy': loss.new_tensor(sum(pair_accuracies) / len(pair_accuracies)).detach(),
        }

    def forward(self, pred, gt, text_info=None, current_epoch=None, total_epochs=None):
        pred_masks = pred['referring_mask_pred']
        gt_masks = gt['referring_masks'] # (B, V, H, W)
        
        num_masks = gt_masks.shape[0] * gt_masks.shape[1]

        losses = {}
        losses["loss_mask"] = F.binary_cross_entropy_with_logits(pred_masks, gt_masks.float())
        losses["loss_dice"] = dice_loss(pred_masks, gt_masks)
        losses["iou_score"] = iou_score_global(pred_masks, gt_masks)
        iou_global_per_sample = iou_score_global_per_sample(pred_masks, gt_masks)
        iou_per_view = iou_score_per_view(pred_masks, gt_masks)
        # iou score just in frame with target
        losses["iou_score_in_frame_with_target"] = iou_score(pred_masks, gt_masks)
        total_loss = self.weight_dict['loss_mask'] * losses['loss_mask'] + self.weight_dict['loss_dice'] * losses['loss_dice']
        contrastive_weight = self.weight_dict.get('loss_contrastive', 0.0)
        if contrastive_weight > 0:
            if self.contrastive_mode == "patch_anchor_text_mask_ratio":
                contrastive_loss, contrastive_details = self._patch_anchor_text_mask_ratio_loss(
                    pred,
                    gt_masks,
                    gt.get('instance_maps'),
                    text_info,
                )
            elif self.contrastive_mode == "patch_anchor_scene_object_text":
                contrastive_loss, contrastive_details = self._patch_anchor_scene_object_text_loss(
                    pred,
                    gt_masks,
                    text_info,
                )
            else:
                raise ValueError(f"Unknown contrastive_mode: {self.contrastive_mode}")

            losses["loss_contrastive"] = contrastive_loss
            losses.update(contrastive_details)
            total_loss += contrastive_weight * contrastive_loss
        # Record the proportion of samples without a target, i.e., the proportion of samples where all views have no target.
        view_has_target = (gt_masks.sum(dim=(-1, -2)) > 0).float() # (B, V)
        sample_no_target = (view_has_target.sum(-1) == 0).float() # (B)
        losses["rate_no_target"] = sample_no_target.mean()

        # Record the average ratio of frames with a target to the total number of frames in each sample.
        view_has_target = gt_masks.sum(dim=(-1, -2)) > 0 # (B, V)
        rate_view_has_target = view_has_target.float().mean(-1) # (B)
        losses["rate_frame_with_target"] = rate_view_has_target.mean()

        # Record the pixel proportion of the target in each sample.
        pixel_rate_per_view = gt_masks.float().mean(dim=(-1, -2)) # (B, V)
        losses["rate_pixel_with_target"] = pixel_rate_per_view.mean()

        # Record the average pixel proportion of the target in frames that contain the target.
        if view_has_target.any():
            rate_pixel_in_target_frame = pixel_rate_per_view[view_has_target].mean()
        else:
            rate_pixel_in_target_frame = torch.tensor(0.0, device=gt_masks.device)
        losses["rate_pixel_in_target_frame"] = rate_pixel_in_target_frame
        # If there are prediction results for intermediate layers, calculate their losses.
        if 'layer_referring_mask_preds' in pred:
            layer_preds = pred['layer_referring_mask_preds']
            layer_losses = {}
            for i, layer_pred in enumerate(layer_preds):
                layer_losses[f"loss_mask_layer_{i}"] = F.binary_cross_entropy_with_logits(layer_pred, gt_masks.float())
                layer_losses[f"loss_dice_layer_{i}"] = dice_loss(layer_pred, gt_masks)
                layer_losses[f"iou_score_layer_{i}"] = iou_score_global(layer_pred, gt_masks)

                # Add intermediate layer losses to the total loss, multiplied by a weight coefficient.
                total_loss += self.layer_weight * (
                    self.weight_dict['loss_mask'] * layer_losses[f"loss_mask_layer_{i}"] + 
                    self.weight_dict['loss_dice'] * layer_losses[f"loss_dice_layer_{i}"]
                )
            
            # Add intermediate layer losses to details.
            losses.update(layer_losses)
        
        # Save detached versions of all losses for logging
        details = {f"refer_{k}": v.detach() for k, v in losses.items()}
        details['refer_iou_score_global_per_sample'] = iou_global_per_sample.detach()
        details['refer_iou_per_view'] = iou_per_view.detach()
        return total_loss, details

class MVGGTLoss(nn.Module):
    def __init__(
        self,
        train_conf=False,
        use_referring_segmentation=False,
        referring_loss_weight_dict=None,
        referring_layer_weight=0.5,
        contrastive_temperature=0.1,
        contrastive_mode=None,
        contrastive_mask_ratio_threshold=0.2,
        contrastive_mask_ratio_metric="view_presence",
        contrastive_loss_version="ori",
        contrastive_pos_normalize=True,
    ):
        super().__init__()
        self.point_loss = PointLoss(train_conf=train_conf)
        self.camera_loss = CameraLoss()
        self.use_referring_segmentation = use_referring_segmentation
        if self.use_referring_segmentation:
            self.referring_mask_loss = ReferringMaskLoss(
                weight_dict=referring_loss_weight_dict,
                layer_weight=referring_layer_weight,
                contrastive_temperature=contrastive_temperature,
                contrastive_mode=contrastive_mode,
                contrastive_mask_ratio_threshold=contrastive_mask_ratio_threshold,
                contrastive_mask_ratio_metric=contrastive_mask_ratio_metric,
                contrastive_loss_version=contrastive_loss_version,
                contrastive_pos_normalize=contrastive_pos_normalize,
            )

    def prepare_gt(self, gt):
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)
        if self.use_referring_segmentation and gt[0]['referring_mask'] is not None:
            referring_masks = torch.stack([view['referring_mask'] for view in gt], dim=1)
        instance_maps = None
        if self.use_referring_segmentation and 'instance_map' in gt[0] and gt[0]['instance_map'] is not None:
            instance_maps = torch.stack([view['instance_map'] for view in gt], dim=1)

        B, N, H, W, _ = gt_pts.shape

        # transform to first frame camera coordinate
        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        # normalize points
        valid_batch = masks.sum([-1, -2, -3]) > 0
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]
        
        dataset_names = gt[0]['dataset']

        return dict(
            imgs = torch.stack([view['img'] for view in gt], dim=1),
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            camera_poses=poses,
            referring_masks=referring_masks if self.use_referring_segmentation else None,
            instance_maps=instance_maps,
            dataset_names=dataset_names
        )
    
    def normalize_pred(self, pred, gt):
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']

        # normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points  = local_points / norm_factor[..., None, None, None, None]

        if 'global_points' in pred and pred['global_points'] is not None:
            pred['global_points'] /= norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        return pred

    def forward(self, pred, gt_raw, text_info=None, current_epoch=None, total_epochs=None):
        gt = self.prepare_gt(gt_raw)
        pred = self.normalize_pred(pred, gt)

        final_loss = 0.0
        details = dict()

        # Local Point Loss
        point_loss, point_loss_details, scale = self.point_loss(pred, gt)
        final_loss += point_loss if not self.use_referring_segmentation else 0.0
        details.update(point_loss_details)

        # Camera Loss
        camera_loss, camera_loss_details = self.camera_loss(pred, gt, scale)
        final_loss += camera_loss * 0.1 if not self.use_referring_segmentation else 0.0
        details.update(camera_loss_details)

        if self.use_referring_segmentation and 'referring_mask_pred' in pred and 'referring_masks' in gt:
            referring_loss, referring_loss_details = self.referring_mask_loss(
                pred,
                gt,
                text_info=text_info,
                current_epoch=current_epoch,
                total_epochs=total_epochs,
            )
            final_loss += referring_loss
            details.update(referring_loss_details)

        return final_loss, details
