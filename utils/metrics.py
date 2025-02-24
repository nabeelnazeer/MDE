import torch
import torch.nn.functional as F

def compute_depth_metrics(pred, target, min_depth=1e-3, max_depth=80.0, eps=1e-7):
    """Enhanced depth metrics computation with better numerical stability"""
    mask = (target > min_depth) & (target < max_depth) & torch.isfinite(target)
    if not mask.any():
        return {k: float('inf') for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'silog', 'delta1', 'delta2', 'delta3']}
    
    pred = torch.clamp(pred[mask], min_depth, max_depth)
    target = target[mask]
    
    thresh = 1.25
    
    # Relative error metrics (with numerical stability)
    abs_rel = torch.mean(torch.abs(target - pred) / (target + eps))
    sq_rel = torch.mean(((target - pred) ** 2) / (target + eps))
    
    # RMSE metrics
    rmse = torch.sqrt(torch.mean((target - pred) ** 2) + eps)
    rmse_log = torch.sqrt(torch.mean((torch.log(target + eps) - torch.log(pred + eps)) ** 2) + eps)
    
    # Scale-invariant log error
    d = torch.log(pred + eps) - torch.log(target + eps)
    silog = torch.sqrt(torch.mean(d ** 2) - 0.5 * (torch.mean(d) ** 2) + eps)
    
    # Threshold accuracies
    max_ratio = torch.max(pred / (target + eps), target / (pred + eps))
    delta1 = torch.mean((max_ratio < thresh).float())
    delta2 = torch.mean((max_ratio < thresh ** 2).float())
    delta3 = torch.mean((max_ratio < thresh ** 3).float())

    return {
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'rmse_log': rmse_log.item(),
        'silog': silog.item(),
        'delta1': delta1.item(),
        'delta2': delta2.item(),
        'delta3': delta3.item()
    }