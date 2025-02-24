import torch
import numpy as np

def compute_depth_metrics(pred, target, mask=None, min_depth=1e-3, max_depth=80):
    """
    Compute depth estimation metrics with numerical stability handling
    pred, target: torch tensors of same shape
    mask: optional binary mask for valid pixels
    """
    # Ensure positive depth values
    valid_mask = (target > min_depth) & (target < max_depth)
    if mask is not None:
        valid_mask = valid_mask & mask
    
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    # Clip predicted depth to valid range
    pred = torch.clamp(pred, min=min_depth, max=max_depth)
    
    # Compute threshold matrix
    thresh = torch.max((target / pred), (pred / target))
    
    # Accuracy metrics
    delta1 = (thresh < 1.25).float().mean()
    delta2 = (thresh < 1.25 ** 2).float().mean()
    delta3 = (thresh < 1.25 ** 3).float().mean()
    
    # Error metrics with numerical stability
    rmse = torch.sqrt(((target - pred) ** 2).mean() + 1e-6)  # Add small epsilon
    
    # Safe log computation
    safe_log_pred = torch.log(pred + 1e-6)
    safe_log_target = torch.log(target + 1e-6)
    
    rmse_log = torch.sqrt(((safe_log_target - safe_log_pred) ** 2).mean() + 1e-6)
    
    # Absolute relative difference with epsilon
    abs_diff = torch.abs(target - pred)
    rel_diff = abs_diff / (target + 1e-6)
    abs_rel = rel_diff.mean()
    
    # Square relative difference with epsilon
    sq_rel = (((target - pred) ** 2) / (target + 1e-6)).mean()
    
    # Scale-invariant logarithmic error
    d = safe_log_pred - safe_log_target
    silog = torch.sqrt(torch.mean(d ** 2) - (torch.mean(d) ** 2) + 1e-6) * 100
    
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