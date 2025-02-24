import torch
import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel
import torch.nn.functional as F

class ResNetDepth(BaseModel):
    def __init__(self):
        super().__init__()
        torch.backends.cudnn.benchmark = True
        
        # Modified initialization with smaller decoder dimensions
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),  # 64 channels
            nn.Sequential(resnet.maxpool, resnet.layer1),          # 256 channels
            resnet.layer2,                                         # 512 channels
            resnet.layer3,                                         # 1024 channels
            resnet.layer4,                                         # 2048 channels
        ])
        
        # Decoder with skip connections
        self.decoder = nn.ModuleList([
            self._make_decoder_layer(2048, 1024),
            self._make_decoder_layer(1024, 512),
            self._make_decoder_layer(512, 256),
            self._make_decoder_layer(256, 64),
            nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1)
            )
        ])

    def _make_decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

    def forward(self, x):
        # Store original size
        orig_size = x.shape[-2:]
        
        # Encoder
        features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            features.append(x)
        
        # Decoder with skip connections
        for i, decoder_layer in enumerate(self.decoder[:-1]):
            x = decoder_layer(x)
            if i < len(features) - 1:
                x = x + F.interpolate(features[-(i+2)], size=x.shape[-2:], mode='bilinear', align_corners=True)
        
        # Final layer
        x = self.decoder[-1](x)
        
        # Improved depth scaling with better numerical stability
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=True)
        x = F.relu(x) + 1e-7  # Ensure positive values
        x = torch.clamp(x, min=1e-3, max=80.0)  # Clip to valid range
        
        return x

    def compute_loss(self, pred, target):
        """Improved multi-scale loss computation with proper tensor dimensions"""
        # Save original shapes for reshaping
        original_shape = pred.shape
        
        # Ensure proper dimensions (B, 1, H, W)
        if len(original_shape) == 3:
            pred = pred.unsqueeze(1)
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
            
        # Valid depth mask
        valid_mask = (target > 1e-3) & (target < 80.0) & torch.isfinite(target)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Basic L1 loss on masked values
        l1_loss = torch.abs(pred[valid_mask] - target[valid_mask]).mean()

        # Scale-invariant loss on masked values
        d = torch.log(pred[valid_mask] + 1e-6) - torch.log(target[valid_mask] + 1e-6)
        si_loss = torch.sqrt((d ** 2).mean() - 0.5 * (d.mean() ** 2) + 1e-6)

        # Edge-awareness loss (compute before masking)
        grad_pred_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        grad_pred_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        grad_target_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        grad_target_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        # Apply mask to gradients
        edge_mask_x = valid_mask[:, :, :, 1:] & valid_mask[:, :, :, :-1]
        edge_mask_y = valid_mask[:, :, 1:, :] & valid_mask[:, :, :-1, :]
        
        edge_loss = (torch.abs(grad_pred_x[edge_mask_x] - grad_target_x[edge_mask_x]).mean() + 
                     torch.abs(grad_pred_y[edge_mask_y] - grad_target_y[edge_mask_y]).mean())

        # Combined weighted loss
        loss = (0.45 * l1_loss + 
                0.45 * si_loss + 
                0.10 * edge_loss)

        return loss

    def _gradient(self, x):
        """Compute gradients for smoothness loss"""
        grad_x = torch.abs(x[..., :, 1:] - x[..., :, :-1])
        grad_y = torch.abs(x[..., 1:, :] - x[..., :-1, :])
        return grad_x, grad_y
