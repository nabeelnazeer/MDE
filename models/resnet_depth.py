import torch
import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel
import torch.nn.functional as F

class ResNetDepth(BaseModel):
    def __init__(self):
        super().__init__()
        # Encoder - update ResNet initialization
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv5 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Store original size for resizing later
        orig_size = x.shape[-2:]
        
        # Encoder
        x = self.encoder(x)
        
        # Decoder
        x = self.relu(self.upconv1(x))
        x = self.relu(self.upconv2(x))
        x = self.relu(self.upconv3(x))
        x = self.relu(self.upconv4(x))
        x = self.upconv5(x)
        
        # Resize output to match input size
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=True)
        
        # Ensure positive depth predictions
        x = F.relu(x) + 1e-6  # Add small epsilon to avoid zero depths
        
        return x
    
    def compute_loss(self, pred, target):
        """Compute L1 loss after ensuring sizes match and values are valid"""
        if pred.shape != target.shape:
            # Resize prediction to match target size
            pred = F.interpolate(pred, size=target.shape[-2:], 
                               mode='bilinear', align_corners=True)
        
        # Create mask for valid target depths
        valid_mask = (target > 1e-3) & (target < 80)
        
        # Apply mask to both prediction and target
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        return nn.L1Loss()(pred, target)
