import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.kitti_dataset import KITTIDataset
from models.resnet_depth import ResNetDepth
from utils.metrics import compute_depth_metrics
from tqdm import tqdm
from collections import defaultdict

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    return parser.parse_args()

def log_metrics(metrics, use_wandb, wandb=None):
    # Print metrics
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Log to wandb if enabled
    if use_wandb:
        wandb.log(metrics)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    metrics_list = []
    
    with torch.no_grad():
        for images, depths in dataloader:
            images, depths = images.to(device), depths.to(device)
            outputs = model(images)
            loss = model.compute_loss(outputs, depths)
            total_loss += loss.item()
            metrics_list.append(compute_depth_metrics(outputs, depths))
    
    mean_metrics = {}
    for key in metrics_list[0].keys():
        mean_metrics[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
    mean_metrics['val_loss'] = total_loss / len(dataloader)
    
    return mean_metrics

def collate_fn(batch):
    """Custom collate function to handle variable sized images/depths"""
    images = torch.stack([item[0] for item in batch])
    depths = torch.stack([item[1] for item in batch])
    return images, depths

def print_metrics_summary(metrics, prefix=""):
    """Helper function to print metrics in a formatted way"""
    print(f"\n{prefix} Metrics:")
    print("-" * 50)
    print(f"Error Metrics:")
    print(f"Abs Rel: {metrics['abs_rel']:.4f}")
    print(f"Sq Rel: {metrics['sq_rel']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"RMSE log: {metrics['rmse_log']:.4f}")
    print(f"SiLog: {metrics['silog']:.4f}")
    print(f"\nAccuracy Metrics (δ):")
    print(f"δ < 1.25: {metrics['delta1']:.4f}")
    print(f"δ < 1.25²: {metrics['delta2']:.4f}")
    print(f"δ < 1.25³: {metrics['delta3']:.4f}")
    print("-" * 50)

def print_detailed_metrics(metrics, prefix=""):
    print(f"\n{prefix} Metrics:")
    print("=" * 60)
    print("Error Metrics:")
    print(f"{'Metric':<15} {'Value':>10}")
    print("-" * 60)
    print(f"{'Abs Rel':<15} {metrics['abs_rel']:>10.4f}")
    print(f"{'Sq Rel':<15} {metrics['sq_rel']:>10.4f}")
    print(f"{'RMSE':<15} {metrics['rmse']:>10.4f}")
    print(f"{'RMSE log':<15} {metrics['rmse_log']:>10.4f}")
    print(f"{'SiLog':<15} {metrics['silog']:>10.4f}")
    print("\nAccuracy Metrics:")
    print(f"{'δ < 1.25':<15} {metrics['delta1']:>10.4f}")
    print(f"{'δ < 1.25²':<15} {metrics['delta2']:>10.4f}")
    print(f"{'δ < 1.25³':<15} {metrics['delta3']:>10.4f}")
    print("=" * 60)

def train_one_epoch(model, train_loader, optimizer, device, config, epoch, use_wandb=False, wandb=None):
    model.train()
    epoch_loss = 0
    epoch_metrics = defaultdict(float)
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
    for batch_idx, (images, depths) in enumerate(progress_bar):
        images, depths = images.to(device), depths.to(device)
        
        # Skip invalid batches
        if not torch.isfinite(depths).all():
            print(f"Warning: Invalid depth values in batch {batch_idx}, skipping...")
            continue
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Check for valid predictions
        if not torch.isfinite(outputs).all():
            print(f"Warning: Invalid predictions in batch {batch_idx}, skipping...")
            continue
        
        loss = model.compute_loss(outputs, depths)
        
        # Check for valid loss
        if not torch.isfinite(loss):
            print(f"Warning: Invalid loss in batch {batch_idx}, skipping...")
            continue
            
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        batch_metrics = compute_depth_metrics(outputs, depths)
        epoch_loss += loss.item()
        for k, v in batch_metrics.items():
            epoch_metrics[k] += v
            
        # Only show loss in progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}'
        })
    
    # Calculate epoch averages
    epoch_loss /= num_batches
    for k in epoch_metrics:
        epoch_metrics[k] /= num_batches
    
    return epoch_loss, epoch_metrics

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize device
    device = get_device()
    print(f"Using device: {device}")
    
    # Check both config and args for wandb usage
    use_wandb = not args.no_wandb and config.get('logging', {}).get('use_wandb', True)
    wandb_instance = None
    
    if use_wandb:
        import wandb
        wandb_instance = wandb
        wandb_instance.init(project=config['logging']['wandb_project'])
    
    # Create output directories
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((config['data']['img_height'], config['data']['img_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset without logging
    dataset = KITTIDataset(
        config['data']['train_path'], 
        transform=transform,
        height=config['data']['img_height'],
        width=config['data']['img_width'],
        max_samples=config['data'].get('max_samples', None)
    )
    
    # Split dataset into train and validation (80-20)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=config['training']['batch_size'],
                            shuffle=True, 
                            num_workers=4,
                            collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, 
                          batch_size=config['training']['batch_size'],
                          shuffle=False, 
                          num_workers=4,
                          collate_fn=collate_fn)
    
    # Create model and move to appropriate device
    model = ResNetDepth().to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=config['training']['learning_rate'],
                               weight_decay=config['training']['weight_decay'])
    
    print(f"\nTraining on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples")
    print(f"Number of batches per epoch: {len(train_loader)}\n")
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{config['training']['num_epochs']}]")
        
        # Disable pair logging after first epoch
        if epoch == 0:
            print("\nFirst epoch - showing image pairs:")
        dataset.log_pairs = (epoch == 0)
        
        # Training phase
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, config, epoch, use_wandb, wandb_instance)
        
        # Validation phase
        val_metrics = validate(model, val_loader, device)
        
        # Only print detailed metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("\n" + "=" * 80)
            print(f"Detailed Metrics at Epoch {epoch+1}")
            print("=" * 80)
            print_detailed_metrics(train_metrics, prefix="Training")
            print_detailed_metrics(val_metrics, prefix="Validation")
        else:
            # Print minimal progress info for other epochs
            print(f"Epoch {epoch+1} - Loss: {train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}")
        
        if use_wandb:
            wandb_instance.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            })
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            print(f"\nSaving best model with validation loss: {best_val_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'metrics': val_metrics,  # Save metrics with checkpoint
            }, f"{config['logging']['save_dir']}/best_model.pth")

        # Save checkpoint with metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, f"{config['logging']['save_dir']}/model_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    main()