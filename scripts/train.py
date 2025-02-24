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
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA. No GPU detected!")
    
    device = torch.device('cuda')
    print(f"\nUsing CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    return device

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
    valid_batches = 0
    
    with torch.no_grad():
        for images, depths in dataloader:
            images, depths = images.to(device), depths.to(device)
            
            # Skip batches with invalid depths
            if not torch.isfinite(depths).all():
                continue
                
            outputs = model(images)
            
            # Skip invalid predictions
            if not torch.isfinite(outputs).all():
                continue
                
            loss = model.compute_loss(outputs, depths)
            
            # Only accumulate valid losses
            if torch.isfinite(loss):
                total_loss += loss.item()
                metrics = compute_depth_metrics(outputs, depths)
                
                # Check if all metric values are finite
                metric_values = torch.tensor([float(v) for v in metrics.values()])
                if torch.isfinite(metric_values).all():
                    metrics_list.append(metrics)
                    valid_batches += 1
    
    # Return default values if no valid batches
    if valid_batches == 0 or not metrics_list:
        print("Warning: No valid batches in validation!")
        return {
            'val_loss': float('inf'),
            'abs_rel': float('inf'),
            'sq_rel': float('inf'),
            'rmse': float('inf'),
            'rmse_log': float('inf'),
            'silog': float('inf'),
            'delta1': 0.0,
            'delta2': 0.0,
            'delta3': 0.0
        }
    
    # Calculate mean metrics
    mean_metrics = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        mean_metrics[key] = sum(values) / len(values)
    
    mean_metrics['val_loss'] = total_loss / valid_batches
    
    return mean_metrics

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

def main():
    args = parse_args()
    
    # Force CUDA device initialization first
    device = get_device()
    print(f"Initial CUDA memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize device first
    device = get_device()
    if device.type == 'cuda':
        print(f"Initial CUDA memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    
    # Update dataset paths
    train_path = os.path.join(config['data']['base_path'], 'training')
    test_path = os.path.join(config['data']['base_path'], 'testing')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training path not found: {train_path}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((config['data']['img_height'], config['data']['img_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create training dataset using all samples
    train_dataset = KITTIDataset(
        train_path,
        transform=transform,
        height=config['data']['img_height'],
        width=config['data']['img_width'],
        max_samples=None,  # Ensure we use all samples
        split='train'
    )

    # Use a portion of training data for validation instead of testing data
    # since testing data might not have ground truth
    total_size = len(train_dataset)
    train_size = int(0.9 * total_size)  # Use 90% for training
    val_size = total_size - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

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
    
    # Create model and explicitly move to CUDA
    model = ResNetDepth()
    model = model.cuda()  # Force CUDA
    print(f"Model is on CUDA: {next(model.parameters()).is_cuda}")
    print(f"CUDA memory after model init: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=config['training']['learning_rate'],
                               weight_decay=config['training']['weight_decay'])
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['training']['scheduler_step_size'],
        gamma=config['training']['scheduler_gamma']
    )
    
    print(f"\nTraining on {len(train_subset)} samples")
    print(f"Validating on {len(val_subset)} samples")
    print(f"Number of training batches per epoch: {len(train_loader)}\n")
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['training']['num_epochs']):
        # Clear CUDA cache at start of epoch
        torch.cuda.empty_cache()
        
        print(f"\nEpoch [{epoch+1}/{config['training']['num_epochs']}]")
        
        # Disable pair logging after first epoch
        if epoch == 0:
            print("\nFirst epoch - showing image pairs:")
        train_dataset.log_pairs = (epoch == 0)
        
        # Training phase
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, config, epoch, use_wandb, wandb_instance)
        
        # Validation phase
        val_metrics = validate(model, val_loader, device)
        
        # Step the scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
        
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