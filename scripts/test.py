import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from data.kitti_dataset import KITTIDataset
from models.resnet_depth import ResNetDepth
from utils.metrics import compute_depth_metrics
from visualization.visualizer import DepthVisualizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")
    
    # Load model
    model = ResNetDepth().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}")
    
    # Setup test dataset
    transform = transforms.Compose([
        transforms.Resize((config['data']['img_height'], config['data']['img_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = KITTIDataset(
        root_dir="/Users/nabeelnazeer/Documents/Project-s6/Datasets/data_scene_flow/testing",
        transform=transform,
        height=config['data']['img_height'],
        width=config['data']['img_width']
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=4,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize visualizer
    visualizer = DepthVisualizer()
    
    # Test loop
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, (images, depths) in enumerate(tqdm(test_loader, desc="Testing")):
            images, depths = images.to(device), depths.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute metrics
            metrics = compute_depth_metrics(outputs, depths)
            all_metrics.append(metrics)
            
            # Save visualizations for first few batches
            if batch_idx < 5:
                for i in range(len(images)):
                    img_idx = batch_idx * test_loader.batch_size + i
                    visualizer.plot_depth_map(
                        images[i].cpu().permute(1,2,0).numpy(),
                        depths[i].cpu().squeeze().numpy(),
                        outputs[i].cpu().squeeze().numpy(),
                        save_path=os.path.join(args.output_dir, f'sample_{img_idx}.png')
                    )
    
    # Compute and print average metrics
    mean_metrics = {}
    for key in all_metrics[0].keys():
        mean_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    print("\nTest Results:")
    print("=" * 50)
    print("Error Metrics:")
    print(f"Abs Rel: {mean_metrics['abs_rel']:.4f}")
    print(f"Sq Rel: {mean_metrics['sq_rel']:.4f}")
    print(f"RMSE: {mean_metrics['rmse']:.4f}")
    print(f"RMSE log: {mean_metrics['rmse_log']:.4f}")
    print(f"SiLog: {mean_metrics['silog']:.4f}")
    print("\nAccuracy Metrics:")
    print(f"δ < 1.25: {mean_metrics['delta1']:.4f}")
    print(f"δ < 1.25²: {mean_metrics['delta2']:.4f}")
    print(f"δ < 1.25³: {mean_metrics['delta3']:.4f}")
    
    # Save metrics to file
    import json
    with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(mean_metrics, f, indent=4)

if __name__ == '__main__':
    main()
