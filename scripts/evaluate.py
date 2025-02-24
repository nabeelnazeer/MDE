import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.kitti_dataset import KITTIDataset
from models.resnet_depth import ResNetDepth
from utils.metrics import compute_depth_metrics
from visualization.visualizer import DepthVisualizer
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt

def evaluate(model, dataloader, device):
    model.eval()
    metrics_list = []
    
    with torch.no_grad():
        for images, depths in tqdm(dataloader):
            images, depths = images.to(device), depths.to(device)
            outputs = model(images)
            metrics = compute_depth_metrics(outputs, depths)
            metrics_list.append(metrics)
    
    # Average metrics
    mean_metrics = {}
    for key in metrics_list[0].keys():
        mean_metrics[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
    
    return mean_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup model and load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetDepth().to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((config['data']['img_height'], config['data']['img_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    test_dataset = KITTIDataset(
        root_dir="~/Documents/Project-s6/Datasets/data_scene_flow/testing",
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # Evaluate
    metrics = evaluate(model, test_loader, device)
    
    # Save results
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(os.path.join(args.output_dir, 'test_results.csv'))
    
    # Create visualization
    visualizer = DepthVisualizer()
    print("\nTest Set Metrics:")
    print(visualizer.create_comparison_table(metrics))
    
    # Plot sample predictions
    test_batch = next(iter(test_loader))
    images, depths = test_batch
    with torch.no_grad():
        pred_depths = model(images.to(device))
    
    for i in range(min(3, len(images))):
        visualizer.plot_depth_map(
            images[i].permute(1,2,0).cpu().numpy(),
            depths[i].squeeze().cpu().numpy(),
            pred_depths[i].squeeze().cpu().numpy(),
            save_path=os.path.join(args.output_dir, f'sample_prediction_{i}.png')
        )

if __name__ == '__main__':
    main()
