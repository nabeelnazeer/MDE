import os
import sys
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from models.resnet_depth import ResNetDepth
from tqdm import tqdm

def load_model(checkpoint_path, device):
    model = ResNetDepth().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def save_depth_map(depth, save_path):
    # Convert to numpy and normalize for visualization
    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    depth = depth.astype(np.uint8)
    
    # Create color map
    plt.imsave(save_path, depth, cmap='magma')

def main():
    # Configuration
    config_path = 'config/kitti_config.yaml'
    checkpoint_path = 'checkpoints/best_model.pth'
    test_dir = r"C:\Users\jesli\Downloads\data_scene_flow\testing\image_2"
    output_dir = 'outputs/test_predictions'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up transform
    transform = transforms.Compose([
        transforms.Resize((config['data']['img_height'], config['data']['img_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path, device)
    
    # Process test images
    print("Processing test images...")
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    
    with torch.no_grad():
        for img_file in tqdm(test_files):
            # Load and preprocess image
            img_path = os.path.join(test_dir, img_file)
            image = preprocess_image(img_path, transform).to(device)
            
            # Predict depth
            depth = model(image)
            
            # Save result
            save_path = os.path.join(output_dir, f'depth_{img_file}')
            save_depth_map(depth, save_path)
            
            # Save side-by-side comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Original image
            img = Image.open(img_path)
            ax1.imshow(img)
            ax1.set_title('Input Image')
            ax1.axis('off')
            
            # Depth map
            depth_vis = depth.squeeze().cpu().numpy()
            ax2.imshow(depth_vis, cmap='magma')
            ax2.set_title('Predicted Depth')
            ax2.axis('off')
            
            plt.savefig(os.path.join(output_dir, f'comparison_{img_file}'))
            plt.close()
    
    print(f"Results saved to {output_dir}")
    
    # Display a few random examples
    n_examples = 3
    random_samples = np.random.choice(test_files, n_examples, replace=False)
    
    plt.figure(figsize=(15, 5*n_examples))
    for i, sample in enumerate(random_samples):
        # Original image
        plt.subplot(n_examples, 2, i*2 + 1)
        img = Image.open(os.path.join(test_dir, sample))
        plt.imshow(img)
        plt.title(f'Input Image {i+1}')
        plt.axis('off')
        
        # Depth map
        plt.subplot(n_examples, 2, i*2 + 2)
        depth_img = plt.imread(os.path.join(output_dir, f'depth_{sample}'))
        plt.imshow(depth_img)
        plt.title(f'Predicted Depth {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'random_examples.png'))
    plt.show()

if __name__ == '__main__':
    main()
