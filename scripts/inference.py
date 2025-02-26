import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models.resnet_depth import ResNetDepth
import numpy as np
from tqdm import tqdm

def load_model(checkpoint_path, device):
    """Load the trained model"""
    model = ResNetDepth().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from checkpoint with validation loss: {checkpoint['val_loss']:.4f}")
    return model

def preprocess_image(image_path, height=256, width=832):
    """Preprocess a single image"""
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

def save_depth_map(depth, original_image, save_path):
    """Save depth map visualization"""
    plt.figure(figsize=(10, 4))
    
    # Original image
    plt.subplot(121)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Depth map
    plt.subplot(122)
    plt.imshow(depth.squeeze(), cmap='plasma')
    plt.title('Depth Map')
    plt.colorbar(label='Depth')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_test_images(test_dir):
    """Get all test images from both left and right cameras"""
    left_dir = os.path.join(test_dir, 'image_2')
    right_dir = os.path.join(test_dir, 'image_3')
    
    left_images = sorted([os.path.join('image_2', f) 
                         for f in os.listdir(left_dir) if f.endswith('.png')])
    right_images = sorted([os.path.join('image_3', f) 
                          for f in os.listdir(right_dir) if f.endswith('.png')])
    
    print(f"Found {len(left_images)} left camera images")
    print(f"Found {len(right_images)} right camera images")
    
    return left_images + right_images

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    with open('config/kitti_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Load the trained model
    model = load_model('checkpoints/best_model.pth', device)
    
    # Create output directory
    output_dir = 'inference_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test images from both cameras
    test_dir = "/Users/nabeelnazeer/Documents/Project-s6/Datasets/data_scene_flow/testing"
    test_images = get_test_images(test_dir)
    
    print(f"\nProcessing {len(test_images)} total test images...")
    
    with torch.no_grad():
        for img_rel_path in tqdm(test_images, desc="Processing images"):
            img_path = os.path.join(test_dir, img_rel_path)
            
            # Create subdirectory structure matching input
            camera_dir = os.path.dirname(img_rel_path)  # image_2 or image_3
            output_subdir = os.path.join(output_dir, camera_dir)
            os.makedirs(output_subdir, exist_ok=True)
            
            # 2. Preprocess image
            img_tensor, original_img = preprocess_image(
                img_path, 
                height=config['data']['img_height'],
                width=config['data']['img_width']
            )
            img_tensor = img_tensor.to(device)
            
            # 3. Run inference
            depth_pred = model(img_tensor)
            
            # 4. Convert to depth map
            depth_map = depth_pred.cpu().numpy()[0, 0]
            
            # 5. Save visualization
            base_name = os.path.basename(img_path)
            save_path = os.path.join(output_subdir, f'depth_{base_name}')
            save_depth_map(depth_map, original_img, save_path)
    
    print(f"\nProcessing complete! Results saved in: {output_dir}")
    print("Summary:")
    print(f"- Processed {len(test_images)} images")
    print(f"  └─ Left camera: {len([f for f in test_images if 'image_2' in f])} images")
    print(f"  └─ Right camera: {len([f for f in test_images if 'image_3' in f])} images")
    print(f"- Input size: {config['data']['img_height']}x{config['data']['img_width']}")
    print(f"- Output directory structure:")
    print(f"  {output_dir}/")
    print(f"  ├─ image_2/")
    print(f"  └─ image_3/")

if __name__ == '__main__':
    main()
