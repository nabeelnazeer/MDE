import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms.functional as TF
import random

class KITTIDataset(Dataset):
    def __init__(self, root_dir, transform=None, height=256, width=832, max_samples=None, split='train'):
        self.root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self.height = height
        self.width = width
        self.split = split
        
        # Verify path exists
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset path not found: {self.root_dir}")
        
        # Updated paths for KITTI structure
        image_dir = os.path.join(self.root_dir, 'image_2')
        if split == 'train':
            disp_dir = os.path.join(self.root_dir, 'disp_occ_0')
        else:
            disp_dir = os.path.join(self.root_dir, 'disp_occ_0')  # Use same for validation
        
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}\nPlease ensure you're pointing to the 'training' or 'testing' subdirectory.")
        
        # Get image and corresponding disparity files
        self.samples = []
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        for img_file in image_files:
            image_path = os.path.join(image_dir, img_file)
            
            # For training split, we need disparity files
            if self.split == 'train':
                disp_file = img_file  # Same filename for disparity
                disp_path = os.path.join(disp_dir, disp_file)
                
                if not os.path.exists(disp_path):
                    continue
                
                try:
                    # Verify files are readable
                    Image.open(image_path).verify()
                    disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
                    
                    if disp is not None:
                        self.samples.append({
                            'image': image_path,
                            'disp': disp_path
                        })
                except Exception as e:
                    print(f"Skipping corrupted files: {img_file} - {str(e)}")
            else:
                # For testing split, we only need images
                self.samples.append({
                    'image': image_path,
                    'disp': None
                })
        
        print(f"Found {len(self.samples)} valid samples for {split} split")
        
        # Randomly sample if max_samples is specified
        if max_samples and max_samples < len(self.samples):
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)
            print(f"Randomly sampled {max_samples} samples")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load and process image
            image = Image.open(sample['image']).convert('RGB')
            image = image.resize((self.width, self.height), Image.BILINEAR)
            
            if self.transform:
                image = self.transform(image)
            else:
                image = TF.to_tensor(image)
            
            # For training split, load disparity
            if self.split == 'train' and sample['disp'] is not None:
                disp = cv2.imread(sample['disp'], cv2.IMREAD_UNCHANGED)
                if disp is None:
                    raise ValueError(f"Failed to load disparity: {sample['disp']}")
                disp = disp.astype(np.float32) / 256.0
                disp = cv2.resize(disp, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                disp = torch.from_numpy(disp).unsqueeze(0)
            else:
                disp = torch.zeros((1, self.height, self.width))
            
            # Ensure tensors are float32
            image = image.float()
            disp = disp.float()
            
            return image.contiguous(), disp.contiguous()
            
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            return torch.zeros((3, self.height, self.width)), torch.zeros((1, self.height, self.width))
