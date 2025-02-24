import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms.functional as TF
import random

class KITTIDataset(Dataset):
    def __init__(self, root_dir, transform=None, height=256, width=832, max_samples=None):
        self.root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self.height = height
        self.width = width
         
        # Verify path exists
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset path not found: {self.root_dir}")
            
        image_dir = os.path.join(self.root_dir, 'image_2')
        disp_noc_0_dir = os.path.join(self.root_dir, 'disp_noc_0')
        disp_noc_1_dir = os.path.join(self.root_dir, 'disp_noc_1')
        
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(disp_noc_0_dir) or not os.path.exists(disp_noc_1_dir):
            raise FileNotFoundError("Disparity directories not found")
        
        # Get image and corresponding disparity files
        self.samples = []
        
        # Improve pair matching
        base_names = []
        for img_file in sorted(os.listdir(image_dir)):
            if not img_file.endswith('_10.png'):  # Only look at first image of each pair
                continue
            base_name = img_file[:-7]  # Remove _10.png suffix
            
            # Verify all required files exist
            files_to_check = {
                'img_10': os.path.join(image_dir, f"{base_name}_10.png"),
                'img_11': os.path.join(image_dir, f"{base_name}_11.png"),
                'disp_0': os.path.join(disp_noc_0_dir, f"{base_name}_10.png"),
                'disp_1': os.path.join(disp_noc_1_dir, f"{base_name}_10.png")
            }
            
            if all(os.path.exists(f) for f in files_to_check.values()):
                # Verify file integrity
                try:
                    # Quick check that files are readable
                    Image.open(files_to_check['img_10']).verify()
                    Image.open(files_to_check['img_11']).verify()
                    disp0 = cv2.imread(files_to_check['disp_0'], cv2.IMREAD_UNCHANGED)
                    disp1 = cv2.imread(files_to_check['disp_1'], cv2.IMREAD_UNCHANGED)
                    
                    if disp0 is not None and disp1 is not None:
                        self.samples.append({
                            'base_name': base_name,
                            **files_to_check
                        })
                except Exception as e:
                    print(f"Skipping corrupted files for {base_name}: {str(e)}")
        
        print(f"Found {len(self.samples)} valid stereo pairs")
        print(f"Total training samples: {len(self.samples) * 2}")
        
        # Print first few pairs for verification
        if len(self.samples) > 0:
            print("\nFirst few image-disparity pairs:")
            for sample in self.samples[:3]:
                print(f"\nBase name: {sample['base_name']}")
                print(f"Image 1: {os.path.basename(sample['img_10'])}")
                print(f"Image 2: {os.path.basename(sample['img_11'])}")
                print(f"Disparity 1: {os.path.basename(sample['disp_0'])}")
                print(f"Disparity 2: {os.path.basename(sample['disp_1'])}")
        
        # Randomly sample pairs if max_samples is specified
        if max_samples and max_samples < len(self.samples):
            random.seed(42)  # For reproducibility
            self.samples = random.sample(self.samples, max_samples)
            print(f"Randomly sampled {max_samples} pairs from {len(base_names)} total pairs")
        
        total_images = len(self.samples) * 2
        print(f"Dataset contains {total_images} images ({len(self.samples)} stereo pairs)")
        
    def __len__(self):
        return len(self.samples) * 2  # Each sample provides 2 training examples
    
    def __getitem__(self, idx):
        sample_idx = idx // 2  # Which stereo pair
        is_second = idx % 2    # Whether to use second image of the pair
        
        sample = self.samples[sample_idx]
        
        # Select appropriate image and disparity based on index
        if is_second:
            img_path = sample['img_11']
            disp_path = sample['disp_1']
        else:
            img_path = sample['img_10']
            disp_path = sample['disp_0']
        
        # Load and process image
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.width, self.height), Image.BILINEAR)
        
        # Load and process disparity
        disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        if disp is None:
            raise ValueError(f"Failed to load disparity file: {disp_path}")
        
        # Convert disparity to float32 and normalize
        disp = disp.astype(np.float32) / 256.0
        disp = cv2.resize(disp, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensors
        if self.transform:
            image = self.transform(image)
        else:
            image = TF.to_tensor(image)
        
        disp = torch.from_numpy(disp).unsqueeze(0)
        
        return image, disp
