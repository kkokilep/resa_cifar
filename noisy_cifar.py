import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader

class NoisyCIFAR100(Dataset):
    def __init__(self, cifar100_dataset, noise_percentage_of_dataset, noise_std):
        self.dataset = cifar100_dataset
        self.noise_std = noise_std
        num_noisy = int(len(cifar100_dataset) * noise_percentage_of_dataset)
        # Randomly select indices to apply noise to
        self.noisy_indices = set(random.sample(range(len(cifar100_dataset)), num_noisy))
        
        # Define a transform that adds noise
        self.add_noise_transform = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(std=noise_std), # Add noise based on desired std
            transforms.ToPILImage()
        ])
        self.to_tensor = transforms.ToTensor() # For clean images
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx] # This already has the base transforms applied

        if idx in self.noisy_indices:
            # Apply additional noise transform if it's a noisy image
            img = self.add_noise_transform(img)
            img = self.to_tensor(img) # Convert to tensor after noise
        else:
            img = self.to_tensor(img)

        # Ensure values are in the correct range [0, 1] before potential normalization
        img = torch.clamp(img, 0., 1.) 
        
        return img, label