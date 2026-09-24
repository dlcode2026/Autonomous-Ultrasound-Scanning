import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.dataset import MultiUltrasoundDataset
from data.transforms import get_default_data_transform
from models.vaegan import ModifiedVAEGAN, train_vaegan

# Hyperparameters
batch_size = 8
img_size = 128
latent_dim = 100
label_dim = 12
num_epochs = 100
lr = 0.0001

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_transform = get_default_data_transform(img_size)
    
    dataset = MultiUltrasoundDataset(
        base_image_dir="images" ,
        base_label_dir="logs",
        transform=data_transform,
        normalization_method='domain_aware',
        calculate_params=True,
        exclude_folders=["1403","2802","1204"]
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,num_workers=8 )
    
    model = ModifiedVAEGAN(latent_dim=latent_dim, label_dim=label_dim, img_channels=1).to(device)
    
    model = train_vaegan(
        model, dataloader, num_epochs=num_epochs,
        lr=lr, device=device, save_dir='results'
    )

if __name__ == "__main__":
    main()
