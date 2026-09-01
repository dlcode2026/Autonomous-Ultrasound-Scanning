
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
from data.dataset_quat import FlatUltrasoundDataset
from data.transforms import get_default_data_transform
from models.residual_gan import ConditionalGAN, train_conditional_gan

BATCH_SIZE  = 32
IMG_SIZE    = 128
LATENT_DIM  = 100
LABEL_DIM   = 7        # [x, y, z, qx, qy, qz, qw]
NUM_EPOCHS  = 100
LR          = 0.00005  


IMAGE_DIR   = "images_path"
CSV_PATH    = "conditions_csv_path"
SAVE_DIR    = "results"

RESUME_PATH = None

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_transform = get_default_data_transform(IMG_SIZE)


    dataset1 = FlatUltrasoundDataset(
    image_dir=IMAGE_DIR,
    csv_path=CSV_PATH,
    transform=data_transform,
    augment=True
    )

    dataset2 = FlatUltrasoundDataset(
        image_dir="images_path2",
        csv_path="conditions_csv_path2",
        transform=data_transform,
        augment=True
    )

    dataset = ConcatDataset([dataset1, dataset2])

    print(f"Dataset ready: {len(dataset)} samples")
    print(f"   NORM_MIN (pos): {dataset.NORM_MIN[:3]}")
    print(f"   NORM_MAX (pos): {dataset.NORM_MAX[:3]}\n")

    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        drop_last   = True,
        pin_memory  = True,
        num_workers = 8,
    )

    model = ConditionalGAN(
        z_dim       = LATENT_DIM,
        label_dim   = LABEL_DIM,
        img_channels= 1,
    ).to(device)

    start_epoch = 0
    if RESUME_PATH is not None and os.path.exists(RESUME_PATH):
        print(f"Resuming from: {RESUME_PATH}")
        ckpt = torch.load(RESUME_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Starting from epoch {start_epoch}")
    else:
        print("Starting fresh training")

    model = train_conditional_gan(
        model            = model,
        dataloader       = dataloader,
        num_epochs       = NUM_EPOCHS,
        lr               = LR,
        device           = device,
        save_dir         = SAVE_DIR,
        save_interval    = 5,
        start_epoch      = start_epoch,
        resume_checkpoint= RESUME_PATH,
    )

    print("Training complete")

if __name__ == "__main__":
    main()
