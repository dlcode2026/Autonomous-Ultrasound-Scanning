import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm.auto import tqdm
from data.label_augment import augment_labels_torch


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, label_dim):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        
        self.gamma_embed = nn.Linear(label_dim, num_features)
        self.beta_embed = nn.Linear(label_dim, num_features)
        
    def forward(self, x, labels):
        out = self.bn(x)
        gamma = self.gamma_embed(labels).view(-1, self.num_features, 1, 1)
        beta = self.beta_embed(labels).view(-1, self.num_features, 1, 1)
        out = gamma * out + beta
        return out


class ConditionalResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, label_dim, stride=1, upsample=False):
        super(ConditionalResidualBlock, self).__init__()
        
        if upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1 if stride > 1 else 0)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            
            if in_channels != out_channels or stride != 1:
                self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, output_padding=1 if stride > 1 else 0)
            else:
                self.shortcut = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            
            if in_channels != out_channels or stride != 1:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            else:
                self.shortcut = nn.Identity()
        
        self.cbn1 = ConditionalBatchNorm2d(out_channels, label_dim)
        self.cbn2 = ConditionalBatchNorm2d(out_channels, label_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, labels):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.cbn1(out, labels)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.cbn2(out, labels)
        
        out += identity
        out = self.relu(out)
        
        return out


class DiscriminatorConditionalResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, label_dim, stride=1):
        super(DiscriminatorConditionalResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()
        
        self.label_proj1 = nn.Linear(label_dim, out_channels)
        self.label_proj2 = nn.Linear(label_dim, out_channels)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x, labels):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        
        label_proj1 = self.label_proj1(labels).view(-1, out.size(1), 1, 1)
        out = out + label_proj1  # Add label information to feature maps
        
        out = self.leaky_relu(out)
        
        out = self.conv2(out)
        
        label_proj2 = self.label_proj2(labels).view(-1, out.size(1), 1, 1)
        out = out + label_proj2
        
        out += identity
        out = self.leaky_relu(out)
        
        return out


class ConditionalGenerator(nn.Module):
    
    def __init__(self, z_dim=100, label_dim=12, img_channels=1):
        super(ConditionalGenerator, self).__init__()
        self.z_dim = z_dim
        self.label_dim = label_dim

        self.label_embed = nn.Sequential(
            nn.Linear(label_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(z_dim + 256, 512 * 2 * 2)

        self.initial_deconv = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)  # 4x4
        self.initial_cbn = ConditionalBatchNorm2d(512, label_dim)

        self.res_block1 = ConditionalResidualBlock(512, 256, label_dim, stride=2, upsample=True)  # 8x8
        self.res_block2 = ConditionalResidualBlock(256, 128, label_dim, stride=2, upsample=True)  # 16x16
        self.res_block3 = ConditionalResidualBlock(128, 64, label_dim, stride=2, upsample=True)   # 32x32
        self.res_block4 = ConditionalResidualBlock(64, 32, label_dim, stride=2, upsample=True)    # 64x64
        
        self.final_deconv = nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1)  # 128x128

    def forward(self, z, labels):

        label_embed = self.label_embed(labels)
        
        x = torch.cat([z, label_embed], dim=1)

        x = self.fc(x)
        x = x.view(-1, 512, 2, 2)

        x = self.initial_deconv(x)
        x = F.relu(self.initial_cbn(x, labels))
        
        x = self.res_block1(x, labels)
        x = self.res_block2(x, labels)
        x = self.res_block3(x, labels)
        x = self.res_block4(x, labels)
        
        x = torch.tanh(self.final_deconv(x))

        return x


class ConditionalDiscriminator(nn.Module):
    
    def __init__(self, img_channels=1, label_dim=12):
        super(ConditionalDiscriminator, self).__init__()
        self.label_dim = label_dim

        self.initial_conv = nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1)  # 64x64
        
        self.initial_label_proj = nn.Linear(label_dim, 32)

        self.res_block1 = DiscriminatorConditionalResidualBlock(32, 64, label_dim, stride=2)    # 32x32
        self.res_block2 = DiscriminatorConditionalResidualBlock(64, 128, label_dim, stride=2)   # 16x16
        self.res_block3 = DiscriminatorConditionalResidualBlock(128, 256, label_dim, stride=2)  # 8x8
        self.res_block4 = DiscriminatorConditionalResidualBlock(256, 512, label_dim, stride=2)  # 4x4
        
        self.final_conv = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)  # 2x2

        self.final_label_proj = nn.Linear(label_dim, 512 * 2 * 2)

        self.fc = nn.Linear(512 * 2 * 2, 1)

    def forward(self, x, labels):

        x = F.leaky_relu(self.initial_conv(x), 0.2)
        
        initial_label_proj = self.initial_label_proj(labels).view(-1, 32, 1, 1)
        x = x + initial_label_proj
        
        x = self.res_block1(x, labels)
        x = self.res_block2(x, labels)
        x = self.res_block3(x, labels)
        x = self.res_block4(x, labels)
        
        x = F.leaky_relu(self.final_conv(x), 0.2)
        
        x_flat = x.view(x.size(0), -1)  # Flatten

        final_label_proj = self.final_label_proj(labels)
        
        output = self.fc(x_flat)
        
        projection = torch.sum(x_flat * final_label_proj, dim=1, keepdim=True)
        
        return output + projection


class ConditionalGAN(nn.Module):
    
    def __init__(self, z_dim=100, label_dim=12, img_channels=1):
        super(ConditionalGAN, self).__init__()

        self.z_dim = z_dim
        self.label_dim = label_dim

        self.generator = ConditionalGenerator(z_dim, label_dim, img_channels)
        self.discriminator = ConditionalDiscriminator(img_channels, label_dim)

    def generate(self, z, labels):
        return self.generator(z, labels)

    def discriminate(self, x, labels):
        return self.discriminator(x, labels)

    def sample(self, n_samples, labels):
        z = torch.randn(n_samples, self.z_dim).to(labels.device)
        return self.generator(z, labels)

    def interpolate(self, label1, label2, steps=10):
        z = torch.randn(1, self.z_dim).to(label1.device)
        z = z.repeat(steps, 1)
        
        alphas = torch.linspace(0, 1, steps).to(label1.device)
        interpolated_labels = torch.zeros(steps, self.label_dim).to(label1.device)
        for i, alpha in enumerate(alphas):
            interpolated_labels[i] = label1 * (1 - alpha) + label2 * alpha
            
        return self.generator(z, interpolated_labels)


    
def train_conditional_gan(model, dataloader, num_epochs=100, lr=0.00005, beta1=0.5, beta2=0.999,
                          device='cuda', save_dir='residual_gan_results', save_interval=5,
                          start_epoch=0, resume_checkpoint=None):   

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)

    optimizer_G = optim.Adam(model.generator.parameters(),     lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        if 'optimizer_G_state_dict' in checkpoint:
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            print("  Restored optimizer states from checkpoint.")

    adversarial_loss = nn.BCEWithLogitsLoss()
    d_losses, g_losses = [], []

    fixed_real_imgs = None
    for real_imgs_batch, labels in dataloader:
        fixed_labels    = labels[:8].to(device)
        fixed_real_imgs = real_imgs_batch[:8]   # CPU
        break
    fixed_noise = torch.randn(8, model.z_dim).to(device)

    total_epochs = start_epoch + num_epochs
    for epoch in range(start_epoch, total_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, (real_imgs, labels) in progress_bar:
            batch_size_i = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels    = labels.to(device)

            valid = torch.ones(batch_size_i,  1).to(device)
            fake  = torch.zeros(batch_size_i, 1).to(device)

            optimizer_D.zero_grad()
            pred_real   = model.discriminate(real_imgs, labels)
            d_real_loss = adversarial_loss(pred_real, valid)
            z           = torch.randn(batch_size_i, model.z_dim).to(device)
            fake_imgs   = model.generate(z, labels)
            pred_fake   = model.discriminate(fake_imgs.detach(), labels)
            d_fake_loss = adversarial_loss(pred_fake, fake)
            d_loss      = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            z         = torch.randn(batch_size_i, model.z_dim).to(device)
            fake_imgs = model.generate(z, labels)
            pred_fake = model.discriminate(fake_imgs, labels)
            g_loss    = adversarial_loss(pred_fake, valid)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
            optimizer_G.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

            progress_bar.set_description(
                f"Epoch {epoch+1}/{total_epochs} | D: {d_loss.item():.4f} | G: {g_loss.item():.4f}"
            )

        epoch_d_loss /= len(dataloader)
        epoch_g_loss /= len(dataloader)
        d_losses.append(epoch_d_loss)
        g_losses.append(epoch_g_loss)

        if (epoch + 1) % save_interval == 0 or epoch == start_epoch:
            with torch.no_grad():
                fake_imgs = model.generate(fixed_noise, fixed_labels)

            n = fixed_real_imgs.size(0)
            fig, axes = plt.subplots(2, n, figsize=(n * 2, 5))
            fig.suptitle(f"Epoch {epoch+1}  |  Top: real    Bottom: generated", fontsize=10)
            for j in range(n):
                real_np = (fixed_real_imgs[j, 0].numpy() + 1) / 2.0
                gen_np  = (fake_imgs[j, 0].cpu().numpy() + 1) / 2.0
                axes[0, j].imshow(real_np, cmap="gray", vmin=0, vmax=1)
                axes[0, j].axis("off")
                axes[1, j].imshow(gen_np,  cmap="gray", vmin=0, vmax=1)
                axes[1, j].axis("off")
                if j == 0:
                    axes[0, j].set_title("Real",      fontsize=7)
                    axes[1, j].set_title("Generated", fontsize=7)
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, 'images', f'comparison_epoch_{epoch+1}.png'),
                dpi=150, bbox_inches="tight"
            )
            plt.close()

        # Save checkpoint — filename reflects the true epoch number
        if (epoch + 1) % save_interval == 0 or epoch + 1 == total_epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'd_loss': epoch_d_loss,
                'g_loss': epoch_g_loss,
            }, os.path.join(save_dir, 'models', f'conditional_gan_residual_epoch_{epoch+1}.pth'))

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.plot(d_losses); plt.title('Discriminator Loss'); plt.xlabel('Epoch (relative)')
    plt.subplot(1, 2, 2); plt.plot(g_losses); plt.title('Generator Loss');     plt.xlabel('Epoch (relative)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_losses_strong_conditioning.png'))

    return model


def load_conditional_gan_model(model_path, z_dim=100, label_dim=12, img_channels=1, device='cuda'):
    model = ConditionalGAN(z_dim, label_dim, img_channels).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def sample_conditional_gan_images(model, labels, n_samples=8, device='cuda'):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, model.z_dim).to(device)
        gen_imgs = model.generate(z, labels)
    return gen_imgs


def interpolate_conditional_gan_labels(model, label1, label2, steps=10, device='cuda'):
    model.eval()
    with torch.no_grad():
        return model.interpolate(label1, label2, steps)
