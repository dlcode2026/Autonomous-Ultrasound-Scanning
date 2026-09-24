import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm.auto import tqdm


class Encoder(nn.Module):
    
    def __init__(self, img_channels=1, latent_dim=100, label_dim=12):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) 
        self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1) 

        self.fc_mu     = nn.Linear(512 * 2 * 2 + label_dim, latent_dim)
        self.fc_logvar = nn.Linear(512 * 2 * 2 + label_dim, latent_dim)

        self.label_encoder = nn.Sequential(
            nn.Linear(label_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, labels):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = F.leaky_relu(self.conv6(x), 0.2)
        x = x.view(x.size(0), -1)

        label_features = self.label_encoder(labels)
        combined = torch.cat([x, labels], dim=1)

        mu     = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Generator(nn.Module):

    def __init__(self, latent_dim=100, label_dim=12, img_channels=1):
        super(Generator, self).__init__()

        self.fc = nn.Linear(latent_dim + label_dim, 512 * 2 * 2)

        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1) 
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) 
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) 
        self.deconv4 = nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1) 
        self.deconv5 = nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1) 
        self.deconv6 = nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        x = self.fc(x)
        x = x.view(-1, 512, 2, 2)

        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = F.relu(self.bn5(self.deconv5(x)))
        x = torch.tanh(self.deconv6(x))
        return x


class Discriminator(nn.Module):

    def __init__(self, img_channels=1, label_dim=12):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(img_channels, 32,  kernel_size=4, stride=2, padding=1) 
        self.conv2 = nn.Conv2d(32,  64,  kernel_size=4, stride=2, padding=1)           
        self.conv3 = nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1)  
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) 
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1) 
        self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1) 

        self.label_embedding = nn.Sequential(
            nn.Linear(label_dim, 512),
            nn.LeakyReLU(0.2)
        )

        self.fc = nn.Linear(512 * 2 * 2 + 512, 1)

    def forward(self, x, labels, return_features=False):
        x1 = F.leaky_relu(self.conv1(x),  0.2)
        x2 = F.leaky_relu(self.conv2(x1), 0.2)
        x3 = F.leaky_relu(self.conv3(x2), 0.2)
        x4 = F.leaky_relu(self.conv4(x3), 0.2)
        x5 = F.leaky_relu(self.conv5(x4), 0.2)
        x6 = F.leaky_relu(self.conv6(x5), 0.2)

        x_flat          = x6.view(x6.size(0), -1)
        label_embedding = self.label_embedding(labels)
        combined        = torch.cat([x_flat, label_embedding], dim=1)
        validity        = self.fc(combined)

        if return_features:
            return validity, [x1, x2, x3, x4, x5, x6]
        return validity


class ModifiedVAEGAN(nn.Module):
    def __init__(self, latent_dim=100, label_dim=12, img_channels=1):
        super(ModifiedVAEGAN, self).__init__()

        self.latent_dim = latent_dim
        self.label_dim  = label_dim

        self.encoder       = Encoder(img_channels, latent_dim, label_dim)
        self.generator     = Generator(latent_dim, label_dim, img_channels)
        self.discriminator = Discriminator(img_channels, label_dim)

    def encode(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = self.encoder.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z, labels):
        return self.generator(z, labels)

    def forward(self, x, labels):
        z, mu, logvar = self.encode(x, labels)
        x_recon = self.decode(z, labels)
        return x_recon, mu, logvar

    def sample(self, n_samples, labels):
        z = torch.randn(n_samples, self.latent_dim, device=labels.device)
        return self.generator(z, labels)

    def interpolate(self, label1, label2, steps=10):
        z = torch.randn(1, self.latent_dim, device=label1.device).repeat(steps, 1)
        alphas = torch.linspace(0, 1, steps, device=label1.device)
        interpolated_labels = torch.stack([
            label1 * (1 - a) + label2 * a for a in alphas
        ])
        return self.generator(z, interpolated_labels)


def train_vaegan(model, dataloader, num_epochs=200, lr=0.0001, beta1=0.5, beta2=0.999,
                 device='cuda', save_dir='results_128x128_fixed', save_interval=5, start_epoch=0):

    torch.backends.cudnn.benchmark = True
    model.encoder       = torch.compile(model.encoder)
    model.generator     = torch.compile(model.generator)
    model.discriminator = torch.compile(model.discriminator)
                   
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)

    optimizer_G = optim.Adam(
        list(model.encoder.parameters()) + list(model.generator.parameters()),
        lr=lr, betas=(beta1, beta2)
    )
    optimizer_D = optim.Adam(
        model.discriminator.parameters(),
        lr=lr * 0.5, betas=(beta1, beta2)
    )

    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.8)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.8)

    scaler_G = torch.cuda.amp.GradScaler()
    scaler_D = torch.cuda.amp.GradScaler()

    adversarial_loss      = nn.BCEWithLogitsLoss()
    reconstruction_loss   = nn.MSELoss()
    feature_matching_loss = nn.L1Loss()

    recon_weight = 5.0 
    fm_weight    = 2.0 
    adv_weight   = 0.1 

    d_losses     = []
    g_losses     = []
    recon_losses = []
    kl_losses    = []
    fm_losses    = []

    fixed_labels = next(iter(dataloader))[1][:8].to(device, non_blocking=True)
    fixed_noise  = torch.randn(8, model.latent_dim, device=device)

    for epoch in range(start_epoch, num_epochs):
        epoch_d_loss     = 0.0
        epoch_g_loss     = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss    = 0.0
        epoch_fm_loss    = 0.0

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, (real_imgs, labels) in progress_bar:
            batch_size = real_imgs.size(0)

            real_imgs = real_imgs.to(device, non_blocking=True)
            labels    = labels.to(device, non_blocking=True)

            valid = torch.full((batch_size, 1), 0.9, device=device)
            fake  = torch.full((batch_size, 1), 0.1, device=device)

            if i % 2 == 0:
                optimizer_D.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast():
                    pred_real   = model.discriminator(real_imgs, labels)
                    d_real_loss = adversarial_loss(pred_real, valid)

                    with torch.no_grad():
                        z, _, _      = model.encode(real_imgs, labels)
                        decoded_imgs = model.decode(z, labels)
                        z_random     = torch.randn(batch_size, model.latent_dim, device=device)
                        gen_imgs     = model.generator(z_random, labels)

                    pred_fake_decoded = model.discriminator(decoded_imgs.detach(), labels)
                    pred_fake_gen     = model.discriminator(gen_imgs.detach(), labels)
                    d_fake_loss = (
                        adversarial_loss(pred_fake_decoded, fake) +
                        adversarial_loss(pred_fake_gen, fake)
                    ) / 2

                    noise_level = max(0.0, 0.1 * (1.0 - epoch / 100.0))
                    if noise_level > 0:
                        instance_noise  = torch.randn_like(real_imgs) * noise_level
                        pred_real_noisy = model.discriminator(real_imgs + instance_noise, labels)
                        pred_fake_noisy = model.discriminator(gen_imgs  + instance_noise, labels)
                        d_noise_loss = (
                            adversarial_loss(pred_real_noisy, valid) +
                            adversarial_loss(pred_fake_noisy, fake)
                        ) / 2
                        d_loss = (d_real_loss + d_fake_loss + d_noise_loss) / 3
                    else:
                        d_loss = (d_real_loss + d_fake_loss) / 2

                scaler_D.scale(d_loss).backward()
                scaler_D.unscale_(optimizer_D)
                torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
                scaler_D.step(optimizer_D)
                scaler_D.update()
            else:
                d_loss = torch.tensor(0.0, device=device)


            optimizer_G.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                z, mu, logvar = model.encode(real_imgs, labels)
                decoded_imgs  = model.decode(z, labels)

                pixel_recon_loss = reconstruction_loss(decoded_imgs, real_imgs)

                # KL divergence
                kl_loss   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
                kl_weight = min(0.1, 0.001 * (epoch + 1))

                # Feature matching loss
                _, real_features                  = model.discriminator(real_imgs,    labels, return_features=True)
                validity_decoded, fake_features   = model.discriminator(decoded_imgs, labels, return_features=True)

                fm_loss = sum(
                    feature_matching_loss(f_fake, f_real.detach())
                    for f_fake, f_real in zip(fake_features, real_features)
                ) / len(real_features)

                # Adversarial loss
                z_random     = torch.randn(batch_size, model.latent_dim, device=device)
                gen_imgs     = model.generator(z_random, labels)
                validity_gen = model.discriminator(gen_imgs, labels)

                g_loss = (
                    adversarial_loss(validity_decoded, valid) +
                    adversarial_loss(validity_gen, valid)
                ) / 2

                # Diversity loss
                if epoch > 20:
                    z1 = torch.randn(batch_size, model.latent_dim, device=device)
                    z2 = torch.randn(batch_size, model.latent_dim, device=device)
                    gen_imgs1      = model.generator(z1, labels)
                    gen_imgs2      = model.generator(z2, labels)
                    img_diff       = torch.mean(torch.abs(gen_imgs1 - gen_imgs2))
                    z_diff         = torch.mean(torch.norm(z1 - z2, dim=1))
                    diversity_loss = -0.1 * (img_diff / z_diff)
                else:
                    diversity_loss = torch.tensor(0.0, device=device)

                # Total generator + encoder loss
                g_total = (
                    recon_weight * pixel_recon_loss +
                    fm_weight    * fm_loss +
                    kl_weight    * kl_loss +
                    adv_weight   * g_loss +
                    diversity_loss
                )

            scaler_G.scale(g_total).backward()
            scaler_G.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),   max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
            scaler_G.step(optimizer_G)
            scaler_G.update()

            epoch_d_loss     += d_loss.item()
            epoch_g_loss     += g_loss.item()
            epoch_recon_loss += pixel_recon_loss.item()
            epoch_kl_loss    += kl_loss.item()
            epoch_fm_loss    += fm_loss.item()

            progress_bar.set_description(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f} | "
                f"R: {pixel_recon_loss.item():.4f} | FM: {fm_loss.item():.4f} | "
                f"KL: {kl_loss.item():.4f}"
            )

        scheduler_G.step()
        scheduler_D.step()

        n_batches         = len(dataloader)
        epoch_d_loss     /= n_batches
        epoch_g_loss     /= n_batches
        epoch_recon_loss /= n_batches
        epoch_kl_loss    /= n_batches
        epoch_fm_loss    /= n_batches

        d_losses.append(epoch_d_loss)
        g_losses.append(epoch_g_loss)
        recon_losses.append(epoch_recon_loss)
        kl_losses.append(epoch_kl_loss)
        fm_losses.append(epoch_fm_loss)

        if (epoch + 1) % save_interval == 0 or epoch == 0:
            with torch.no_grad():
                fake_imgs = model.generator(fixed_noise, fixed_labels)

                for j, (real_batch, labels_batch) in enumerate(dataloader):
                    if j == 0:
                        real_batch   = real_batch[:8].to(device, non_blocking=True)
                        labels_batch = labels_batch[:8].to(device, non_blocking=True)
                        z, _, _      = model.encode(real_batch, labels_batch)
                        recon_batch  = model.decode(z, labels_batch)

                        save_image(real_batch,
                                   os.path.join(save_dir, 'images', f'real_epoch_{epoch+1}.png'),
                                   normalize=True)
                        save_image(recon_batch,
                                   os.path.join(save_dir, 'images', f'recon_epoch_{epoch+1}.png'),
                                   normalize=True)
                        save_image(fake_imgs,
                                   os.path.join(save_dir, 'images', f'generated_epoch_{epoch+1}.png'),
                                   normalize=True)
                    break

        if (epoch + 1) % save_interval == 0 or epoch + 1 == num_epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scaler_G_state_dict': scaler_G.state_dict(),
                'scaler_D_state_dict': scaler_D.state_dict(),
                'd_loss':    epoch_d_loss,
                'g_loss':    epoch_g_loss,
                'recon_loss': epoch_recon_loss,
                'kl_loss':   epoch_kl_loss,
                'fm_loss':   epoch_fm_loss,
            }, os.path.join(save_dir, 'models', f'vaegan_epoch_{epoch+1}.pth'))

    # -----------------------------------------------------------------
    # Plot losses
    # -----------------------------------------------------------------
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.plot(d_losses)
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')

    plt.subplot(2, 3, 2)
    plt.plot(g_losses)
    plt.title('Generator Loss (Adversarial)')
    plt.xlabel('Epoch')

    plt.subplot(2, 3, 3)
    plt.plot(recon_losses)
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')

    plt.subplot(2, 3, 4)
    plt.plot(kl_losses)
    plt.title('KL Divergence Loss')
    plt.xlabel('Epoch')

    plt.subplot(2, 3, 5)
    plt.plot(fm_losses)
    plt.title('Feature Matching Loss')
    plt.xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_losses.png'))

    return model


def load_model(model_path, latent_dim=100, label_dim=12, img_channels=1, device='cuda'):
    model = ModifiedVAEGAN(latent_dim, label_dim, img_channels).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def sample_images(model, labels, n_samples=8, device='cuda'):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim, device=device)
        gen_imgs = model.generator(z, labels)
    return gen_imgs


def interpolate_labels(model, label1, label2, steps=10, device='cuda'):
    model.eval()
    with torch.no_grad():
        return model.interpolate(label1, label2, steps)
