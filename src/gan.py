
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random
from dataset import adultDataset, get_dataloader


# ---------------------------
# Generator & Discriminator
# ---------------------------
class Generator(nn.Module):
    """ Generator network that outputs autoencoder embeddings """
    def __init__(self, noise_dim, embedding_dim=32, hidden_dim=128):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(256, embedding_dim),
            nn.Tanh()  # Output scaled between [-1,1]
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    """ Discriminator network that classifies embeddings """
    def __init__(self, embedding_dim=32, hidden_dim=128):
        super(Discriminator, self).__init__()
        print(f"[Debug]: Discriminator expected input dim: {embedding_dim}")
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ---------------------------
# GAN Class
# ---------------------------
class GAN(nn.Module):
    """ GAN that generates autoencoder embeddings """
    def __init__(self, noise_dim, embedding_dim=32, hidden_dim=128, device="cpu", pretrained_path=None):
        super(GAN, self).__init__()
        print("[GAN]: Initializing model...")
        self.device = device
        self.generator = Generator(noise_dim=noise_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(self.device)
        self.discriminator = Discriminator(embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(self.device)
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim

        if pretrained_path and os.path.isfile(pretrained_path):
            self.load_weights(pretrained_path)

    def load_weights(self, pretrained_path):
        """ Load saved weights into the model """
        print(f"[GAN]: Loading model weights from {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint["generator_state_dict"])
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            print("[GAN]: Model loaded successfully!")
        except Exception as e:
            print(f"[GAN]: Failed to load model - {e}")

    def save_weights(self, epoch, gen_loss, disc_loss, save_path):
        """ Save model weights """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "gen_loss": gen_loss,
            "disc_loss": disc_loss
        }, save_path)
        print(f"[GAN]: Model saved at {save_path}")

    def train_model(self, train_loader, autoencoder, epochs, lr_gen, lr_disc, save_path=None):
        """ Train the GAN with autoencoder integration """
        writer = SummaryWriter(log_dir="../runs/GAN")
        criterion = nn.BCELoss()

        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))

        scheduler_gen = optim.lr_scheduler.OneCycleLR(
            optimizer_g,
            max_lr=lr_gen,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )

        scheduler_disc = optim.lr_scheduler.OneCycleLR(
            optimizer_d,
            max_lr=lr_disc / 2,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )

        for epoch in range(epochs):
            gen_loss_epoch, disc_loss_epoch = 0, 0
            for batch_idx, (real_data, _) in enumerate(train_loader):
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)

                # Get real embeddings from autoencoder
                with torch.no_grad():
                    real_embeddings = autoencoder.encoder(real_data)

                # Generate fake embeddings
                noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_embeddings = self.generator(noise)

                # Train Discriminator
                real_labels = torch.full((batch_size, 1), 0.9, device=self.device)  # Label smoothing
                fake_labels = torch.full((batch_size, 1), 0.1, device=self.device)

                real_output = self.discriminator(real_embeddings)
                fake_output = self.discriminator(fake_embeddings.detach())

                loss_real = criterion(real_output, real_labels)
                loss_fake = criterion(fake_output, fake_labels)
                d_loss = (loss_real + loss_fake) / 2

                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()
                scheduler_disc.step()

                disc_loss_epoch += d_loss.item()

                # Train Generator
                fake_output = self.discriminator(fake_embeddings)
                g_loss = criterion(fake_output, real_labels)

                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()
                scheduler_gen.step()

                gen_loss_epoch += g_loss.item()

            writer.add_scalar("Loss/Generator", gen_loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Loss/Discriminator", disc_loss_epoch / len(train_loader), epoch)

            print(f"[GAN]: Epoch {epoch+1} | G Loss: {gen_loss_epoch:.4f} | D Loss: {disc_loss_epoch:.4f}")

        writer.close()
        print("[GAN]: Training completed!")

        if save_path:
            self.save_weights(epoch, gen_loss_epoch, disc_loss_epoch, save_path)

    def generate(self, num_samples, autoencoder):
        """ Generate synthetic data samples using autoencoder """
        self.generator.eval()
        autoencoder.eval()
        
        noise = torch.randn(num_samples, self.noise_dim, device=self.device)
        
        with torch.no_grad():
            # Generate embeddings
            fake_embeddings = self.generator(noise)
            # Decode embeddings to full data
            synthetic_data = autoencoder.decoder(fake_embeddings)
        
        return synthetic_data.cpu()


# if __name__ == "__main__":
#     print("[Test]: Initializing dataset...")
#     dataset = adultDataset(data_path=DATA_PATH, target_column=TARGET_COLUMN)
#     dataset.load_data()
#     dataset.preprocess()

#     print("[Test]: Splitting dataset...")
#     train_set, _, _ = dataset.stratified_split(test_size=0.2, val_size=0.1, random_state=42)
#     train_loader = get_dataloader(train_set, batch_size=8, shuffle=True)  # Small batch for testing

#     print("[Test]: Training Autoencoder...")
#     autoencoder = train_autoencoder(train_loader=train_loader, input_dim=108, embedding_dim=32, num_epochs=2, device=DEVICE)

#     print("[Test]: Extracting embeddings...")
#     autoencoder.encoder.eval()
#     train_embeddings = []
#     train_labels = []

#     for real_data, labels in train_loader:
#         with torch.no_grad():
#             encoded_data = autoencoder.encoder(real_data.to(DEVICE)).cpu()
#         train_embeddings.append(encoded_data)
#         train_labels.append(labels.cpu())

#     train_embeddings = torch.cat(train_embeddings, dim=0)
#     train_labels = torch.cat(train_labels, dim=0)

#     # Create DataLoader for embeddings
#     train_dataset = torch.utils.data.TensorDataset(train_embeddings, train_labels)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

#     print("[Test]: Initializing GAN model...")
#     gan = GAN(noise_dim=NOISE_DIM, embedding_dim=32, device=DEVICE)

#     print("[Test]: Training GAN for 2 epochs (quick test)...")
#     gan.train_model(train_loader=train_loader, epochs=1, lr_gen=LEARNING_RATE,lr_disc=0.0002)

#     print("[Test]: Generating synthetic embeddings...")
#     synthetic_embeddings = gan.generate(num_samples=5)

#     print(f"Generated synthetic embeddings shape: {synthetic_embeddings.shape}")
#     assert synthetic_embeddings.shape == (5, 32), "Error: Output dimensions incorrect!"

#     print("[Test]: GAN test completed successfully!")


