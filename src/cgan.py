
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset import adultDataset, get_dataloader
from autoencoder import train_autoencoder

# ---------------------------
# Conditional Generator & Discriminator
# ---------------------------
class ConditionalGenerator(nn.Module):
    """ Conditional Generator network that outputs autoencoder embeddings """
    def __init__(self, noise_dim, label_dim, embedding_dim=32, hidden_dim=128):
        super(ConditionalGenerator, self).__init__()
        self.input_dim = noise_dim + label_dim  # Concatenated Input
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
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

    def forward(self, z, labels):
        """ Concatenate noise and labels before passing to network """
        combined_input = torch.cat([z, labels], dim=1)
        return self.model(combined_input)

class ConditionalDiscriminator(nn.Module):
    """ Conditional Discriminator network that classifies embeddings """
    def __init__(self, embedding_dim=32, label_dim=2, hidden_dim=128):
        super(ConditionalDiscriminator, self).__init__()
        self.input_dim = embedding_dim + label_dim
        print(f"[Debug]: Discriminator expected input dim: {self.input_dim}")
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
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

    def forward(self, x, labels):
        """ Concatenate embeddings and labels before passing to network """
        combined_input = torch.cat([x, labels], dim=1)
        return self.model(combined_input)


# ---------------------------
# Conditional GAN Class
# ---------------------------
class ConditionalGAN(nn.Module):
    """ Conditional GAN that generates autoencoder embeddings """
    def __init__(self, noise_dim, embedding_dim=32, label_dim=2, hidden_dim=128, device="cpu", pretrained_path=None):
        super(ConditionalGAN, self).__init__()
        print("[cGAN]: Initializing model...")
        self.device = device
        self.generator = ConditionalGenerator(
            noise_dim=noise_dim,
            label_dim=label_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        self.discriminator = ConditionalDiscriminator(
            embedding_dim=embedding_dim,
            label_dim=label_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        self.noise_dim = noise_dim
        self.label_dim = label_dim
        self.embedding_dim = embedding_dim

        if pretrained_path and os.path.isfile(pretrained_path):
            self.load_weights(pretrained_path)

    def load_weights(self, pretrained_path):
        """ Load saved weights into the model """
        print(f"[cGAN]: Loading model weights from {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint["generator_state_dict"])
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            print("[cGAN]: Model loaded successfully!")
        except Exception as e:
            print(f"[cGAN]: Failed to load model - {e}")

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
        print(f"[cGAN]: Model saved at {save_path}")

    def train_model(self, train_loader, autoencoder, epochs, lr_gen, lr_disc, save_path=None):
        """ Train the Conditional GAN with autoencoder embeddings """
        writer = SummaryWriter(log_dir="../runs/cGAN")
        criterion = nn.BCELoss()

        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))

        scheduler_gen = optim.lr_scheduler.OneCycleLR(
            optimizer_g,
            max_lr=lr_gen,
            epochs=epochs,
            steps_per_epoch=len(train_loader) * 2,
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
            for batch_idx, (real_data, labels) in enumerate(train_loader):
                real_data = real_data.to(self.device)
                # Get embeddings from autoencoder
                with torch.no_grad():
                    real_embeddings = autoencoder.encoder(real_data)
                
                # Convert labels to one-hot
                labels = labels.long()
                labels = torch.nn.functional.one_hot(labels, num_classes=self.label_dim).float().to(self.device)
                batch_size = real_data.size(0)

                # Train Discriminator
                noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_embeddings = self.generator(noise, labels)

                real_labels = torch.full((batch_size, 1), 0.9, device=self.device)
                fake_labels = torch.full((batch_size, 1), 0.1, device=self.device)

                real_output = self.discriminator(real_embeddings, labels)
                fake_output = self.discriminator(fake_embeddings.detach(), labels)

                loss_real = criterion(real_output, real_labels)
                loss_fake = criterion(fake_output, fake_labels)
                d_loss = (loss_real + loss_fake) / 2

                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()
                scheduler_disc.step()

                disc_loss_epoch += d_loss.item()

                # Train Generator
                fake_output = self.discriminator(fake_embeddings, labels)
                g_loss = criterion(fake_output, real_labels)

                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()
                scheduler_gen.step()

                gen_loss_epoch += g_loss.item()

            writer.add_scalar("Loss/Generator", gen_loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Loss/Discriminator", disc_loss_epoch / len(train_loader), epoch)

            print(f"[cGAN]: Epoch {epoch+1} | G Loss: {gen_loss_epoch:.4f} | D Loss: {disc_loss_epoch:.4f}")

        writer.close()
        print("[cGAN]: Training completed!")

        if save_path:
            self.save_weights(epoch, gen_loss_epoch, disc_loss_epoch, save_path)

    def generate(self, num_samples, labels, autoencoder):
        """ Generate synthetic data samples using autoencoder """
        self.generator.eval()
        autoencoder.eval()
        
        noise = torch.randn(num_samples, self.noise_dim, device=self.device)
        labels = labels.long()
        labels = torch.nn.functional.one_hot(labels, num_classes=self.label_dim).float().to(self.device)
        
        with torch.no_grad():
            # Generate embeddings
            fake_embeddings = self.generator(noise, labels)
            # Decode embeddings to full data
            synthetic_data = autoencoder.decoder(fake_embeddings)
        
        return synthetic_data.cpu()



# if __name__ == "__main__":
#   # ---------------------------
#   # Configuration
#   # ---------------------------
#   DATA_PATH = "/content/adult.arff"
#   # TARGET_COLUMN = "income"  # The column used as labels for conditioning
#   # NUM_CLASSES = 2  # Binary classification (income >50K, <=50K)
#   # NOISE_DIM = 32  # Noise vector size
#   # EMBEDDING_DIM = 64  # Autoencoder embedding size
#   # LEARNING_RATE = 0.0005
#   # BATCH_SIZE = 8  # Small batch for quick testing
#   # EPOCHS = 2  # Quick test
#   # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#   # ---------------------------
#   # Step 1: Load and Preprocess Dataset
#   # ---------------------------
#   print("[Test]: Initializing dataset...")
#   dataset = adultDataset(data_path=DATA_PATH, target_column=TARGET_COLUMN)
#   dataset.load_data()
#   dataset.preprocess()

#   # ---------------------------
#   # Step 2: Split and Prepare DataLoader
#   # ---------------------------
#   print("[Test]: Splitting dataset...")
#   train_set, _, _ = dataset.stratified_split(test_size=0.2, val_size=0.1, random_state=42)
#   train_loader = get_dataloader(train_set, batch_size=BATCH_SIZE, shuffle=True)

#   # ---------------------------
#   # Step 3: Train Autoencoder
#   # ---------------------------
#   # print("[Test]: Training Autoencoder...")
#   # autoencoder = train_autoencoder(
#   #     train_loader=train_loader,
#   #     input_dim=dataset.df.shape[1] - 1,  # Exclude target column
#   #     embedding_dim=EMBEDDING_DIM,
#   #     num_epochs=5,  # Short training
#   #     device=DEVICE
#   # )

#   # ---------------------------
#   # Step 4: Extract Embeddings
#   # ---------------------------
#   print("[Test]: Extracting embeddings...")
#   autoencoder.encoder.eval()
#   train_embeddings, train_labels = [], []

#   for real_data, labels in train_loader:
#       with torch.no_grad():
#           encoded_data = autoencoder.encoder(real_data.to(DEVICE)).cpu()
#       train_embeddings.append(encoded_data)
#       train_labels.append(labels.cpu())

#   train_embeddings = torch.cat(train_embeddings, dim=0)
#   train_labels = torch.cat(train_labels, dim=0)

#   # Wrap embeddings in DataLoader
#   train_dataset = torch.utils.data.TensorDataset(train_embeddings, train_labels)
#   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

#   # ---------------------------
#   # Step 5: Train cGAN
#   # ---------------------------
#   print("[Test]: Initializing cGAN model...")
#   cgan = ConditionalGAN(
#       noise_dim=NOISE_DIM,
#       embedding_dim=32,
#       label_dim=NUM_CLASSES,
#       device=DEVICE
#   )

#   print("[Test]: Training cGAN for 2 epochs (quick test)...")
#   cgan.train_model(train_loader=train_loader, epochs=2, lr=LEARNING_RATE)

#   # ---------------------------
#   # Step 6: Generate Synthetic Data
#   # ---------------------------
#   print("[Test]: Generating synthetic embeddings...")
#   sample_labels = torch.tensor([0, 1, 1, 0, 0])  # Generate 5 samples, conditioning on income labels
#   synthetic_embeddings = cgan.generate(num_samples=5, labels=sample_labels.to(DEVICE))

#   # ---------------------------
#   # Step 7: Validate Outputs
#   # ---------------------------
#   print(f"Generated synthetic embeddings shape: {synthetic_embeddings.shape}")
#   assert synthetic_embeddings.shape == (5, EMBEDDING_DIM), "Error: Output dimensions incorrect!"

#   print("[Test]: cGAN test completed successfully!")
