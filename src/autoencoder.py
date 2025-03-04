
import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    """ Autoencoder with encoder-decoder structure """
    def __init__(self, input_dim, embedding_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)  # Compressed representation
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Tanh()  # Reconstruct original input
        )

    def forward(self, x):
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)
        return reconstruction

def train_autoencoder(train_loader, input_dim, embedding_dim=32, num_epochs=50, lr=0.001, device="cpu"):
    """ Train an Autoencoder on real data """
    model = Autoencoder(input_dim=input_dim, embedding_dim=embedding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        loss_epoch = 0
        for real_data, _ in train_loader:
            real_data = real_data.to(device)
            reconstruction = model(real_data)
            loss = criterion(reconstruction, real_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

        print(f"[Epoch {epoch+1}/{num_epochs}] Autoencoder Loss: {loss_epoch / len(train_loader):.4f}")

    return model

if __name__ == "__main__":
    CONFIG = Config()
    # ---------------------------
    # 1. Load and Preprocess Data
    # ---------------------------
    print("[Main]: Initializing dataset...")
    dataset = adultDataset(data_path=CONFIG.FULL_DATA_PATH, target_column=CONFIG.TARGET_COLUMN)
    dataset.load_data()
    dataset.preprocess()

    print("[Main]: Performing stratified train-val-test split...")
    train_set, val_set, test_set = dataset.stratified_split(val_size=CONFIG.VAL_RATIO, test_size=CONFIG.TEST_RATIO, random_state=CONFIG.SEED)

    print("[Main]: Creating DataLoaders...")
    train_loader = get_dataloader(train_set, batch_size=CONFIG.BATCH_SIZE, shuffle=True)

    # ---------------------------
    # 2. Train Autoencoder
    # ---------------------------
    print("[Main]: Training Autoencoder on real data...")
    autoencoder = train_autoencoder(
        train_loader=train_loader,
        input_dim=108,  # Matches dataset feature size
        embedding_dim=CONFIG.LATENT_DIM,
        num_epochs=2,
        device=CONFIG.DEVICE
    )

