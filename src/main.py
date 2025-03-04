import torch
import os
import numpy as np
from dataset import adultDataset, get_dataloader
from autoencoder import train_autoencoder
from gan import GAN
from cgan import ConditionalGAN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import Config

# ---------------------------
# Configuration
# ---------------------------
CONFIG = Config()
MODEL_TYPE = 'cgan'  # Choose 'gan' or 'cgan'

# ---------------------------
# Step 1: Load and Preprocess Dataset
# ---------------------------
print("[Main]: Initializing dataset...")
dataset = adultDataset(data_path=CONFIG.FULL_DATA_PATH, target_column=CONFIG.TARGET_COLUMN)
dataset.load_data()
dataset.preprocess()

# ---------------------------
# Step 2: Split and Prepare DataLoader
# ---------------------------
print("[Main]: Splitting dataset...")
train_set, _, _ = dataset.stratified_split(test_size=0.2, val_size=0.1, random_state=42)
train_loader = get_dataloader(train_set, batch_size=CONFIG.BATCH_SIZE, shuffle=True)

# ---------------------------
# Step 3: Train Autoencoder
# ---------------------------
print("[Main]: Training Autoencoder...")
autoencoder = train_autoencoder(
    train_loader=train_loader,
    input_dim=dataset.df.shape[1] - 1,  # Exclude target column
    embedding_dim=CONFIG.LATENT_DIM,
    num_epochs=CONFIG.AUTOENCODER_EPOCHS,
    device=CONFIG.DEVICE
)

# ---------------------------
# Step 4: Initialize GAN/cGAN Model
# ---------------------------
checkpoint_path = f"models/{MODEL_TYPE}_checkpoint.pt"

if MODEL_TYPE == "gan":
    print("[Main]: Initializing GAN...")
    model = GAN(
        noise_dim=CONFIG.NOISE_DIM,
        data_dim=dataset.df.shape[1] - 1,  # Exclude target column
        hidden_dim=CONFIG.HIDDEN_DIM,
        device=CONFIG.DEVICE,
        pretrained_path=checkpoint_path
    )
elif MODEL_TYPE == "cgan":
    print("[Main]: Initializing cGAN...")
    model = ConditionalGAN(
        noise_dim=CONFIG.NOISE_DIM,
        embedding_dim=CONFIG.LATENT_DIM,
        label_dim=CONFIG.NUM_CLASSES,
        hidden_dim=CONFIG.HIDDEN_DIM,
        device=CONFIG.DEVICE,
        pretrained_path=checkpoint_path
    )
else:
    raise ValueError("Invalid MODEL_TYPE. Choose 'gan' or 'cgan'.")

# ---------------------------
# Step 5: Train Model (Skip if checkpoint exists)
# ---------------------------
if not os.path.exists(checkpoint_path):
    print(f"[Main]: Training {MODEL_TYPE.upper()}...")
    model.train_model(
        train_loader=train_loader,
        autoencoder=autoencoder,
        epochs=CONFIG.EPOCHS,
        lr_gen=CONFIG.LEARNING_RATE_GEN,
        lr_disc=CONFIG.LEARNING_RATE_DISC,
        save_path=checkpoint_path
    )
else:
    print(f"[Main]: Using pre-trained {MODEL_TYPE.upper()} model from {checkpoint_path}")

# ---------------------------
# Step 6: Generate Synthetic Data
# ---------------------------
print("[Main]: Generating synthetic dataset...")
num_samples = len(train_set)

if MODEL_TYPE == "gan":
    synthetic_data = model.generate(num_samples=num_samples, autoencoder=autoencoder)
else:  # cGAN
    # Generate balanced dataset with equal class distribution
    samples_per_class = num_samples // CONFIG.NUM_CLASSES
    synthetic_data = []
    for class_idx in range(CONFIG.NUM_CLASSES):
        labels = torch.full((samples_per_class,), class_idx, device=CONFIG.DEVICE)
        class_samples = model.generate(
            num_samples=samples_per_class,
            labels=labels,
            autoencoder=autoencoder
        )
        synthetic_data.append(class_samples)
    synthetic_data = torch.cat(synthetic_data, dim=0)

# Convert to Pandas DataFrame
real_df = pd.DataFrame(
    dataset.df.iloc[:num_samples, :-1].values,
    columns=dataset.df.columns[:-1]
)
synthetic_df = pd.DataFrame(
    synthetic_data.cpu().numpy(),
    columns=dataset.df.columns[:-1]
)

# ---------------------------
# Step 7: Compare Distributions
# ---------------------------
print("[Main]: Comparing Real vs. Synthetic Distributions...")

# Create directory for plots
os.makedirs("../plots", exist_ok=True)

# Plot numerical features
numerical_features = dataset.numerical_columns
fig, axes = plt.subplots(len(numerical_features), 1, figsize=(12, 4 * len(numerical_features)))
for idx, feature in enumerate(numerical_features):
    sns.kdeplot(data=real_df[feature], ax=axes[idx], label='Real', color='blue')
    sns.kdeplot(data=synthetic_df[feature], ax=axes[idx], label='Synthetic', color='red')
    axes[idx].set_title(f'Distribution of {feature}')
    axes[idx].legend()
plt.tight_layout()
plt.savefig(f"plots/{MODEL_TYPE}_numerical_distributions.png")
plt.close()

# # Plot categorical features
# categorical_features = dataset.categorical_columns
# fig, axes = plt.subplots(len(categorical_features), 1, figsize=(12, 4 * len(categorical_features)))
# for idx, feature in enumerate(categorical_features):
#     real_counts = real_df[feature].value_counts(normalize=True)
#     synthetic_counts = synthetic_df[feature].value_counts(normalize=True)
#
#     pd.DataFrame({
#         'Real': real_counts,
#         'Synthetic': synthetic_counts
#     }).plot(kind='bar', ax=axes[idx])
#
#     axes[idx].set_title(f'Distribution of {feature}')
#     axes[idx].legend()
# plt.tight_layout()
# plt.savefig(f"plots/{MODEL_TYPE}_categorical_distributions.png")
# plt.close()
# After generating synthetic data and before plotting distributions, add:

# ---------------------------
# Step 8: Evaluate Model
# ---------------------------
print("[Main]: Evaluating model performance...")
from evaluate import evaluate_model

# Prepare data for evaluation
train_data = dataset.df.iloc[:, :-1].values
train_labels = dataset.df.iloc[:, -1].values
test_data = dataset.df.iloc[:, :-1].values
test_labels = dataset.df.iloc[:, -1].values

# Convert synthetic data to numpy if it's a tensor
if torch.is_tensor(synthetic_data):
    synthetic_data = synthetic_data.cpu().numpy()

# For cGAN, get the synthetic labels
if MODEL_TYPE == "cgan":
    synthetic_labels = np.repeat(np.arange(CONFIG.NUM_CLASSES),
                               len(synthetic_data) // CONFIG.NUM_CLASSES)
else:
    # For GAN, use the same distribution as training data
    synthetic_labels = np.random.choice(train_labels, size=len(synthetic_data))

# Run evaluation
results = evaluate_model(
    real_data=train_data,
    synthetic_data=synthetic_data,
    real_labels=train_labels,
    synthetic_labels=synthetic_labels,
    test_data=test_data,
    test_labels=test_labels
)

# Save results
with open(f"results/{MODEL_TYPE}_evaluation.txt", "w") as f:
    f.write(f"Model Type: {MODEL_TYPE}\n")
    f.write(f"Detection AUC: {results['detection_auc']:.4f}\n")
    f.write(f"Efficacy Score: {results['efficacy_score']:.4f}\n")

print(f"[Main]: Training and evaluation completed! Check plots/ directory for distribution comparisons.")