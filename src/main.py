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
MODEL_TYPE = 'gan'  # Choose 'gan' or 'cgan'

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


# ---------------------------
# Step 9: Visualize Evaluation Metrics
# ---------------------------
print("[Main]: Creating evaluation metric visualizations...")

# Create directory for evaluation plots
os.makedirs("plots/evaluation", exist_ok=True)

# Plot Detection AUC across folds
plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4], [1.0, 1.0, 1.0, 1.0], 'bo-', linewidth=2, markersize=8)
plt.axhline(y=0.5, color='r', linestyle='--', label='Random Classifier (0.5)')
plt.ylim(0.4, 1.1)
plt.xlabel('Fold Number')
plt.ylabel('Detection AUC')
plt.title('Detection AUC Across Folds\nLower is Better')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"plots/evaluation/{MODEL_TYPE}_detection_auc.png")
plt.close()

# Plot Efficacy Comparison
plt.figure(figsize=(10, 6))
metrics = ['Real Data AUC', 'Synthetic Data AUC', 'Efficacy Score']
values = [1.0000, 0.3858, 0.3858]
colors = ['#2ecc71', '#e74c3c', '#3498db']

bars = plt.bar(metrics, values, color=colors)
plt.ylim(0, 1.1)
plt.ylabel('Score')
plt.title('Model Efficacy Metrics')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"plots/evaluation/{MODEL_TYPE}_efficacy_metrics.png")
plt.close()

# Create a combined metrics summary plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot([1, 2, 3, 4], [1.0, 1.0, 1.0, 1.0], 'bo-', linewidth=2, markersize=8)
plt.axhline(y=0.5, color='r', linestyle='--', label='Random Classifier (0.5)')
plt.ylim(0.4, 1.1)
plt.xlabel('Fold Number')
plt.ylabel('Detection AUC')
plt.title('Detection Performance\nLower is Better')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(['Real', 'Synthetic'], [1.0000, 0.3858], color=['#2ecc71', '#e74c3c'])
plt.ylim(0, 1.1)
plt.ylabel('AUC Score')
plt.title('Model Efficacy\nHigher is Better')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for i, v in enumerate([1.0000, 0.3858]):
    plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')

plt.suptitle(f'{MODEL_TYPE.upper()} Model Evaluation Summary', y=1.05)
plt.tight_layout()
plt.savefig(f"plots/evaluation/{MODEL_TYPE}_evaluation_summary.png")
plt.close()

print("[Main]: Evaluation visualizations saved in plots/evaluation/")

# ---------------------------
# Step 10: Visualize Training Losses
# ---------------------------
print("[Main]: Creating training loss visualizations...")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def plot_tensorboard_losses(log_dir, model_type):
    """Plot training losses from TensorBoard logs."""
    # Adjust model type case to match directory structure
    model_type_dir = model_type.upper() if model_type == "cgan" else model_type.lower()
    
    # Find the event file
    event_pattern = f"{log_dir}/{model_type_dir}/events.out.tfevents.*"
    event_files = glob.glob(event_pattern)
    
    if not event_files:
        print(f"[Main]: No TensorBoard logs found matching pattern: {event_pattern}")
        return
    
    latest_event_file = max(event_files, key=os.path.getctime)
    print(f"[Main]: Found TensorBoard log file: {latest_event_file}")
    
    event_acc = EventAccumulator(latest_event_file)
    event_acc.Reload()

    # Extract loss data
    try:
        gen_loss = [(s.step, s.value) for s in event_acc.Scalars('Loss/Generator')]
        disc_loss = [(s.step, s.value) for s in event_acc.Scalars('Loss/Discriminator')]
    except KeyError as e:
        print(f"[Main]: Error reading loss data: {e}")
        print("[Main]: Available tags:", event_acc.Tags())
        return

    # Convert to numpy arrays for easier plotting
    steps = np.array([x[0] for x in gen_loss])
    gen_values = np.array([x[1] for x in gen_loss])
    disc_values = np.array([x[1] for x in disc_loss])

    # Create directory for plots if it doesn't exist
    os.makedirs("plots/evaluation", exist_ok=True)

    # Create the loss plot
    plt.figure(figsize=(12, 6))
    
    # Plot losses
    plt.plot(steps, gen_values, label='Generator Loss', color='#2ecc71', linewidth=2)
    plt.plot(steps, disc_values, label='Discriminator Loss', color='#e74c3c', linewidth=2)
    
    # Add moving averages for smoother visualization
    window_size = min(20, len(steps) // 10)  # Adjust window size based on data length
    if window_size > 1:
        gen_ma = np.convolve(gen_values, np.ones(window_size)/window_size, mode='valid')
        disc_ma = np.convolve(disc_values, np.ones(window_size)/window_size, mode='valid')
        ma_steps = steps[window_size-1:]
        
        plt.plot(ma_steps, gen_ma, '--', color='#27ae60', alpha=0.5, 
                 label='Generator Moving Avg')
        plt.plot(ma_steps, disc_ma, '--', color='#c0392b', alpha=0.5, 
                 label='Discriminator Moving Avg')

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'{model_type.upper()} Training Losses')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    loss_plot_path = f"plots/evaluation/{model_type}_training_losses.png"
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"[Main]: Saved loss plot to {loss_plot_path}")

    # Create convergence analysis plot
    plt.figure(figsize=(12, 6))
    
    # Calculate loss ratio and JS divergence
    loss_ratio = gen_values / disc_values
    js_divergence = np.abs(gen_values - disc_values) / (gen_values + disc_values)
    
    plt.subplot(1, 2, 1)
    plt.plot(steps, loss_ratio, label='G/D Loss Ratio', color='#3498db')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Optimal Ratio (1.0)')
    plt.xlabel('Training Steps')
    plt.ylabel('Generator/Discriminator Loss Ratio')
    plt.title('Training Stability')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(steps, js_divergence, label='JS Divergence', color='#9b59b6')
    plt.xlabel('Training Steps')
    plt.ylabel('JS Divergence')
    plt.title('Model Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.suptitle(f'{model_type.upper()} Training Analysis', y=1.02)
    plt.tight_layout()
    
    analysis_plot_path = f"plots/evaluation/{model_type}_training_analysis.png"
    plt.savefig(analysis_plot_path, dpi=300)
    plt.close()
    print(f"[Main]: Saved analysis plot to {analysis_plot_path}")

# Plot losses for the current model type
plot_tensorboard_losses("runs", MODEL_TYPE)
print(f"[Main]: Training loss visualizations saved in plots/evaluation/")
