# Adult Income Dataset GAN/cGAN Generator

This project implements both a Generative Adversarial Network (GAN) and a Conditional GAN (cGAN) to generate synthetic data based on the Adult Income dataset. The implementation includes an autoencoder for dimensionality reduction and feature learning.

## Project Structure

```
├── main.py              # Main training and evaluation script
├── gan.py              # GAN implementation
├── cgan.py             # Conditional GAN implementation
├── dataset.py          # Dataset loading and preprocessing
├── config.py           # Configuration parameters
├── autoencoder.py      # Autoencoder for dimensionality reduction
├── models/             # Directory for saved model checkpoints
│   ├── gan_checkpoint.pt
│   └── cgan_checkpoint.pt
```

## Features

- **Data Processing**:
  - Handles both numerical and categorical features
  - Automatic preprocessing including normalization and one-hot encoding
  - Stratified splitting for balanced training

- **Model Architecture**:
  - Autoencoder for dimensionality reduction
  - Standard GAN implementation
  - Conditional GAN for controlled generation
  - Checkpoint saving and loading

- **Evaluation**:
  - Distribution comparison between real and synthetic data
  - Visualization of both numerical and categorical features
  - Support for model evaluation metrics

## Requirements

```
torch
pandas
numpy
scikit-learn
seaborn
matplotlib
tqdm
tensorboard
```

## Usage

1. Configure the model parameters in `config.py`:
```python
# Choose model type ('gan' or 'cgan')
MODEL_TYPE = 'cgan'

# Set hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE_GEN = 0.0002
LEARNING_RATE_DISC = 0.0002
```

2. Run the training script:
```bash
python main.py
```

3. Check the generated plots in the `plots/` directory to compare distributions.

## Model Details

### Autoencoder
- Reduces input dimensionality while preserving important features
- Used as a preprocessing step for both GAN and cGAN

### GAN
- Generates synthetic data samples
- Uses the autoencoder's latent space for better feature learning
- Includes batch normalization and dropout for stability

### Conditional GAN (cGAN)
- Generates synthetic data conditioned on class labels
- Allows controlled generation of samples for specific income categories
- Enhanced architecture with label embedding and conditioning

## Training Process

1. Data preprocessing and splitting
2. Autoencoder training for dimensionality reduction
3. GAN/cGAN training with the following steps:
   - Generate fake samples/embeddings
   - Train discriminator on real/fake samples
   - Train generator to fool discriminator
4. Distribution comparison and evaluation

## Results

The models generate synthetic data that preserves:
- Feature distributions of numerical variables
- Category proportions of categorical variables
- Relationships between features
- Class balance (for cGAN)

## Checkpoints

The models save checkpoints during training, which can be loaded for:
- Continuing training from a previous state
- Generating new synthetic samples
- Model evaluation and comparison
