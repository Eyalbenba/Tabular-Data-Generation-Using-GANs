import os
import torch

class Config:
    """Configuration class to store hyperparameters and settings."""
    # Paths:
    DATA_DIR_PATH = 'data'
    FULL_DATA_PATH = os.path.join(DATA_DIR_PATH, 'adult.arff')
    TARGET_COLUMN = 'income'
    TRAINED_MODELS_DIR_PATH = 'Trained Models'
    MODEL_NAME = 'cgan'  # Change between model types ['gan', 'cgan']
    SAVE_PATH = os.path.join(TRAINED_MODELS_DIR_PATH, MODEL_NAME)
    PRETRAIN_PATH = os.path.join(SAVE_PATH, 'best_model.pth')

    # Data Config:
    NUM_CLASSES = 2     # Number of classes in the dataset, for cGAN
    LABEL_RATIO = {0: 0.76, 1: 0.24}    # The ratio of the labels, manually derived from dataset, for generation purposes.
    BATCH_SIZE = 32
    VAL_RATIO = 0.2     # Ratio out of the training dataset
    TEST_RATIO = 0.2    # Ratio out of the full dataset
    SEED = 42   # Change the seed and check influence on the model
    HIDDEN_DIM = 128
    # Model Config
    DATA_DIM = 108 # The size of the feature vector
    NOISE_DIM = 32  # The size of the initial noise vector
    LATENT_DIM = 32 # The dimension of the latent (encoding) dimension
    LEARNING_RATE = 0.0005
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    AUTOENCODER_EPOCHS = 10
    EPOCHS = 100
    LEARNING_RATE_GEN = 0.0001
    LEARNING_RATE_DISC = 0.0001
