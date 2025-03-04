
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy.io import arff

class adultDataset(Dataset):
    """
    A PyTorch Dataset class including preprocessing, stratified train-validation-test splitting.
    """
    def __init__(self, data_path, target_column):
        self.data_path = data_path
        self.df = None
        self.target_column = target_column
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.numerical_columns = []
        self.categorical_columns = []

    def load_data(self):
        """Load and decode ARFF file into a Pandas DataFrame."""
        data, meta = arff.loadarff(self.data_path)
        df = pd.DataFrame(data)

        # Decode categorical values
        for col in df.select_dtypes(['object', 'category']):
            df[col] = df[col].str.decode('utf-8')

        self.df = df

    def preprocess(self):
        """Perform normalization and one-hot encoding."""
        # Store column types before preprocessing
        self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_columns = self.df.select_dtypes(include=['number']).columns.tolist()

        if self.target_column in self.categorical_columns:
            self.categorical_columns.remove(self.target_column)

        self.df[self.target_column] = self.df[self.target_column].astype(str)

        # Convert target column to numerical labels
        self.df[self.target_column], _ = pd.factorize(self.df[self.target_column])

        # Normalize continuous features
        self.df[self.numerical_columns] = self.scaler.fit_transform(self.df[self.numerical_columns])

        # Replace '?' with 'Unknown' before encoding
        self.df[self.categorical_columns] = self.df[self.categorical_columns].replace('?', 'Unknown')

        # One-hot encode categorical features
        self.df = pd.get_dummies(self.df, columns=self.categorical_columns, dtype=int)

        # Update categorical columns to include one-hot encoded columns
        self.categorical_columns = [col for col in self.df.columns 
                                  if col not in self.numerical_columns 
                                  and col != self.target_column]

    def stratified_split(self, val_size=0.1, test_size=0.2, random_state=42):
        """Perform stratified train-test-validation split while maintaining target distribution."""
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        # train - test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        # adjust for validation from the train set proportion
        new_val_size = val_size / (1 - test_size)

        # train - validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=new_val_size, stratify=y_train, random_state=random_state
        )

        return PreprocessorDataset(X_train, y_train), PreprocessorDataset(X_val, y_val), PreprocessorDataset(X_temp, y_temp)


class PreprocessorDataset(Dataset):
    """Custom Dataset Wrapper."""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)
        target = torch.tensor(int(self.y.iloc[idx]), dtype=torch.long)  # Ensure target is an integer
        return features, target


def get_dataloader(dataset, batch_size=32, shuffle=True):
    """Wraps the dataset in a DataLoader."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


import torch
import pandas as pd



if __name__ == "__main__":
    # ---------------------------
    # 1. Load and Preprocess Data
    # ---------------------------
    print("[Dataset Example]: Initializing dataset...")
    CONFIG = Config()
    dataset = adultDataset(data_path=CONFIG.FULL_DATA_PATH, target_column=CONFIG.TARGET_COLUMN)

    print("[Dataset Example]: Loading data from ARFF file...")
    dataset.load_data()

    print("[Dataset Example]: Preprocessing data (scaling, encoding)...")
    dataset.preprocess()

    # ---------------------------
    # 2. Perform Stratified Split
    # ---------------------------
    print("[Dataset Example]: Performing stratified train-val-test split...")
    train_set, val_set, test_set = dataset.stratified_split(
        val_size=CONFIG.VAL_RATIO, test_size=CONFIG.TEST_RATIO, random_state=CONFIG.SEED
    )

    # ---------------------------
    # 3. Check Class Distributions
    # ---------------------------
    print("[Dataset Example]: Checking class distributions after split...")

    train_labels = [train_set[i][1].item() for i in range(len(train_set))]
    val_labels = [val_set[i][1].item() for i in range(len(val_set))]
    test_labels = [test_set[i][1].item() for i in range(len(test_set))]

    print(f"Train Set ({len(train_labels)} samples):\n{pd.Series(train_labels).value_counts(normalize=True)}")
    print(f"Validation Set ({len(val_labels)} samples):\n{pd.Series(val_labels).value_counts(normalize=True)}")
    print(f"Test Set ({len(test_labels)} samples):\n{pd.Series(test_labels).value_counts(normalize=True)}")

    # ---------------------------
    # 4. Create DataLoaders
    # ---------------------------
    print("[Dataset Example]: Creating DataLoaders...")
    train_loader = get_dataloader(train_set, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(val_set, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(test_set, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

    # ---------------------------
    # 5. Check a Batch from DataLoader
    # ---------------------------
    print("[Dataset Example]: Checking a batch from the training DataLoader...")
    for i, (batch_X, batch_y) in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"Features shape: {batch_X.shape}, Labels shape: {batch_y.shape}")
        print(f"First batch features (first 5 samples):\n{batch_X[:5]}")
        print(f"First batch labels (first 5 samples):\n{batch_y[:5]}")
        break  # Only check the first batch
