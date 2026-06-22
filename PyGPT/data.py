import torch
from torch.utils.data import Dataset, DataLoader
import requests
import numpy as np

# Let's download a public domain book (Macbeth)
DATA_URL = "https://www.gutenberg.org/files/1533/1533-0.txt"

def download_data():
    print("Downloading data...")
    response = requests.get(DATA_URL)
    text = response.text
    # A simple clean-up to remove some of the Project Gutenberg header/footer
    start_marker = "THIS ELECTRONIC VERSION"
    end_marker = "END OF THIS PROJECT GUTENBERG EBOOK"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    return text


class CharDataset(Dataset):
    """
    A simple dataset that converts a string of text into a sequence of character indices.
    """

    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = { ch: i for i, ch in enumerate(chars) } # String To Index
        self.itos = { i: ch for i, ch in enumerate(chars) } # Index To String

        self.data = [self.stoi[ch] for ch in text]
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Grab a chunk of characters of length block_size from the data
        chunk = self.data[idx:idx + self.block_size]
        # The input (x) is all characters in the chunk except the last one
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        # The target (y) is all characters in the chunk except the first one
        # We are training the model to predict the *next* character.
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_dataloaders(block_size=128, batch_size=32, num_workers=None, pin_memory=None):
    text = download_data()
    dataset = CharDataset(text, block_size)
    # Split into train and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Auto-configure for available hardware
    if num_workers is None:
        num_workers = 2 if torch.cuda.is_available() else 0
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, dataset.vocab_size, dataset.itos