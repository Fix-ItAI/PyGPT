import torch
from torch.utils.data import Dataset, DataLoader
import requests
from pathlib import Path

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


def read_text_files(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [Path(p) for p in paths]
    text_pieces = []
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    for path in sorted(paths):
        if not path.exists():
            alt_path = Path.cwd() / path
            if alt_path.exists():
                path = alt_path
            else:
                alt_path = script_dir / path
                if alt_path.exists():
                    path = alt_path
                else:
                    matches = list(repo_root.rglob(path.name))
                    if matches:
                        if len(matches) > 1:
                            print(f"Warning: found multiple matches for {path.name}; using first match: {matches[0]}")
                        path = matches[0]
        if path.is_dir():
            files = sorted(path.rglob('*.txt'))
            if not files:
                raise FileNotFoundError(f"No .txt files found in directory: {path}")
            for file_path in files:
                text_pieces.append(file_path.read_text(encoding='utf-8', errors='ignore'))
        elif path.is_file():
            text_pieces.append(path.read_text(encoding='utf-8', errors='ignore'))
        else:
            raise FileNotFoundError(
                f"Training data path not found: {path}. Checked cwd: {Path.cwd()} and script dir: {script_dir}"
            )
    return "\n".join(text_pieces)


class TokenDataset(Dataset):
    """
    A simple dataset that converts a string of text into a sequence of word tokens.
    """

    def __init__(self, text, block_size):
        tokens = text.split()
        self.vocab_size = len(set(tokens))
        self.stoi = {token: i for i, token in enumerate(sorted(set(tokens)))}
        self.itos = {i: token for token, i in self.stoi.items()}

        self.data = [self.stoi[token] for token in tokens]
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_dataloaders(block_size=128, batch_size=32, file_paths=None, num_workers=None, pin_memory=None):
    if file_paths:
        text = read_text_files(file_paths)
    else:
        text = download_data()

    dataset = TokenDataset(text, block_size)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    if num_workers is None:
        num_workers = 2 if torch.cuda.is_available() else 0
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, dataset.vocab_size, dataset.itos
