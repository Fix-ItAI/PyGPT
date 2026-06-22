import torch
from model import GPT
from data import get_dataloaders

# Must use the same hyperparameters as in train.py
block_size = 128
n_embd = 384
n_head = 6
n_layer = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the data to get vocab size and the itos mapping
_, _, vocab_size, itos = get_dataloaders(block_size, batch_size=1)

# Instantiate the model
model = GPT(vocab_size, n_embd, n_head, n_layer, block_size)
# Load the trained weights
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
model.eval() # Set to evaluation mode

# Function to decode a list of indices into a string
def decode(idx):
return ''.join([itos[i] for i in idx])

# Start with a prompt (can be empty)
prompt = "\n" # Start with a newline to begin a new line of text
context = torch.tensor([itos[ch] for ch in prompt], dtype=torch.long, device=device).unsqueeze(0) # shape (1, T)

# Generate new text
print("--- Generating text ---")
generated_indices = model.generate(context, max_new_tokens=500, block_size=block_size)
generated_text = decode(generated_indices[0].tolist())
print(generated_text)
print("--- End of generation ---")