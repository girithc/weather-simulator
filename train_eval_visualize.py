
import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import time
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim

# PARAMETERS & DEVICE
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, 'tif_folder')
SEQ_LEN      = 5
BATCH_SIZE   = 4
LR           = 1e-4 # smaller learning rate
EPOCHS       = 12
RESIZE_SHAPE = (512, 512)

DEVICE = (
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else
    'cpu'
)
print(f"Using device: {DEVICE}")

# Reset device cache
if DEVICE=='cuda':
    torch.cuda.empty_cache()
elif DEVICE=='mps':
    torch.mps.empty_cache()


# DATASET
class MultiBandDataset(Dataset):
    def __init__(self, dir_path, seq_len):
        self.seq_len = seq_len
        self.frames = []
        tif_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.tif')])
        if not tif_files:
            raise ValueError("No .tif files found in the directory")
        
        for fname in tif_files:
            with rasterio.open(os.path.join(dir_path, fname)) as src:
                data = src.read().astype(np.float32)
                # Normalize bands for better convergence
                for band in data:
                    resized_band = resize(band, RESIZE_SHAPE, anti_aliasing=True)
                    norm_band = (resized_band - resized_band.min()) / (resized_band.max() - resized_band.min() + 1e-8)
                    self.frames.append(norm_band)

        self.num_frames = len(self.frames)
        if self.num_frames <= seq_len:
            raise ValueError("Not enough frames for the given sequence length.")

    def __len__(self):
        return self.num_frames - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(np.stack(self.frames[idx:idx + self.seq_len]), dtype=torch.float32)
        y = torch.tensor(self.frames[idx + self.seq_len], dtype=torch.float32)
        return x, y

# MODEL
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# Change hidden_dim from 32 to 16 for a smaller model
# Better for smaller datasets
class ConvLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.output_layer = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, x):
        b, t, h, w = x.size()
        h_cur = torch.zeros(b, 16, h, w, device=x.device)
        c_cur = torch.zeros(b, 16, h, w, device=x.device)
        for i in range(t):
            x_t = x[:, i:i+1, :, :]
            h_cur, c_cur = self.cell(x_t, h_cur, c_cur)
        out = self.output_layer(h_cur)
        return out.squeeze(1)

# TRAINING
def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# EVALUATION
def evaluate(model, dataloader, criterion):
    model.eval()
    total_mse = 0
    total_mae = 0
    ssim_scores = []
    samples = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)

            mse = criterion(preds, y)
            mae = torch.mean(torch.abs(preds - y)).item()
            total_mse += mse.item()
            total_mae += mae

            ssim_score = ssim(preds[0].cpu().numpy(), y[0].cpu().numpy(), data_range=1.0)
            ssim_scores.append(ssim_score)

            
            samples.append((preds.cpu().numpy(), y.cpu().numpy()))
    return total_mse / len(dataloader), total_mae / len(dataloader), np.mean(ssim_scores), samples

# MAIN
dataset = MultiBandDataset(DATA_DIR, SEQ_LEN)
test_size = int(0.2 * len(dataset))
train_dataset = Subset(dataset, list(range(len(dataset) - test_size)))
test_dataset = Subset(dataset, list(range(len(dataset) - test_size, len(dataset))))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = ConvLSTM().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
test_mse_scores = []
test_mae_scores = []
ssim_epochs = []

for epoch in range(EPOCHS):
    loss = train(model, train_loader, criterion, optimizer)
    train_losses.append(loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {loss:.4f}")

    test_mse, test_mae, test_ssim, test_samples = evaluate(model, test_loader, criterion)
    test_mse_scores.append(test_mse)
    test_mae_scores.append(test_mae)
    ssim_epochs.append(test_ssim)

print(f"Final MSE: {test_mse_scores[-1]:.4f}")
print(f"Final MAE: {test_mae_scores[-1]:.4f}")
print(f"Final SSIM: {ssim_epochs[-1]:.4f}")    
print(f"Average MSE: {np.mean(test_mse_scores):.4f}")
print(f"Average MAE: {np.mean(test_mae_scores):.4f}")
print(f"Average SSIM: {np.mean(ssim_epochs):.4f}")

# Results variable
TEST_NAME='final_run'
os.mkdir(TEST_NAME)
TEST_DIR = BASE_DIR + '/' + TEST_NAME

# VISUALIZATION
# 1. Train/Test Loss Plot
plt.figure()
plt.plot(range(1, EPOCHS+1), train_losses, label='Train', color='blue')
plt.plot(range(1, EPOCHS+1), test_mse_scores, label='Validation', color='orange')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Model Loss')
plt.legend()
plt.savefig(os.path.join(TEST_DIR, 'loss.png'))

# 2. MAE Plot
plt.figure()
plt.plot(range(1, EPOCHS+1), test_mae_scores, label='MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE Scores')
plt.title('MAE Over Epochs')
plt.legend()
plt.savefig(os.path.join(TEST_DIR, 'mae.png'))

# 3. SSIM Plot
plt.figure()
plt.plot(range(1, EPOCHS+1), ssim_epochs, label='SSIM')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.title('SSIM Over Epochs')
plt.legend()
plt.savefig(os.path.join(TEST_DIR, 'ssim.png'))


# 4. Prediction Visualization
for i in range(len(test_samples)):
    preds, targets = test_samples[i]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Sample {i+1}")
    axes[0].imshow(targets[0], cmap='viridis')
    axes[0].set_title('Actual Frame')
    axes[1].imshow(preds[0], cmap='viridis')
    axes[1].set_title('Predicted Frame')
    plt.tight_layout()
    plt.savefig(os.path.join(TEST_DIR, f'result{i+1}.png'))
