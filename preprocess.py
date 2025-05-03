import os
import glob
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# 1. PARAMETERS
RASTER_DIR     = '/rasters/'    # folder with files like 0001.tif, 0002.tif, …
STATIC_LAYER   = '/land2.tif'   # optional initial conditions
SEQ_LEN        = 5                       # # of past timesteps fed into model
BATCH_SIZE     = 4
LR             = 1e-3
EPOCHS         = 10
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. DATASET CLASS
class SpatioTemporalDataset(Dataset):
    def __init__(self, files, seq_len, static_path=None):
        # sort by timestamp in filename
        self.files = sorted(files)
        self.seq_len = seq_len
        # load static once
        self.static = None
        if static_path:
            with rasterio.open(static_path) as src:
                self.static = src.read().astype(np.float32)  # (C,H,W)
                self.static = self.static / (self.static.max() + 1e-6)
        # convert all rasters to arrays
        self.arrays = []
        for fp in self.files:
            with rasterio.open(fp) as src:
                arr = src.read(1).astype(np.float32)  # single‐band fire mask / intensity
                self.arrays.append(arr / 255.0)        # normalize pixel intensities

    def __len__(self):
        # # of seqs possible (predict t from t−seq_len … t−1)
        return len(self.arrays) - self.seq_len

    def __getitem__(self, i):
        # stack past seq_len frames: shape (seq_len, H, W)
        x = np.stack(self.arrays[i : i + self.seq_len], axis=0)
        # add static channels if present
        if self.static is not None:
            # broadcast static to each timestep
            static_seq = np.repeat(self.static[None], self.seq_len, axis=0)
            x = np.concatenate([x[:,None,:,:], static_seq], axis=1)
            # → shape (seq_len, C, H, W)
        else:
            x = x[:,None,:,:]  # add channel dim
        # target is next frame
        y = self.arrays[i + self.seq_len]
        # to torch
        return torch.from_numpy(x), torch.from_numpy(y[None,:,:])

# 3. SPLIT FILE LIST
all_files = glob.glob(os.path.join(RASTER_DIR, '*.tif'))
n = len(all_files)
train_f = all_files[: int(0.7*n)]
val_f   = all_files[int(0.7*n): int(0.85*n)]
test_f  = all_files[int(0.85*n): ]

train_ds = SpatioTemporalDataset(train_f, SEQ_LEN, STATIC_LAYER)
val_ds   = SpatioTemporalDataset(val_f,   SEQ_LEN, STATIC_LAYER)
test_ds  = SpatioTemporalDataset(test_f,  SEQ_LEN, STATIC_LAYER)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# 4. MODEL (ConvLSTM cell unrolled over sequence)
class ConvLSTMNet(nn.Module):
    def __init__(self, in_ch, hidden_ch=16, kernel=3):
        super().__init__()
        # simple conv‐LSTM: one layer
        self.clstm = nn.LSTM(
            input_size=in_ch * 64 * 64,  # flatten H×W
            hidden_size=hidden_ch * 64 * 64,
            batch_first=True
        )
        self.conv_out = nn.Conv2d(hidden_ch, 1, kernel_size=kernel, padding=kernel//2)

    def forward(self, x):
        # x: (B, seq_len, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B, T, C*H*W)
        out, _ = self.clstm(x)              # (B, T, hidden)
        last = out[:, -1, :].view(B, -1, H, W)
        y_hat = torch.sigmoid(self.conv_out(last))
        return y_hat

# instantiate
# infer in_ch from dataset
sample_x, _ = next(iter(train_loader))
in_ch = sample_x.shape[2]
model = ConvLSTMNet(in_ch).to(DEVICE)
loss_fn = nn.BCELoss()
opt     = optim.Adam(model.parameters(), lr=LR)

# 5. TRAIN & VALIDATE
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        y_pred = model(x)
        loss   = loss_fn(y_pred, y)
        loss.backward()
        opt.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x)
            val_loss += loss_fn(y_pred, y).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch:2d} — train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")

# 6. TEST METRIC (e.g., IoU)
def iou_score(pred, target, thresh=0.5):
    pred_bin = (pred > thresh).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter + 1e-6
    return inter / union

model.eval()
ious = []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_hat = model(x)
        for yh, yt in zip(y_hat, y):
            ious.append(iou_score(yh, yt).item())
print(f"Test IoU: {np.mean(ious):.4f}")