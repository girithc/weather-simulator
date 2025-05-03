
import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

# 1. PARAMETERS & DEVICE
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DYNAMIC_FILE = os.path.join(BASE_DIR, 'land2.tif')
STATIC_LAYER = None
SEQ_LEN      = 5
BATCH_SIZE   = 4
LR           = 1e-3
EPOCHS       = 10

DEVICE = (
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else
    'cpu'
)
print(f"Using device: {DEVICE}")

# 2. DATASET: multi-band TIFF -> frames
class MultiBandDataset(Dataset):
    def __init__(self, path, seq_len, static_path=None):
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)  # (bands, H, W)
        self.frames = [band / 255.0 for band in data]
        self.num_frames = len(self.frames)
        if self.num_frames <= seq_len:
            raise ValueError(f"Need > SEQ_LEN frames; got {self.num_frames}, SEQ_LEN={seq_len}")
        self.seq_len = seq_len

        self.static = None
        if static_path:
            with rasterio.open(static_path) as src:
                st = src.read().astype(np.float32)
            self.static = st / (st.max() + 1e-6)

    def __len__(self):
        return self.num_frames - self.seq_len

    def __getitem__(self, idx):
        seq = self.frames[idx:idx + self.seq_len]         # list of (H, W)
        x = np.stack(seq, axis=0)                         # (T, H, W)
        if self.static is not None:
            static_seq = np.repeat(self.static[None], self.seq_len, axis=0)
            x = np.concatenate([x[:, None], static_seq], axis=1)  # (T, C+1, H, W)
        else:
            x = x[:, None]                                 # (T, 1, H, W)
        y = self.frames[idx + self.seq_len][None]          # (1, H, W)
        return torch.from_numpy(x), torch.from_numpy(y)

# 3. SPLIT into train / val / test
dataset = MultiBandDataset(DYNAMIC_FILE, SEQ_LEN, STATIC_LAYER)
num_seq = len(dataset)
train_end = int(0.7 * num_seq)
val_end   = int(0.85 * num_seq)

train_ds = Subset(dataset, range(0, train_end))
val_ds   = Subset(dataset, range(train_end, val_end))
test_ds  = Subset(dataset, range(val_end, num_seq))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# 4. CONVLSTM Cell & Net
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel):
        super().__init__()
        self.hid_ch = hid_ch
        padding = kernel // 2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=padding)

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f)
        o = torch.sigmoid(o); g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch, H, W, device):
        return (torch.zeros(batch, self.hid_ch, H, W, device=device),
                torch.zeros(batch, self.hid_ch, H, W, device=device))

class ConvLSTMNet(nn.Module):
    def __init__(self, in_ch, hid_ch=16, kernel=3):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hid_ch, kernel)
        self.conv_out = nn.Conv2d(hid_ch, 1, kernel, padding=kernel//2)

    def forward(self, x):
        B, T, C, H, W = x.size()
        h, c = self.cell.init_state(B, H, W, x.device)
        for t in range(T):
            h, c = self.cell(x[:, t], (h, c))
        return torch.sigmoid(self.conv_out(h))  # (B, 1, H, W)

# 5. INSTANTIATE model, loss, optimizer
sample_x, _ = next(iter(train_loader))
_, T, C, H, W = sample_x.shape
model   = ConvLSTMNet(in_ch=C).to(DEVICE)
loss_fn = nn.BCELoss()
opt     = optim.Adam(model.parameters(), lr=LR)

# 6. TRAIN & VALIDATE
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            val_loss += loss_fn(model(x), y).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch} — train: {train_loss:.4f}, val: {val_loss:.4f}")

# 7. TEST IoU
def iou(p, t, thr=0.5):
    pbin = (p > thr).float()
    inter = (pbin * t).sum()
    union = pbin.sum() + t.sum() - inter + 1e-6
    return inter / union

model.eval()
ious = []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x)
        for pi, yi in zip(preds, y):
            ious.append(iou(pi, yi).item())

print(f"Test IoU: {np.mean(ious):.4f}")
