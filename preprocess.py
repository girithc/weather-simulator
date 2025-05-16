import os
import shutil
import numpy as np
import rasterio
from rasterio.enums import Resampling
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    concatenate, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- Configuration Parameters ---
# List of feature (time t) and label (time t+1) GeoTIFFs
INPUT_TIF_FILES  = ['data/run2.tif', 'data/run4.tif']
TARGET_TIF_FILES = ['data/run3.tif', 'data/run5.tif']
INPUT_BANDS      = list(range(1, 17))
TARGET_BAND      = 1
IMG_HEIGHT       = 256
IMG_WIDTH        = 256
IN_CHANNELS      = len(INPUT_BANDS)
OUT_CHANNELS     = 1
BATCH_SIZE       = 4  # will auto-reduce if too large for dataset
EPOCHS           = 50
VALIDATION_SPLIT = 0.2
LEARNING_RATE    = 1e-4

# Path to marplot library file to copy into project folder\NAMESPACE_ERROR
MARPLOT_LIB_FILE = 'marplot.lib'
DEST_FOLDER      = 'data'

# --- Utility: copy marplot library to destination folder ---
def save_marplot_lib(src, dst_folder):
    try:
        os.makedirs(dst_folder, exist_ok=True)
        dst = os.path.join(dst_folder, os.path.basename(src))
        shutil.copy(src, dst)
        print(f"Copied {src} to {dst}")
    except Exception as e:
        print(f"Error copying marplot lib: {e}")

# --- Utility: load & normalize a GeoTIFF patch ---
def load_and_preprocess_tif(fp, band_indices, H, W, normalize=True):
    try:
        bands = band_indices if isinstance(band_indices, list) else [band_indices]
        with rasterio.open(fp) as src:
            data = src.read(
                bands,
                out_shape=(len(bands), H, W),
                resampling=Resampling.bilinear
            )
        # reshape to (H, W, C)
        data = np.transpose(data, (1, 2, 0)).astype(np.float32)

        if normalize:
            for i in range(data.shape[-1]):
                band = data[..., i]
                mn, mx = band.min(), band.max()
                data[..., i] = (band - mn) / (mx - mn) if mx > mn else 0.0

        return data

    except Exception as e:
        print(f"Error loading {fp}: {e}")
        return None

# --- Build a 4-level U-Net ---
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS), num_classes=OUT_CHANNELS):
    inputs = Input(input_size)

    def conv_block(x, filters, dropout):
        x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = Dropout(dropout)(x)
        x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        return x

    # Encoder
    c1 = conv_block(inputs,  16, 0.1); p1 = MaxPooling2D((2,2))(c1)
    c2 = conv_block(p1,      32, 0.1); p2 = MaxPooling2D((2,2))(c2)
    c3 = conv_block(p2,      64, 0.2); p3 = MaxPooling2D((2,2))(c3)
    c4 = conv_block(p3,     128, 0.2); p4 = MaxPooling2D((2,2))(c4)

    # Bottleneck
    bn = conv_block(p4,      256, 0.3)

    def up_block(x, skip, filters, dropout):
        x = UpSampling2D((2,2))(x)
        x = concatenate([x, skip])
        x = conv_block(x, filters, dropout)
        return x

    # Decoder
    u4 = up_block(bn, c4, 128, 0.2)
    u3 = up_block(u4, c3,  64, 0.2)
    u2 = up_block(u3, c2,  32, 0.1)
    u1 = up_block(u2, c1,  16, 0.1)

    outputs = Conv2D(num_classes, 1, activation='sigmoid', name='output')(u1)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Main ---
def main():
    # Copy marplot library into project
    save_marplot_lib(MARPLOT_LIB_FILE, DEST_FOLDER)

    print("Loading and preprocessing data…")
    X_list, y_list = [], []
    for inp_fp, tgt_fp in zip(INPUT_TIF_FILES, TARGET_TIF_FILES):
        Xp = load_and_preprocess_tif(inp_fp, INPUT_BANDS, IMG_HEIGHT, IMG_WIDTH)
        yp = load_and_preprocess_tif(tgt_fp, TARGET_BAND, IMG_HEIGHT, IMG_WIDTH)
        if Xp is None or yp is None:
            print("Failed to load one of the files: exiting.")
            return
        X_list.append(np.expand_dims(Xp, axis=0))
        y_list.append(np.expand_dims(yp, axis=0))

    # combine samples
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # train/val split
    if X.shape[0] < 2:
        print("Warning: less than two samples—using same for train & val.")
        X_train, X_val = X, X
        y_train, y_val = y, y
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VALIDATION_SPLIT, random_state=42
        )

    print(f"Train samples: {len(X_train)}  Val samples: {len(X_val)}")

    # adjust batch size if needed
    bs = min(BATCH_SIZE, len(X_train))
    if bs < BATCH_SIZE:
        print(f"Batch size too large for dataset; using batch_size={bs}")

    # build & summarize
    model = unet_model()
    model.summary()

    # train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=bs,
        epochs=EPOCHS
    )

    # plot loss & accuracy
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'],   label='val loss')
    plt.legend(); plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'],     label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend(); plt.title('Accuracy')

    plt.show()

if __name__ == "__main__":
    main()
