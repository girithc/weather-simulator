import os
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
INPUT_TIF_FILE   = 'run2.tif'   # features at time t
TARGET_TIF_FILE  = 'run3.tif'   # label at time t+1
INPUT_BANDS      = list(range(1, 17))
TARGET_BAND      = 1
IMG_HEIGHT       = 256
IMG_WIDTH        = 256
IN_CHANNELS      = len(INPUT_BANDS)
OUT_CHANNELS     = 1
BATCH_SIZE       = 4    # will auto-reduce to 1 if you only have 1 sample
EPOCHS           = 50
VALIDATION_SPLIT = 0.2
LEARNING_RATE    = 1e-4

# --- Utility: load & normalize a GeoTIFF patch ---
def load_and_preprocess_tif(fp, band_indices, H, W, normalize=True):
    try:
        band_list = band_indices if isinstance(band_indices, list) else [band_indices]
        with rasterio.open(fp) as src:
            data = src.read(
                band_list,
                out_shape=(len(band_list), H, W),
                resampling=Resampling.bilinear
            )
        # to (H, W, C)
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

    # Encoder
    def conv_block(x, filters, dropout):
        x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = Dropout(dropout)(x)
        x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        return x

    c1 = conv_block(inputs,  16, 0.1); p1 = MaxPooling2D((2,2))(c1)
    c2 = conv_block(p1,      32, 0.1); p2 = MaxPooling2D((2,2))(c2)
    c3 = conv_block(p2,      64, 0.2); p3 = MaxPooling2D((2,2))(c3)
    c4 = conv_block(p3,     128, 0.2); p4 = MaxPooling2D((2,2))(c4)

    # Bottleneck
    bn = conv_block(p4,      256, 0.3)

    # Decoder
    def up_block(x, skip, filters, dropout):
        x = UpSampling2D((2,2))(x)
        x = concatenate([x, skip])
        x = conv_block(x, filters, dropout)
        return x

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
    print("Loading and preprocessing data…")
    X = load_and_preprocess_tif(INPUT_TIF_FILE, INPUT_BANDS, IMG_HEIGHT, IMG_WIDTH)
    y = load_and_preprocess_tif(TARGET_TIF_FILE, TARGET_BAND, IMG_HEIGHT, IMG_WIDTH)

    if X is None or y is None:
        print("Failed to load data. Exiting.")
        return

    # shape (1,H,W,C)
    X = np.expand_dims(X, axis=0)
    y = np.expand_dims(y, axis=0)

    # train/val split (with only 1 sample, just duplicate)
    if X.shape[0] == 1:
        print("Warning: single sample—using it for both train & val.")
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
