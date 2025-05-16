import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    concatenate, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
import matplotlib.pyplot as plt

# --- Configuration Parameters ---
INPUT_TIF_FILES  = ['data/run2.tif', 'data/run4.tif']
TARGET_TIF_FILES = ['data/run3.tif', 'data/run5.tif']
INPUT_BANDS      = list(range(1, 17))
TARGET_BAND      = 1
IMG_HEIGHT       = 256
IMG_WIDTH        = 256
OUT_CHANNELS     = 1
BATCH_SIZE       = 4
EPOCHS           = 50
VALIDATION_SPLIT = 0.2
LEARNING_RATE    = 1e-4

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

# --- Losses and Metrics ---
def dice_loss(y_true, y_pred, smooth=1.0):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))

def bce_dice_loss(y_true, y_pred):
    bce = BinaryCrossentropy()(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return bce + dl

# --- Build U-Net with Augmentation and BatchNorm ---
def unet_model(input_size, num_classes=OUT_CHANNELS):
    inputs = Input(shape=input_size)
    # Data augmentation
    x = RandomFlip(mode='horizontal_and_vertical')(inputs)
    x = RandomRotation(factor=0.1)(x)

    def conv_block(x, filters, dropout_rate):
        x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = Dropout(dropout_rate)(x)
        x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        return x

    # Encoder
    c1 = conv_block(x,  16, 0.1); p1 = MaxPooling2D((2,2))(c1)
    c2 = conv_block(p1, 32, 0.1); p2 = MaxPooling2D((2,2))(c2)
    c3 = conv_block(p2, 64, 0.2); p3 = MaxPooling2D((2,2))(c3)
    c4 = conv_block(p3,128, 0.2); p4 = MaxPooling2D((2,2))(c4)
    bn = conv_block(p4,256, 0.3)

    # Decoder
    def up_block(x, skip, filters, dropout_rate):
        x = UpSampling2D((2,2))(x)
        x = concatenate([x, skip])
        return conv_block(x, filters, dropout_rate)

    u4 = up_block(bn, c4, 128, 0.2)
    u3 = up_block(u4, c3, 64,  0.2)
    u2 = up_block(u3, c2, 32,  0.1)
    u1 = up_block(u2, c1, 16,  0.1)

    outputs = Conv2D(num_classes, 1, activation='sigmoid')(u1)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=['accuracy', MeanIoU(num_classes=2)]
    )
    return model

# --- Main ---
def main():
    print("Loading data...")
    X_list, y_list = [], []
    for inp_fp, tgt_fp in zip(INPUT_TIF_FILES, TARGET_TIF_FILES):
        Xp = load_and_preprocess_tif(inp_fp, INPUT_BANDS, IMG_HEIGHT, IMG_WIDTH)
        yp = load_and_preprocess_tif(tgt_fp, TARGET_BAND, IMG_HEIGHT, IMG_WIDTH)
        if Xp is None or yp is None:
            print(f"Error loading sample {inp_fp} or {tgt_fp}")
            return
        X_list.append(np.expand_dims(Xp, 0))
        y_list.append(np.expand_dims(yp, 0))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    print(f"Dataset shape X: {X.shape}, y: {y.shape}")

    # Show class imbalance
    pos_frac = np.mean(y)
    print(f"Positive pixel fraction in labels: {pos_frac:.4f}")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42
    )
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # Callbacks
    callbacks = [
        ModelCheckpoint('best_unet.h5', monitor='val_mean_io_u', mode='max', save_best_only=True),
        EarlyStopping(monitor='val_mean_io_u', mode='max', patience=10, restore_best_weights=True)
    ]

    # Build and train
    in_ch = X.shape[-1]
    model = unet_model((IMG_HEIGHT, IMG_WIDTH, in_ch))
    model.summary()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=min(BATCH_SIZE, len(X_train)),
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Plot training
    plt.figure(figsize=(18,4))
    plt.subplot(1,3,1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss'); plt.legend()
    plt.subplot(1,3,2)
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title('Accuracy'); plt.legend()
    plt.subplot(1,3,3)
    plt.plot(history.history['mean_io_u'], label='train IoU')
    plt.plot(history.history['val_mean_io_u'], label='val IoU')
    plt.title('MeanIoU'); plt.legend()
    plt.tight_layout(); plt.show()

if __name__ == '__main__':
    if not os.path.exists('data'):
        print("Ensure 'data' folder exists with your TIFFs.")
    main()
