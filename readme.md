# Spatio-Temporal Wildfire Spread Prediction

This project implements a Convolutional LSTM (ConvLSTM) model in PyTorch to predict wildfire spread over time using multi-band GeoTIFF inputs. Each band of the input GeoTIFF represents a timestamped snapshot of fire masks or intensities. The model learns temporal dynamics and spatial features to forecast the next time step's burn area.

## Features

* **Multi-Band GeoTIFF Input**: Reads a single GeoTIFF (`land2.tif`) with multiple bands as time-series frames.
* **ConvLSTM Architecture**: A custom convolutional LSTM cell preserves spatial dimensions and captures temporal dependencies.
* **MPS/CUDA/CPU Support**: Automatically selects Apple MPS on macOS, NVIDIA CUDA if available, or CPU.
* **Train/Validation/Test Split**: Splits sequences 70% train, 15% validation, 15% test to avoid data leakage over time.
* **Evaluation Metric**: Reports Intersection-over-Union (IoU) on held-out test sequences.

## Repository Structure

```
project-root/
├── land2.tif            # Multi-band GeoTIFF (dynamic wildfire masks)
├── static.tif           # (Optional) Static condition layer (e.g., elevation)
├── preprocess.py        # Main script: dataset, model, training loop
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Dependencies

Create a virtual environment and install:

```bash
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
# or venv\Scripts\activate  # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` should include:

```
torch
rasterio
numpy
```

(Additional packages may include `torchvision`, `matplotlib`, etc.)

## Usage

1. **Place Data**
   Save your multi-band `land2.tif` in the project root. Optionally add a static layer `static.tif`.

2. **Configure Parameters**
   In `preprocess.py`, adjust:

   ```python
   DYNAMIC_FILE = 'land2.tif'
   STATIC_LAYER = None  # or 'static.tif'
   SEQ_LEN      = 5
   BATCH_SIZE   = 4
   EPOCHS       = 10
   LR           = 1e-3
   ```

3. **Run the Script**

   ```bash
   python preprocess.py
   ```

   The script will:

   * Detect device (MPS/CUDA/CPU)
   * Load frames and optional static layer
   * Train the ConvLSTM model
   * Print training & validation losses each epoch
   * Compute and print test IoU at the end

## Model Details

* **ConvLSTMCell**: Combines current input and hidden state via convolutional gates (input, forget, output, cell).
* **ConvLSTMNet**: Stacks `T` time steps, updates hidden states, then applies a final convolution to predict the next mask.

## Extending the Project

* **3D-CNN or Transformers**: Swap out `ConvLSTMCell` with a 3D convolutional network or a temporal transformer encoder for potentially improved performance.
* **Additional Inputs**: Incorporate weather data or fuel moisture as extra channels in the dataset class.
* **Hyperparameter Tuning**: Experiment with different `hidden_ch`, `kernel` sizes, learning rates, and sequence lengths.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or performance improvements.

---

*Created by the Spatio-Temporal Wildfire Prediction Team*
