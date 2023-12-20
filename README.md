# Siamese Network for Change Detection

This project rocks a Siamese neural network, built on the mighty ResNet-18, to spot changes between two images. Feed it two pics, and it'll spit out a binary mask showing you what's different.

## Setup

1.  **Clone the repo (if you haven't already):**
    ```bash
    # git clone <your-repo-link>
    # cd Siamese-ResNet18
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    Make sure you have `pip` installed. Then run:
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` should include:
    *   `torch`
    *   `torchvision`
    *   `Pillow`
    *   `numpy`
    *   `matplotlib` (for `test.py` visualization)
    *   `wandb` (for logging during training in `model.py`)
    *   `scikit-learn` (for metrics in `model.py`)

## Model Architecture (`model.py` and `test.py`)

The brains of the operation is the `SiameseChangeDetectionModel` class.

1.  **Twin Encoders (Siamese Style):**
    *   The model uses two identical encoders, one for each input image.
    *   Each encoder is based on a pre-trained ResNet-18 model, with its final fully connected and average pooling layers removed. This extracts high-level features from the images. The output of this stage is a feature map of size 512 channels (e.g., 512x8x8 for 256x256 input).

2.  **Feature Difference:**
    *   The feature maps from the two encoders are then subtracted element-wise, and the absolute difference is taken. This highlights the regions where the features differ significantly between the two images.

3.  **Decoder Magic:**
    *   This difference map is fed into a decoder.
    *   The decoder consists of a series of `ConvTranspose2d` (deconvolution) layers, each followed by a `ReLU` activation function.
    *   These layers progressively upsample the feature difference map back to the original input image resolution (e.g., 256x256), while reducing the number of channels.
    *   The sequence of transpose convolutions is:
        *   512 channels -> 256 channels (e.g., 8x8 -> 16x16)
        *   256 channels -> 128 channels (e.g., 16x16 -> 32x32)
        *   128 channels -> 64 channels (e.g., 32x32 -> 64x64)
        *   64 channels -> 32 channels (e.g., 64x64 -> 128x128)
        *   32 channels -> 16 channels (e.g., 128x128 -> 256x256)
    *   A final `ConvTranspose2d` layer with a kernel size of 1 reduces the channels to 1.

4.  **Output Mask:**
    *   A `Sigmoid` activation function is applied to the output of the decoder. This squashes the pixel values to a range between 0 and 1.
    *   During inference (in `quick_detect`) or metric calculation, these values are typically thresholded (e.g., at 0.5) to produce a binary change mask, where 1 indicates a change and 0 indicates no change.

## Training (`model.py`)

The `model.py` script handles the training of the Siamese network.

1.  **Dataset (`ChangeDetectionDataset`):**
    *   You'll need to prepare your dataset with pairs of images (`image_dir1`, `image_dir2`) and corresponding ground truth change masks (`mask_dir`).
    *   The script expects image files with the same names in these three directories.
    *   Update the paths:
        ```python
        image_dir1 = '/path/to/your/data/Train/current'
        image_dir2 = '/path/to/your/data/Train/past'
        mask_dir = '/path/to/your/data/Train/masks'
        ```
2.  **Configuration:**
    *   Adjust hyperparameters like `batch_size`, `lr` (learning rate), and `num_epochs` as needed.
    *   The script uses `wandb` for logging. Make sure to log in to wandb (`wandb login`) or comment out the wandb-related sections if you don't want to use it.
3.  **Run Training:**
    ```bash
    python model.py
    ```
    *   The script will train the model, print metrics (Loss, IoU, Dice, Precision, Recall, F1) per epoch, and log them to Weights & Biases.
    *   The trained model weights will be saved (e.g., `siamese_change_detection_model_50_stats.pth`).

## Inference / Quick Detection (`test.py`)

The `test.py` script provides a `quick_detect` function for performing change detection on a pair of images using a trained model.

1.  **Ensure `test.py` is accessible:**
    The script includes `sys.path.append` to help with imports if you run it from a different directory, but it's often easiest to run from the project root or ensure `Siamese-ResNet18` is in your `PYTHONPATH`.
2.  **How to use `quick_detect`:**
    ```python
    # Make sure test.py is in your Python path or run from its directory
    from test import quick_detect # Or from your_module import quick_detect

    # Paths to your images
    img1_path = "path/to/your/current_image.png"
    img2_path = "path/to/your/past_image.png"

    # Path to your trained model
    # Defaults to 'siamese_change_detection_model_50.pth' if not provided
    model_weights_path = "siamese_change_detection_model_50_stats.pth" # Or your trained model

    # Get the change mask (it's a NumPy array)
    # Set show_plot=True to see the images and the mask
    change_mask_array = quick_detect(
        image1_path=img1_path,
        image2_path=img2_path,
        model_path=model_weights_path,
        show_plot=True
    )

    if change_mask_array is not None:
        print(f"Boom! Change detection done. Mask shape: {change_mask_array.shape}")
    ```
3.  **`quick_detect` Parameters:**
    *   `image1_path` (str): Path to the first (e.g., current) image.
    *   `image2_path` (str): Path to the second (e.g., past) image.
    *   `model_path` (str, optional): Path to the pre-trained model weights. Defaults to `'siamese_change_detection_model_50.pth'`. **Make sure this matches the name of your saved model from training, or the default one if you're using it.**
    *   `show_plot` (bool, optional): If `True`, it'll pop up a matplotlib window showing the input images and the resulting change mask. Defaults to `False`.

4.  **Running the Example in `test.py`:**
    If you want to run the example code at the bottom of `test.py`:
    1.  Uncomment the `if __name__ == "__main__":` block.
    2.  Update `img1_path` and `img2_path` to point to actual images you have.
    3.  Ensure the `model_path` in `quick_detect` (or its default) points to your trained model.
    4.  Run from your terminal:
        ```bash
        python test.py
        ```

## Notes
*   The model expects input images to be resized to 256x256.
*   The default model name in `test.py` is `siamese_change_detection_model_50.pth`. The training script `model.py` saves as `siamese_change_detection_model_50_stats.pth`. You might need to align these names or provide the correct path when calling `quick_detect`.
