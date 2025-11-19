# Pneumonia Detection Model Training Script

## Overview

This repository contains a complete Kaggle-compatible training script for building and evaluating a Convolutional Neural Network (CNN) model to detect pneumonia from chest X-ray images. The script is designed to run in a Kaggle Notebook environment with GPU acceleration enabled.

The dataset used is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle, which includes labeled images categorized as `NORMAL` or `PNEUMONIA`.

[Kaggle-NoteBook](https://www.kaggle.com/code/jaganpradhan0/pneumoxpert)

### Key Features
- **Data Augmentation**: Applies transformations like rotation, shifting, shearing, zooming, flipping, and brightness adjustments to improve model robustness.
- **Advanced CNN Architecture**: A deep sequential model with convolutional blocks, batch normalization, max pooling, and dropout for regularization.
- **Class Imbalance Handling**: Computes balanced class weights to address the dataset's imbalance.
- **Comprehensive Training**: Includes early stopping, model checkpointing, learning rate reduction, and TensorBoard logging.
- **Evaluation Metrics**: Reports accuracy, precision, recall, AUC, classification report, and confusion matrix.
- **Production-Ready Outputs**: Saves the model in multiple formats (H5, pickle pipeline, JSON architecture) along with visualizations.
- **Reproducibility**: Sets random seeds and uses fixed hyperparameters.

## Prerequisites

- **Environment**: Kaggle Notebook with GPU accelerator (e.g., Tesla P100).
- **Dependencies**:
  - Python 3.8+
  - TensorFlow 2.x (with Keras)
  - NumPy, Pandas
  - Matplotlib, Seaborn
  - Scikit-learn
  - Joblib

Install via pip (in Kaggle, most are pre-installed):
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn joblib
```

## Dataset

The script assumes the dataset is attached as a Kaggle input dataset:
- Path: `/kaggle/input/chest-xray-pneumonia/chest_xray`
- Splits: `train/`, `val/` (optional; falls back to train split), `test/`
- Classes: `NORMAL/` and `PNEUMONIA/`

### Dataset Structure Analysis
Run the script to see image counts:
```
train/NORMAL: 1341 images
train/PNEUMONIA: 3875 images
val/NORMAL: 16 images
val/PNEUMONIA: 8 images
test/NORMAL: 234 images
test/PNEUMONIA: 390 images
```

**Note**: The validation set is small; the script handles this by optionally splitting from training data.

## Usage

1. **Setup in Kaggle**:
   - Create a new Kaggle Notebook.
   - Add the dataset: Search for "chest-xray-pneumonia" and add as input.
   - Enable GPU accelerator in Notebook settings.

2. **Run the Script**:
   - Copy the full script into a cell and execute.
   - Or, run modularly by calling functions (e.g., `load_and_analyze_data()`).

   ```python
   if __name__ == "__main__":
       main()
   ```

3. **Expected Output**:
   - Console logs: Data analysis, training progress, evaluation metrics.
   - Files in `/kaggle/working/models/`:
     - `best_model_kaggle.h5`: Best checkpoint.
     - `pneumonia_model_final.h5`: Final trained model.
     - `pneumonia_pipeline.pkl`: Pickle with model and metadata.
     - `model_architecture.json`: JSON schema.
     - `training_history.png`: Training/validation plots.
     - `confusion_matrix.png`: Test set confusion matrix.
   - TensorBoard logs: `/kaggle/working/models/logs/` (view in Kaggle or locally).

## Script Breakdown

The script is modular for easy customization:

| Function | Description |
|----------|-------------|
| `load_and_analyze_data()` | Counts and prints dataset images per split/class. |
| `create_data_generators()` | Sets up ImageDataGenerators for train/val/test with augmentation. |
| `build_advanced_model()` | Defines the CNN architecture (4 conv blocks + dense layers). |
| `train_model()` | Loads data, computes class weights, compiles, and trains with callbacks. |
| `evaluate_model(model, test_gen)` | Predicts on test set, prints metrics, plots confusion matrix. |
| `plot_training_history(history)` | Visualizes loss, accuracy, precision, recall, AUC over epochs. |
| `save_model_for_production(model, test_gen)` | Saves model artifacts for deployment. |
| `create_submission_file(model, test_gen)` | Generates sample CSV for Kaggle submission verification. |
| `main()` | Orchestrates the full pipeline. |

### Model Architecture Summary
- Input: 224x224x3 RGB images.
- Conv Blocks: 64→128→256→512 filters (2 convs each, ReLU, BatchNorm, MaxPool2D, Dropout 0.25).
- Dense: 512→256 units (ReLU, BatchNorm, Dropout 0.5).
- Output: Sigmoid for binary classification.
- Optimizer: Adam (lr=0.0001).
- Loss: Binary Crossentropy.
- Metrics: Accuracy, Precision, Recall, AUC.

Total params: ~25M (trainable).

### Training Hyperparameters
- Batch Size: 32
- Epochs: 50 (early stopping at 15 patience on val_loss)
- Augmentation: Rotation ±20°, shifts 20%, shear/zoom 15%, flip, brightness ±20%.
- Class Weights: Balanced (e.g., {0: 1.28, 1: 0.44}).

## Expected Performance
On the test set (624 images):
- Accuracy: ~90-95%
- Precision (Pneumonia): ~95%
- Recall (Pneumonia): ~92%
- AUC: ~0.97

Results vary with random seeds and GPU utilization. Monitor via TensorBoard.

## Customization

- **Model Tweaks**: Modify `build_advanced_model()` (e.g., add ResNet base: `tf.keras.applications.ResNet50`).
- **Hyperparameters**: Adjust batch size, learning rate, or augmentation in `create_data_generators()`.
- **Validation Split**: If no `val/` dir, enable proper split by adding `validation_split=0.2` to `train_datagen`.
- **Deployment**: Load pickle via `joblib.load()` for inference:
  ```python
  pipeline = joblib.load('pneumonia_pipeline.pkl')
  pred = pipeline['model'].predict(img_resized) > pipeline['threshold']
  ```

## Limitations
- Relies on Kaggle paths; adapt for local runs (e.g., change `BASE_PATH`).
- Validation split fallback is simplified—use `tf.keras.utils.image_dataset_from_directory` for better control.
- No transfer learning (add pre-trained backbone for SOTA).
- Dataset bias: Real-world X-rays may differ.

## License
MIT License. Use responsibly for medical applications—consult experts for clinical use.

## Contact
For issues, open a GitHub issue or reach out via Kaggle discussions.

---

*Last Updated: December 2025*