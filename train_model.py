import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_preprocessing import DataPreprocessor
from model import PneumoniaModel
from utils import save_model_pipeline, plot_training_history, calculate_class_weights
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

import matplotlib.pyplot as plt

def main():
    # Paths to data (modify these based on your directory structure)
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'  # You'll need to download the dataset here
    
    train_dir = DATA_DIR / 'chest_xray' / 'train'
    val_dir = DATA_DIR / 'chest_xray' / 'val'
    test_dir = DATA_DIR / 'chest_xray' / 'test'
    
    # Create models directory if it doesn't exist
    models_dir = BASE_DIR / 'models'
    models_dir.mkdir(exist_ok=True)
    
    print("Initializing data preprocessor...")
    preprocessor = DataPreprocessor(img_size=(224, 224), batch_size=32)
    
    print("Loading data...")
    try:
        train_generator, val_generator, test_generator = preprocessor.load_data_from_directory(
            train_dir, val_dir, test_dir
        )
        print(f"Classes: {train_generator.class_indices}")
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Test samples: {test_generator.samples}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure the dataset is organized correctly.")
        print("Expected structure: data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/")
        return
    
    # Calculate class weights for imbalanced dataset
    class_weights = calculate_class_weights(train_generator)
    print(f"Class weights: {class_weights}")
    
    print("\nBuilding model...")
    pneumonia_model = PneumoniaModel(input_shape=(224, 224, 3))
    model = pneumonia_model.build_cnn_model()
    pneumonia_model.model = model
    
    print("Model summary:")
    pneumonia_model.model.summary()
    
    print("\nCompiling model...")
    pneumonia_model.compile_model(learning_rate=0.001)
    
    print("\nTraining model...")
    early_stop = EarlyStopping(
    monitor='val_loss',      # Monitor validation loss
    patience=5,              # Stop after 5 epochs of no improvement
    restore_best_weights=True,  # Restore model weights from the epoch with the best val_loss
    verbose=1
   )
    checkpoint = ModelCheckpoint(
    models_dir / 'best_model.h5', 
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

    history = pneumonia_model.train(
        train_generator,
        val_generator,
        epochs=30,
        class_weight=class_weights,
        callbacks=[early_stop,checkpoint]
        
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = pneumonia_model.model.evaluate(test_generator)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test Precision: {test_results[2]:.4f}")
    print(f"Test Recall: {test_results[3]:.4f}")
    print(f"Test AUC: {test_results[4]:.4f}")
    
    # Save the trained model
    print("\nSaving model...")
    pneumonia_model.model.save(models_dir / 'pneumonia_model.h5')
    
    # Save pipeline with joblib
    save_model_pipeline(pneumonia_model.model, preprocessor, 
                       models_dir / 'pneumonia_model.pkl')
    
    # Plot training history
    print("\nGenerating training plots...")
    fig = plot_training_history(history)
    plt.savefig(models_dir / 'training_history.png', dpi=100)
    plt.show()
    
    print("\nTraining completed successfully!")
    print(f"Models saved in: {models_dir}")

if __name__ == "__main__":
    main()