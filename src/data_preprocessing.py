import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

class DataPreprocessor:
    """Handles loading and preprocessing of chest X-ray images"""
    
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['NORMAL', 'PNEUMONIA']
        
    def load_data_from_directory(self, train_dir, val_dir, test_dir):
        """
        Load data from directory structure
        Expected structure:
            train/
                NORMAL/
                PNEUMONIA/
            val/
                NORMAL/
                PNEUMONIA/
            test/
                NORMAL/
                PNEUMONIA/
        """
        # Create data generators with augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # For validation and test, only rescaling
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode='rgb'
        )
        
        validation_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode='rgb',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode='rgb',
            shuffle=False
        )
        
        return train_generator, validation_generator, test_generator
    
    def preprocess_single_image(self, image_path):
        """Preprocess a single image for prediction"""
        try:
            # Load and resize image
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img)
            
            # Normalize and expand dimensions
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
    
    def validate_image_file(self, image_path):
        """Validate if file is a proper medical image"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        file_ext = os.path.splitext(image_path)[1].lower()
        
        if file_ext not in valid_extensions:
            return False, "Invalid file format. Please upload JPG, JPEG, PNG, BMP, or TIFF."
        
        try:
            with Image.open(image_path) as img:
                # Check if it's a grayscale or RGB image
                if img.mode not in ['L', 'RGB', 'RGBA']:
                    return False, "Invalid image mode. Please upload a grayscale or RGB image."
                
                # Check minimum dimensions
                if img.size[0] < 50 or img.size[1] < 50:
                    return False, "Image dimensions too small. Minimum 50x50 pixels required."
                
                # Try to load as array
                img_array = np.array(img)
                if len(img_array.shape) not in [2, 3]:
                    return False, "Invalid image dimensions."
                
                return True, "Image is valid"
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"