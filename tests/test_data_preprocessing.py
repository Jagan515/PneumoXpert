import unittest
import numpy as np
from PIL import Image
import tempfile
import os

# Add src to path
import sys
sys.path.append('src')

from data_preprocessing import DataPreprocessor

class TestDataPreprocessing(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = DataPreprocessor(img_size=(150, 150))
        
    def test_initialization(self):
        self.assertEqual(self.preprocessor.img_size, (150, 150))
        self.assertEqual(self.preprocessor.batch_size, 32)
        self.assertEqual(self.preprocessor.class_names, ['NORMAL', 'PNEUMONIA'])
    
    def test_preprocess_single_image(self):
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.fromarray(dummy_image)
            img.save(tmp.name)
            
            # Test preprocessing
            processed = self.preprocessor.preprocess_single_image(tmp.name)
            
            # Check shape
            self.assertEqual(processed.shape, (1, 150, 150, 3))
            
            # Check normalization
            self.assertGreaterEqual(processed.min(), 0)
            self.assertLessEqual(processed.max(), 1)
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_invalid_image_path(self):
        with self.assertRaises(ValueError):
            self.preprocessor.preprocess_single_image('nonexistent.jpg')
    
    def test_validate_image_file(self):
        # Test with valid image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.fromarray(dummy_image)
            img.save(tmp.name)
            
            is_valid, message = self.preprocessor.validate_image_file(tmp.name)
            self.assertTrue(is_valid)
            
            os.unlink(tmp.name)
        
        # Test with invalid extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b'not an image')
            tmp.flush()
            
            is_valid, message = self.preprocessor.validate_image_file(tmp.name)
            self.assertFalse(is_valid)
            
            os.unlink(tmp.name)

if __name__ == '__main__':
    unittest.main()