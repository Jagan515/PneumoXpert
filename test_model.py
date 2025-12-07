import unittest
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_preprocessing import DataPreprocessor
from model import PneumoniaModel

class TestPneumoniaDetection(unittest.TestCase):
    """Test cases for pneumonia detection system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = DataPreprocessor(img_size=(224, 224))
        self.model_builder = PneumoniaModel(input_shape=(224, 224, 3))
        
        # Create a dummy image for testing
        self.dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    def test_preprocessor_initialization(self):
        """Test data preprocessor initialization"""
        self.assertEqual(self.preprocessor.img_size, (224, 224))
        self.assertEqual(self.preprocessor.class_names, ['NORMAL', 'PNEUMONIA'])
        self.assertEqual(self.preprocessor.batch_size, 32)
    
    def test_model_building(self):
        """Test if model builds correctly"""
        model = self.model_builder.build_cnn_model()
        self.assertIsNotNone(model)
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 1))
        
        # Check number of layers
        self.assertGreater(len(model.layers), 10)
    
    def test_image_preprocessing(self):
        """Test single image preprocessing"""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.fromarray(self.dummy_image)
            img.save(tmp.name)
            
            # Test preprocessing
            processed = self.preprocessor.preprocess_single_image(tmp.name)
            
            # Check shape
            self.assertEqual(processed.shape, (1, 224, 224, 3))
            
            # Check normalization
            self.assertGreaterEqual(processed.min(), 0.0)
            self.assertLessEqual(processed.max(), 1.0)
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_image_validation(self):
        """Test image validation function"""
        # Create test images
        valid_image = Image.fromarray(self.dummy_image)
        
        # Test with valid image
        is_valid, message = self.preprocessor.validate_image_file('test.jpg')
        # Since file doesn't exist, should return False
        self.assertFalse(is_valid)
    
    def test_model_compilation(self):
        """Test model compilation"""
        model = self.model_builder.build_cnn_model()
        self.model_builder.model = model
        self.model_builder.compile_model()
        
        # Check if model is compiled
        self.assertTrue(hasattr(self.model_builder.model, 'optimizer'))
        self.assertTrue(hasattr(self.model_builder.model, 'loss'))
    
    def test_class_weight_calculation(self):
        """Test class weight calculation"""
        # Mock class distribution
        class_counts = np.array([0, 0, 1, 1, 1])  # 2 of class 0, 3 of class 1
        
        from utils import calculate_class_weights
        
        # Create a mock generator
        class MockGenerator:
            def __init__(self, classes):
                self.classes = classes
        
        mock_generator = MockGenerator(class_counts)
        weights = calculate_class_weights(mock_generator)
        
        # Check weights are calculated
        self.assertIn(0, weights)
        self.assertIn(1, weights)
        self.assertGreater(weights[0], 0)
        self.assertGreater(weights[1], 0)
    
    def test_transfer_learning_model(self):
        """Test transfer learning model building"""
        model = self.model_builder.build_transfer_learning_model('VGG16')
        self.assertIsNotNone(model)
        self.assertEqual(model.output_shape, (None, 1))
    
    def test_invalid_image_handling(self):
        """Test handling of invalid images"""
        # Test with non-existent file
        is_valid, message = self.preprocessor.validate_image_file('nonexistent.jpg')
        self.assertFalse(is_valid)
        
        # Test with invalid extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b'Not an image')
            tmp.flush()
            is_valid, message = self.preprocessor.validate_image_file(tmp.name)
            self.assertFalse(is_valid)
            os.unlink(tmp.name)
    
    def test_pipeline_saving(self):
        """Test model pipeline saving and loading"""
        import joblib
        
        # Create a simple pipeline
        pipeline = {
            'model': self.model_builder.build_cnn_model(),
            'img_size': (224, 224),
            'class_names': ['NORMAL', 'PNEUMONIA'],
            'version': '1.0.0'
        }
        
        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            joblib.dump(pipeline, tmp.name)
            
            # Load back
            loaded = joblib.load(tmp.name)
            
            self.assertEqual(loaded['img_size'], pipeline['img_size'])
            self.assertEqual(loaded['class_names'], pipeline['class_names'])
            self.assertEqual(loaded['version'], pipeline['version'])
            
            os.unlink(tmp.name)

if __name__ == '__main__':
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPneumoniaDetection)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print('='*50)