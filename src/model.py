import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

class PneumoniaModel:
    """Deep learning model for pneumonia detection"""
    
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        
    def build_cnn_model(self):
        """Build a CNN model following best practices from the book"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with appropriate optimizer and metrics"""
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
    
    def train(self, train_generator, validation_generator, epochs=30, class_weight=None):
        """Train the model with callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=class_weight
        )
        
        return history
    
    def build_transfer_learning_model(self, base_model_name='VGG16'):
        """Build model using transfer learning (optional)"""
        if base_model_name == 'VGG16':
            base_model = tf.keras.applications.VGG16(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        elif base_model_name == 'ResNet50':
            base_model = tf.keras.applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model