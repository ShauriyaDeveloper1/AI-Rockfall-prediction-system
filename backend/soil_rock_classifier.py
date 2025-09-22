"""
Soil and Rock Classification System
Uses deep learning to classify soil/rock types from uploaded images
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
from pathlib import Path
import json
from PIL import Image
import base64
import io

class SoilRockClassifier:
    def __init__(self, model_dir="ml_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Soil/Rock classes based on provided dataset
        self.classes = [
            'Alluvial Soil',
            'Black Soil', 
            'Cinder Soil',
            'Red Soil'
        ]
        
        # Classification details for each soil/rock type
        self.classification_info = {
            'Alluvial Soil': {
                'description': 'Fine-grained sedimentary soil formed by water deposition',
                'characteristics': [
                    'High water retention capacity',
                    'Rich in minerals and organic matter',
                    'Good for agriculture but moderate stability'
                ],
                'mining_implications': [
                    'Moderate slope stability risk',
                    'High water table concerns',
                    'May require drainage systems'
                ],
                'risk_level': 'MEDIUM',
                'stability_score': 6.5
            },
            'Black Soil': {
                'description': 'Clay-rich soil with high shrink-swell potential',
                'characteristics': [
                    'High clay content (montmorillonite)',
                    'Expands when wet, contracts when dry',
                    'Rich in iron and aluminum oxides'
                ],
                'mining_implications': [
                    'High slope instability risk during wet seasons',
                    'Requires careful moisture management',
                    'Prone to landslides and rockfalls'
                ],
                'risk_level': 'HIGH',
                'stability_score': 4.2
            },
            'Cinder Soil': {
                'description': 'Volcanic origin soil with porous structure',
                'characteristics': [
                    'Highly porous and lightweight',
                    'Good drainage properties',
                    'Contains volcanic minerals'
                ],
                'mining_implications': [
                    'Generally stable for mining operations',
                    'Low water retention reduces slide risk',
                    'May have hidden cavities'
                ],
                'risk_level': 'LOW',
                'stability_score': 8.1
            },
            'Red Soil': {
                'description': 'Iron oxide rich soil with good drainage',
                'characteristics': [
                    'High iron oxide content',
                    'Well-drained structure',
                    'Low organic matter content'
                ],
                'mining_implications': [
                    'Moderate to good stability',
                    'Less prone to water-related failures',
                    'May have hardpan layers'
                ],
                'risk_level': 'MEDIUM',
                'stability_score': 7.3
            }
        }
        
        self.model = None
        self.model_path = self.model_dir / "soil_rock_classifier.h5"
        self.img_height = 224
        self.img_width = 224
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the classifier"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def create_model(self):
        """Create CNN model architecture for soil/rock classification"""
        model = Sequential([
            # Rescaling layer
            layers.Rescaling(1./255),
            
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Fifth Convolutional Block
            Conv2D(512, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Flatten and Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(len(self.classes), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, dataset_path):
        """Train the soil/rock classification model"""
        self.logger.info("Starting model training...")
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        # Training data generator
        train_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(self.img_height, self.img_width),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        # Validation data generator
        validation_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(self.img_height, self.img_width),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        # Create model
        self.model = self.create_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(str(self.model_path), save_best_only=True)
        ]
        
        # Train model
        history = self.model.fit(
            train_generator,
            epochs=50,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info("Model training completed!")
        return history
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            if self.model_path.exists():
                self.model = load_model(str(self.model_path))
                self.logger.info("Model loaded successfully")
                return True
            else:
                self.logger.warning("No pre-trained model found")
                return False
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image_data):
        """Preprocess image for prediction"""
        try:
            # If image_data is base64 encoded
            if isinstance(image_data, str):
                # Remove data URL prefix if present
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                # Decode base64
                img_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(img_bytes))
            else:
                # If it's already a PIL Image or file path
                if isinstance(image_data, str) and os.path.exists(image_data):
                    img = Image.open(image_data)
                else:
                    img = image_data
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image
            img = img.resize((self.img_width, self.img_height))
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_data):
        """Predict soil/rock type from image"""
        try:
            if self.model is None:
                if not self.load_model():
                    return {
                        'error': 'Model not available',
                        'success': False
                    }
            
            # Preprocess image
            processed_image = self.preprocess_image(image_data)
            if processed_image is None:
                return {
                    'error': 'Failed to process image',
                    'success': False
                }
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.classes[predicted_class_idx]
            
            # Get classification info
            classification_info = self.classification_info.get(predicted_class, {})
            
            # Prepare result
            result = {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': round(confidence * 100, 2),
                'classification_info': classification_info,
                'all_predictions': {
                    self.classes[i]: round(float(predictions[0][i]) * 100, 2)
                    for i in range(len(self.classes))
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def get_model_info(self):
        """Get information about the model"""
        return {
            'classes': self.classes,
            'model_exists': self.model_path.exists(),
            'input_shape': (self.img_height, self.img_width, 3),
            'classification_info': self.classification_info
        }

# Initialize global classifier instance
soil_rock_classifier = SoilRockClassifier()