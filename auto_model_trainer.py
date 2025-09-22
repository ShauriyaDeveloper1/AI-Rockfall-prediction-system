"""
Automatic Model Training Service
Handles automatic training of LSTM and other ML models when data is uploaded
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import logging

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class AutoModelTrainer:
    def __init__(self, models_dir="ml_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model file paths
        self.classifier_path = self.models_dir / "rockfall_classifier.pkl"
        self.regressor_path = self.models_dir / "rockfall_regressor.pkl"
        self.scaler_path = self.models_dir / "scaler.pkl"
        self.lstm_model_path = self.models_dir / "lstm_model.h5"
        self.lstm_scaler_path = self.models_dir / "lstm_scaler.pkl"
        
        # Initialize models
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.lstm_model = None
        self.lstm_scaler = None
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for training process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('model_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_training_data(self, n_samples=10000):
        """Generate comprehensive training data for all models"""
        self.logger.info(f"Generating {n_samples} training samples...")
        
        # Generate synthetic sensor data
        np.random.seed(42)
        
        # Time series data
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=365),
            end=datetime.now(),
            periods=n_samples
        )
        
        # Base sensor readings with realistic patterns
        data = []
        for i, timestamp in enumerate(timestamps):
            # Seasonal patterns
            day_of_year = timestamp.timetuple().tm_yday
            seasonal_factor = np.sin(2 * np.pi * day_of_year / 365) * 0.1 + 1
            
            # Time of day patterns
            hour_factor = np.sin(2 * np.pi * timestamp.hour / 24) * 0.05 + 1
            
            # Progressive displacement (rockfall risk increases over time)
            time_factor = i / n_samples
            
            # Base readings with noise
            displacement = np.random.normal(1.0 + time_factor * 2, 0.3) * seasonal_factor
            strain = np.random.normal(100 + time_factor * 50, 15) * seasonal_factor
            pore_pressure = np.random.normal(50 + time_factor * 30, 8) * seasonal_factor
            temperature = np.random.normal(20, 5) * hour_factor
            
            # Weather-related factors
            rainfall = max(0, np.random.normal(5, 10))
            wind_speed = max(0, np.random.normal(15, 8))
            humidity = np.clip(np.random.normal(60, 20), 0, 100)
            
            # Geological factors
            slope_angle = np.random.uniform(30, 75)
            rock_quality = np.random.uniform(0.2, 0.9)
            joint_spacing = np.random.uniform(0.1, 2.0)
            
            # Calculate risk factors
            displacement_risk = min(1.0, displacement / 5.0)
            strain_risk = min(1.0, strain / 300.0)
            pressure_risk = min(1.0, pore_pressure / 150.0)
            weather_risk = min(1.0, rainfall / 50.0 + wind_speed / 100.0)
            geological_risk = (1 - rock_quality) * (slope_angle / 75.0) * (1 / joint_spacing)
            
            # Combined risk calculation
            combined_risk = (
                displacement_risk * 0.25 +
                strain_risk * 0.20 +
                pressure_risk * 0.15 +
                weather_risk * 0.15 +
                geological_risk * 0.25
            )
            
            # Add some randomness
            combined_risk = np.clip(combined_risk + np.random.normal(0, 0.1), 0, 1)
            
            # Determine risk level and rockfall occurrence
            if combined_risk < 0.25:
                risk_level = 'LOW'
                rockfall_occurred = np.random.choice([0, 1], p=[0.95, 0.05])
            elif combined_risk < 0.5:
                risk_level = 'MEDIUM'
                rockfall_occurred = np.random.choice([0, 1], p=[0.85, 0.15])
            elif combined_risk < 0.75:
                risk_level = 'HIGH'
                rockfall_occurred = np.random.choice([0, 1], p=[0.65, 0.35])
            else:
                risk_level = 'CRITICAL'
                rockfall_occurred = np.random.choice([0, 1], p=[0.4, 0.6])
            
            # LIDAR-related data
            point_cloud_density = np.random.uniform(100, 1000)
            surface_roughness = np.random.uniform(0.1, 1.0)
            displacement_vector_magnitude = np.sqrt(
                displacement**2 + np.random.normal(0, 0.5)**2 + np.random.normal(0, 0.5)**2
            )
            
            data.append({
                'timestamp': timestamp,
                'displacement': displacement,
                'strain': strain,
                'pore_pressure': pore_pressure,
                'temperature': temperature,
                'rainfall': rainfall,
                'wind_speed': wind_speed,
                'humidity': humidity,
                'slope_angle': slope_angle,
                'rock_quality': rock_quality,
                'joint_spacing': joint_spacing,
                'point_cloud_density': point_cloud_density,
                'surface_roughness': surface_roughness,
                'displacement_vector_magnitude': displacement_vector_magnitude,
                'risk_probability': combined_risk,
                'risk_level': risk_level,
                'rockfall_occurred': rockfall_occurred
            })
        
        df = pd.DataFrame(data)
        
        # Save training data
        training_data_path = self.models_dir / "comprehensive_training_data.csv"
        df.to_csv(training_data_path, index=False)
        
        self.logger.info(f"Training data saved to {training_data_path}")
        self.logger.info(f"Data shape: {df.shape}")
        self.logger.info(f"Risk level distribution: {df['risk_level'].value_counts().to_dict()}")
        self.logger.info(f"Rockfall occurrence rate: {df['rockfall_occurred'].mean():.2%}")
        
        return df
    
    def train_traditional_models(self, data=None):
        """Train Random Forest and Gradient Boosting models"""
        self.logger.info("Training traditional ML models...")
        
        if data is None:
            data = self.generate_comprehensive_training_data()
        
        # Prepare features and targets
        feature_columns = [
            'displacement', 'strain', 'pore_pressure', 'temperature',
            'rainfall', 'wind_speed', 'humidity', 'slope_angle',
            'rock_quality', 'joint_spacing', 'point_cloud_density',
            'surface_roughness', 'displacement_vector_magnitude'
        ]
        
        X = data[feature_columns]
        y_classification = data['rockfall_occurred']
        y_regression = data['risk_probability']
        
        # Split data
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_classification, y_regression, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest Classifier
        self.logger.info("Training Random Forest Classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.classifier.fit(X_train_scaled, y_class_train)
        
        # Train Gradient Boosting Regressor
        self.logger.info("Training Gradient Boosting Regressor...")
        self.regressor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        self.regressor.fit(X_train_scaled, y_reg_train)
        
        # Evaluate models
        class_predictions = self.classifier.predict(X_test_scaled)
        reg_predictions = self.regressor.predict(X_test_scaled)
        
        class_accuracy = accuracy_score(y_class_test, class_predictions)
        reg_mse = mean_squared_error(y_reg_test, reg_predictions)
        
        self.logger.info(f"Classifier Accuracy: {class_accuracy:.3f}")
        self.logger.info(f"Regressor MSE: {reg_mse:.3f}")
        
        # Save models
        joblib.dump(self.classifier, self.classifier_path)
        joblib.dump(self.regressor, self.regressor_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        self.logger.info("Traditional models saved successfully")
        
        return {
            'classifier_accuracy': class_accuracy,
            'regressor_mse': reg_mse,
            'features_used': feature_columns
        }
    
    def train_lstm_model(self, data=None, sequence_length=24):
        """Train LSTM model for time series prediction"""
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow not available for LSTM training")
            return None
        
        self.logger.info("Training LSTM model...")
        
        if data is None:
            data = self.generate_comprehensive_training_data()
        
        # Prepare time series data
        feature_columns = [
            'displacement', 'strain', 'pore_pressure', 'temperature',
            'rainfall', 'wind_speed', 'humidity'
        ]
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        features = data[feature_columns].values
        targets = data['risk_probability'].values
        
        # Scale features
        self.lstm_scaler = StandardScaler()
        features_scaled = self.lstm_scaler.fit_transform(features)
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(features_scaled)):
            X_sequences.append(features_scaled[i-sequence_length:i])
            y_sequences.append(targets[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Split data
        split_idx = int(0.8 * len(X_sequences))
        X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
        
        # Build LSTM model
        self.lstm_model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, len(feature_columns))),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.lstm_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate model
        test_loss = self.lstm_model.evaluate(X_test, y_test, verbose=0)
        
        self.logger.info(f"LSTM Test Loss: {test_loss}")
        
        # Save model
        self.lstm_model.save(str(self.lstm_model_path))
        joblib.dump(self.lstm_scaler, self.lstm_scaler_path)
        
        self.logger.info("LSTM model saved successfully")
        
        return {
            'test_loss': test_loss,
            'sequence_length': sequence_length,
            'features_used': feature_columns,
            'training_history': history.history
        }
    
    def auto_train_on_upload(self, uploaded_data_path):
        """Automatically train models when new data is uploaded"""
        self.logger.info(f"Auto-training models with uploaded data: {uploaded_data_path}")
        
        try:
            # Load uploaded data
            if uploaded_data_path.endswith('.csv'):
                uploaded_data = pd.read_csv(uploaded_data_path)
            else:
                self.logger.error(f"Unsupported file format: {uploaded_data_path}")
                return None
            
            # Validate data has required columns
            required_columns = ['displacement', 'strain', 'pore_pressure', 'temperature']
            missing_columns = [col for col in required_columns if col not in uploaded_data.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing columns: {missing_columns}. Using generated data...")
                training_data = self.generate_comprehensive_training_data()
            else:
                # Augment uploaded data with generated data
                generated_data = self.generate_comprehensive_training_data(n_samples=5000)
                training_data = pd.concat([uploaded_data, generated_data], ignore_index=True)
                self.logger.info(f"Combined data shape: {training_data.shape}")
            
            # Train all models
            traditional_results = self.train_traditional_models(training_data)
            lstm_results = self.train_lstm_model(training_data)
            
            training_summary = {
                'timestamp': datetime.now().isoformat(),
                'data_samples': len(training_data),
                'traditional_models': traditional_results,
                'lstm_model': lstm_results,
                'models_saved': {
                    'classifier': str(self.classifier_path),
                    'regressor': str(self.regressor_path),
                    'scaler': str(self.scaler_path),
                    'lstm_model': str(self.lstm_model_path),
                    'lstm_scaler': str(self.lstm_scaler_path)
                }
            }
            
            # Save training summary
            summary_path = self.models_dir / "training_summary.json"
            import json
            with open(summary_path, 'w') as f:
                json.dump(training_summary, f, indent=2)
            
            self.logger.info("Auto-training completed successfully")
            return training_summary
            
        except Exception as e:
            self.logger.error(f"Auto-training failed: {str(e)}")
            return None
    
    def load_trained_models(self):
        """Load pre-trained models"""
        try:
            if self.classifier_path.exists():
                self.classifier = joblib.load(self.classifier_path)
                self.logger.info("Classifier loaded")
            
            if self.regressor_path.exists():
                self.regressor = joblib.load(self.regressor_path)
                self.logger.info("Regressor loaded")
            
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                self.logger.info("Scaler loaded")
            
            if TENSORFLOW_AVAILABLE and self.lstm_model_path.exists():
                self.lstm_model = keras.models.load_model(str(self.lstm_model_path))
                self.logger.info("LSTM model loaded")
            
            if self.lstm_scaler_path.exists():
                self.lstm_scaler = joblib.load(self.lstm_scaler_path)
                self.logger.info("LSTM scaler loaded")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            return False
    
    def get_model_status(self):
        """Get current status of all models"""
        status = {
            'traditional_models': {
                'classifier_available': self.classifier is not None,
                'regressor_available': self.regressor is not None,
                'scaler_available': self.scaler is not None,
                'classifier_file_exists': self.classifier_path.exists(),
                'regressor_file_exists': self.regressor_path.exists(),
                'scaler_file_exists': self.scaler_path.exists()
            },
            'lstm_model': {
                'available': TENSORFLOW_AVAILABLE,
                'model_loaded': self.lstm_model is not None,
                'scaler_loaded': self.lstm_scaler is not None,
                'model_file_exists': self.lstm_model_path.exists(),
                'scaler_file_exists': self.lstm_scaler_path.exists()
            }
        }
        
        return status

if __name__ == "__main__":
    # Initialize trainer and generate comprehensive training data
    trainer = AutoModelTrainer()
    
    # Generate and train with comprehensive data
    print("Generating comprehensive training data...")
    training_data = trainer.generate_comprehensive_training_data(n_samples=15000)
    
    print("Training traditional models...")
    traditional_results = trainer.train_traditional_models(training_data)
    
    print("Training LSTM model...")
    lstm_results = trainer.train_lstm_model(training_data)
    
    print("Training completed successfully!")
    print(f"Traditional models results: {traditional_results}")
    print(f"LSTM model results: {lstm_results}")