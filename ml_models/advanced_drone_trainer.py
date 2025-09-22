#!/usr/bin/env python3
"""
Advanced AI Model Trainer for Drone Image Analysis
Trains on real geological features for rockfall prediction
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.segmentation import slic
from skimage.color import label2rgb
import warnings
warnings.filterwarnings('ignore')

class DroneImageAnalyzer:
    """Advanced analyzer for geological drone imagery"""
    
    def __init__(self):
        self.feature_extractors = {
            'texture': self._extract_texture_features,
            'color': self._extract_color_features,
            'edge': self._extract_edge_features,
            'geological': self._extract_geological_features,
            'vegetation': self._extract_vegetation_features,
            'slope': self._extract_slope_features
        }
        
    def analyze_image(self, image_path):
        """Comprehensive analysis of drone image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}
            
            # Extract all features
            features = {}
            for feature_type, extractor in self.feature_extractors.items():
                features.update(extractor(image, image_path))
            
            # Generate risk assessment
            risk_assessment = self._assess_risk(features)
            
            return {
                'features': features,
                'risk_assessment': risk_assessment,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _extract_texture_features(self, image, image_path):
        """Extract texture-based features for rock surface analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern for texture analysis
        lbp = local_binary_pattern(gray, 24, 8, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-6)
        
        # Gray Level Co-occurrence Matrix for texture properties
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
        
        glcm_features = {}
        for d in distances:
            for a in angles:
                glcm = graycomatrix(gray, [d], [a], levels=256, symmetric=True, normed=True)
                for prop in properties:
                    value = graycoprops(glcm, prop)[0, 0]
                    glcm_features[f'glcm_{prop}_d{d}_a{int(a*180/np.pi)}'] = value
        
        # Roughness indicators
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return {
            'texture_roughness': float(laplacian_var),
            'texture_uniformity': float(np.std(lbp_hist)),
            'texture_contrast': float(glcm_features.get('glcm_contrast_d1_a0', 0)),
            'surface_complexity': float(np.mean(list(glcm_features.values())))
        }
    
    def _extract_color_features(self, image, image_path):
        """Extract color-based geological features"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Rock color detection (brownish/grayish)
        lower_rock = np.array([10, 30, 30])
        upper_rock = np.array([30, 255, 200])
        rock_mask = cv2.inRange(hsv, lower_rock, upper_rock)
        rock_percentage = (np.sum(rock_mask > 0) / rock_mask.size) * 100
        
        # Weathering detection (oxidation - reddish colors)
        lower_oxidation = np.array([0, 50, 50])
        upper_oxidation = np.array([10, 255, 255])
        oxidation_mask = cv2.inRange(hsv, lower_oxidation, upper_oxidation)
        oxidation_percentage = (np.sum(oxidation_mask > 0) / oxidation_mask.size) * 100
        
        # Water/moisture detection
        lower_water = np.array([100, 50, 50])
        upper_water = np.array([130, 255, 255])
        water_mask = cv2.inRange(hsv, lower_water, upper_water)
        moisture_percentage = (np.sum(water_mask > 0) / water_mask.size) * 100
        
        # Color diversity (geological complexity)
        colors_bgr = image.reshape(-1, 3)
        color_variance = np.var(colors_bgr, axis=0).mean()
        
        return {
            'rock_exposure_percent': float(rock_percentage),
            'weathering_indicators': float(oxidation_percentage),
            'moisture_content': float(moisture_percentage),
            'geological_diversity': float(color_variance)
        }
    
    def _extract_edge_features(self, image, image_path):
        """Extract crack and fracture features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale edge detection for cracks
        edges_canny = cv2.Canny(gray, 50, 150)
        edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        edges_sobel = np.uint8(np.absolute(edges_sobel))
        
        # Find contours (potential cracks)
        contours, _ = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze crack characteristics
        crack_count = len(contours)
        crack_lengths = []
        crack_orientations = []
        
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # Filter small noise
                # Calculate crack length
                length = cv2.arcLength(contour, False)
                crack_lengths.append(length)
                
                # Calculate orientation
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    crack_orientations.append(ellipse[2])
        
        # Fracture density
        total_crack_length = sum(crack_lengths) if crack_lengths else 0
        image_area = image.shape[0] * image.shape[1]
        fracture_density = total_crack_length / image_area if image_area > 0 else 0
        
        # Crack network complexity
        orientation_variance = np.var(crack_orientations) if crack_orientations else 0
        
        return {
            'crack_count': int(crack_count),
            'fracture_density': float(fracture_density),
            'avg_crack_length': float(np.mean(crack_lengths)) if crack_lengths else 0,
            'crack_network_complexity': float(orientation_variance)
        }
    
    def _extract_geological_features(self, image, image_path):
        """Extract specific geological stability features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Bedding plane detection (horizontal structures)
        kernel_horizontal = np.ones((1, 15), np.uint8)
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_horizontal)
        horizontal_score = np.sum(horizontal_lines > 128) / horizontal_lines.size
        
        # Joint detection (vertical structures)
        kernel_vertical = np.ones((15, 1), np.uint8)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_vertical)
        vertical_score = np.sum(vertical_lines > 128) / vertical_lines.size
        
        # Block size estimation using superpixel segmentation
        segments = slic(image, n_segments=100, compactness=10)
        block_count = len(np.unique(segments))
        avg_block_size = image.shape[0] * image.shape[1] / block_count if block_count > 0 else 0
        
        # Surface irregularity
        gradient_magnitude = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        )
        surface_roughness = np.std(gradient_magnitude)
        
        return {
            'bedding_plane_strength': float(horizontal_score),
            'joint_density': float(vertical_score),
            'average_block_size': float(avg_block_size),
            'surface_irregularity': float(surface_roughness)
        }
    
    def _extract_vegetation_features(self, image, image_path):
        """Extract vegetation coverage for stability assessment"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Green vegetation detection
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Healthy vegetation (bright green)
        lower_healthy = np.array([35, 100, 100])
        upper_healthy = np.array([85, 255, 255])
        healthy_mask = cv2.inRange(hsv, lower_healthy, upper_healthy)
        
        # Dead/sparse vegetation (brownish)
        lower_sparse = np.array([10, 20, 20])
        upper_sparse = np.array([25, 100, 100])
        sparse_mask = cv2.inRange(hsv, lower_sparse, upper_sparse)
        
        total_pixels = image.shape[0] * image.shape[1]
        vegetation_coverage = (np.sum(green_mask > 0) / total_pixels) * 100
        healthy_vegetation = (np.sum(healthy_mask > 0) / total_pixels) * 100
        sparse_vegetation = (np.sum(sparse_mask > 0) / total_pixels) * 100
        
        # Vegetation distribution uniformity
        if np.sum(green_mask) > 0:
            vegetation_coords = np.where(green_mask > 0)
            vegetation_distribution = np.std(vegetation_coords[0]) + np.std(vegetation_coords[1])
        else:
            vegetation_distribution = 0
        
        return {
            'vegetation_coverage_percent': float(vegetation_coverage),
            'healthy_vegetation_percent': float(healthy_vegetation),
            'sparse_vegetation_percent': float(sparse_vegetation),
            'vegetation_distribution_score': float(vegetation_distribution)
        }
    
    def _extract_slope_features(self, image, image_path):
        """Extract slope and terrain features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gradient analysis for slope estimation
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Slope magnitude and direction
        slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        slope_direction = np.arctan2(grad_y, grad_x)
        
        # Slope statistics
        mean_slope = np.mean(slope_magnitude)
        max_slope = np.max(slope_magnitude)
        slope_variance = np.var(slope_magnitude)
        
        # Steep area detection
        steep_threshold = np.percentile(slope_magnitude, 80)
        steep_areas = slope_magnitude > steep_threshold
        steep_percentage = (np.sum(steep_areas) / steep_areas.size) * 100
        
        # Slope consistency (uniform vs. varied)
        direction_variance = np.var(slope_direction)
        
        return {
            'estimated_slope_angle': float(mean_slope),
            'maximum_slope_angle': float(max_slope),
            'slope_variance': float(slope_variance),
            'steep_areas_percent': float(steep_percentage),
            'slope_direction_consistency': float(direction_variance)
        }
    
    def _assess_risk(self, features):
        """Assess overall rockfall risk based on extracted features"""
        risk_factors = []
        risk_scores = []
        
        # High crack density = high risk
        if features.get('crack_count', 0) > 10:
            risk_factors.append('high_crack_density')
            risk_scores.append(0.8)
        
        # High fracture density = high risk
        if features.get('fracture_density', 0) > 0.1:
            risk_factors.append('high_fracture_density')
            risk_scores.append(0.7)
        
        # Low vegetation = higher erosion risk
        if features.get('vegetation_coverage_percent', 0) < 20:
            risk_factors.append('low_vegetation_stability')
            risk_scores.append(0.6)
        
        # High weathering = increased risk
        if features.get('weathering_indicators', 0) > 10:
            risk_factors.append('significant_weathering')
            risk_scores.append(0.5)
        
        # Steep areas = higher risk
        if features.get('steep_areas_percent', 0) > 30:
            risk_factors.append('steep_terrain')
            risk_scores.append(0.6)
        
        # Large block size might be unstable
        if features.get('average_block_size', 0) > 1000:
            risk_factors.append('large_unstable_blocks')
            risk_scores.append(0.7)
        
        # Calculate overall risk
        if risk_scores:
            overall_risk_score = np.mean(risk_scores)
            if overall_risk_score > 0.7:
                risk_level = 'HIGH'
            elif overall_risk_score > 0.5:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
        else:
            overall_risk_score = 0.2
            risk_level = 'LOW'
        
        return {
            'risk_level': risk_level,
            'risk_score': float(overall_risk_score),
            'risk_factors': risk_factors,
            'confidence': min(0.95, 0.5 + len(risk_factors) * 0.1)
        }

class ModelTrainer:
    """Train ML models on drone image features"""
    
    def __init__(self, training_data_path):
        self.training_data_path = training_data_path
        self.analyzer = DroneImageAnalyzer()
        self.models = {}
        self.scalers = {}
        
    def extract_features_from_dataset(self):
        """Extract features from all training images"""
        print("🔍 Analyzing training images...")
        
        features_data = []
        image_files = [f for f in os.listdir(self.training_data_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif'))]
        
        for i, filename in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {filename}")
            
            image_path = os.path.join(self.training_data_path, filename)
            analysis = self.analyzer.analyze_image(image_path)
            
            if 'error' not in analysis:
                feature_row = {
                    'filename': filename,
                    **analysis['features'],
                    **analysis['risk_assessment']
                }
                features_data.append(feature_row)
        
        if not features_data:
            raise ValueError("No valid features extracted from images")
        
        self.features_df = pd.DataFrame(features_data)
        print(f"✅ Extracted features from {len(features_data)} images")
        
        return self.features_df
    
    def train_models(self):
        """Train ML models for risk prediction"""
        print("🤖 Training AI models...")
        
        if not hasattr(self, 'features_df'):
            self.extract_features_from_dataset()
        
        # Prepare features for training
        feature_columns = [col for col in self.features_df.columns 
                          if col not in ['filename', 'risk_level', 'risk_score', 'risk_factors', 'confidence']]
        
        X = self.features_df[feature_columns].fillna(0)
        
        # Train risk level classifier
        y_risk_level = self.features_df['risk_level']
        le_risk = LabelEncoder()
        y_risk_encoded = le_risk.fit_transform(y_risk_level)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_risk_encoded, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest for risk classification
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train_scaled, y_train)
        
        # Train Gradient Boosting for risk score regression
        y_risk_score = self.features_df['risk_score']
        y_score_train = y_risk_score.iloc[X_train.index]
        y_score_test = y_risk_score.iloc[X_test.index]
        
        gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_regressor.fit(X_train_scaled, y_score_train)
        
        # Evaluate models
        y_pred = rf_classifier.predict(X_test_scaled)
        score_pred = gb_regressor.predict(X_test_scaled)
        
        print("📊 Model Performance:")
        print("Risk Level Classification:")
        print(classification_report(y_test, y_pred, target_names=le_risk.classes_))
        
        print(f"Risk Score Regression MSE: {mean_squared_error(y_score_test, score_pred):.4f}")
        
        # Store models
        self.models = {
            'risk_classifier': rf_classifier,
            'risk_regressor': gb_regressor,
            'label_encoder': le_risk
        }
        
        self.scalers = {
            'feature_scaler': scaler
        }
        
        self.feature_columns = feature_columns
        
        print("✅ Model training completed!")
        
        return self.models, self.scalers
    
    def save_models(self, model_dir='ml_models'):
        """Save trained models to disk"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        joblib.dump(self.models['risk_classifier'], 
                   os.path.join(model_dir, 'drone_risk_classifier.pkl'))
        joblib.dump(self.models['risk_regressor'], 
                   os.path.join(model_dir, 'drone_risk_regressor.pkl'))
        joblib.dump(self.models['label_encoder'], 
                   os.path.join(model_dir, 'drone_label_encoder.pkl'))
        joblib.dump(self.scalers['feature_scaler'], 
                   os.path.join(model_dir, 'drone_feature_scaler.pkl'))
        
        # Save feature columns
        with open(os.path.join(model_dir, 'drone_feature_columns.json'), 'w') as f:
            json.dump(self.feature_columns, f)
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'num_training_images': len(self.features_df),
            'feature_count': len(self.feature_columns),
            'model_version': '2.0'
        }
        
        with open(os.path.join(model_dir, 'drone_model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Models saved to {model_dir}/")

def main():
    """Main training pipeline"""
    print("🚀 Starting Advanced Drone Image AI Training...")
    
    # Initialize trainer
    training_data_path = "training_data/drone_images"
    if not os.path.exists(training_data_path):
        print(f"❌ Training data not found at {training_data_path}")
        return
    
    trainer = ModelTrainer(training_data_path)
    
    try:
        # Extract features
        features_df = trainer.extract_features_from_dataset()
        print(f"📈 Feature extraction summary:")
        print(f"   - Images processed: {len(features_df)}")
        print(f"   - Features per image: {len([col for col in features_df.columns if col not in ['filename']])}")
        
        # Train models
        models, scalers = trainer.train_models()
        
        # Save models
        trainer.save_models()
        
        print("🎉 Training completed successfully!")
        print("🔧 Models are ready for integration with the upload system.")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return None

if __name__ == "__main__":
    trainer = main()