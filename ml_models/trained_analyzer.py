#!/usr/bin/env python3
"""
Trained AI Model Integration for Real-time Drone Image Analysis
Uses the trained models to provide actual geological analysis
"""

import os
import cv2
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import the analyzer from the trainer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from advanced_drone_trainer import DroneImageAnalyzer

class TrainedDroneAnalyzer:
    """Production analyzer using trained AI models"""
    
    def __init__(self, model_dir='ml_models'):
        self.model_dir = model_dir
        self.base_analyzer = DroneImageAnalyzer()
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.load_models()
    
    def load_models(self):
        """Load the trained models and scalers"""
        try:
            # Load models
            self.models = {
                'risk_classifier': joblib.load(os.path.join(self.model_dir, 'drone_risk_classifier.pkl')),
                'risk_regressor': joblib.load(os.path.join(self.model_dir, 'drone_risk_regressor.pkl')),
                'label_encoder': joblib.load(os.path.join(self.model_dir, 'drone_label_encoder.pkl'))
            }
            
            # Load scalers
            self.scalers = {
                'feature_scaler': joblib.load(os.path.join(self.model_dir, 'drone_feature_scaler.pkl'))
            }
            
            # Load feature columns
            with open(os.path.join(self.model_dir, 'drone_feature_columns.json'), 'r') as f:
                self.feature_columns = json.load(f)
            
            print("✅ Trained models loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Could not load trained models: {e}")
            print("Using base analyzer without ML predictions")
            self.models = {}
    
    def analyze_image(self, image_path):
        """Analyze drone image using trained AI models"""
        try:
            # Get base features from the analyzer
            base_analysis = self.base_analyzer.analyze_image(image_path)
            
            if 'error' in base_analysis:
                return base_analysis
            
            features = base_analysis['features']
            
            # If we have trained models, use them for enhanced prediction
            if self.models:
                enhanced_analysis = self._predict_with_models(features)
                
                # Combine base analysis with ML predictions
                result = {
                    'features': features,
                    'ml_predictions': enhanced_analysis,
                    'risk_assessment': enhanced_analysis['risk_assessment'],
                    'analysis_timestamp': datetime.now().isoformat(),
                    'model_version': '2.0',
                    'confidence': enhanced_analysis.get('confidence', 0.8)
                }
            else:
                # Fallback to base analysis
                result = base_analysis
                result['model_version'] = '1.0 (fallback)'
            
            return result
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _predict_with_models(self, features):
        """Use trained models to predict risk"""
        try:
            # Prepare feature vector
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0))
            
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Scale features
            feature_scaled = self.scalers['feature_scaler'].transform(feature_array)
            
            # Predict risk level
            risk_level_encoded = self.models['risk_classifier'].predict(feature_scaled)[0]
            risk_level = self.models['label_encoder'].inverse_transform([risk_level_encoded])[0]
            
            # Predict risk score
            risk_score = float(self.models['risk_regressor'].predict(feature_scaled)[0])
            
            # Get prediction probabilities for confidence
            risk_probabilities = self.models['risk_classifier'].predict_proba(feature_scaled)[0]
            confidence = float(np.max(risk_probabilities))
            
            # Generate detailed insights
            insights = self._generate_insights(features, risk_level, risk_score)
            
            return {
                'risk_assessment': {
                    'risk_level': risk_level,
                    'risk_score': risk_score,
                    'confidence': confidence,
                    'risk_factors': insights['risk_factors'],
                    'stability_indicators': insights['stability_indicators']
                },
                'detailed_analysis': insights['detailed_analysis'],
                'recommendations': insights['recommendations']
            }
            
        except Exception as e:
            # Fallback to base assessment if ML prediction fails
            return self.base_analyzer._assess_risk(features)
    
    def _generate_insights(self, features, risk_level, risk_score):
        """Generate detailed insights based on features"""
        risk_factors = []
        stability_indicators = []
        recommendations = []
        
        # Crack analysis
        crack_count = features.get('crack_count', 0)
        if crack_count > 15:
            risk_factors.append('High crack density detected')
            recommendations.append('Monitor crack propagation regularly')
        elif crack_count > 5:
            risk_factors.append('Moderate crack density')
            recommendations.append('Schedule periodic crack monitoring')
        else:
            stability_indicators.append('Low crack density')
        
        # Fracture density analysis
        fracture_density = features.get('fracture_density', 0)
        if fracture_density > 0.15:
            risk_factors.append('High fracture network density')
            recommendations.append('Conduct detailed structural geology survey')
        elif fracture_density > 0.05:
            risk_factors.append('Moderate fracture density')
        else:
            stability_indicators.append('Low fracture density')
        
        # Vegetation coverage
        vegetation = features.get('vegetation_coverage_percent', 0)
        if vegetation < 15:
            risk_factors.append('Low vegetation coverage - erosion risk')
            recommendations.append('Consider slope stabilization with vegetation')
        elif vegetation > 40:
            stability_indicators.append('Good vegetation coverage')
        
        # Weathering indicators
        weathering = features.get('weathering_indicators', 0)
        if weathering > 15:
            risk_factors.append('Significant rock weathering detected')
            recommendations.append('Assess rock strength and stability')
        
        # Slope analysis
        steep_areas = features.get('steep_areas_percent', 0)
        if steep_areas > 40:
            risk_factors.append('High percentage of steep terrain')
            recommendations.append('Install rockfall protection measures')
        elif steep_areas > 20:
            risk_factors.append('Moderate steep terrain')
        
        # Block size analysis
        block_size = features.get('average_block_size', 0)
        if block_size > 1500:
            risk_factors.append('Large potentially unstable rock blocks')
            recommendations.append('Assess individual block stability')
        
        # Moisture content
        moisture = features.get('moisture_content', 0)
        if moisture > 10:
            risk_factors.append('High moisture content detected')
            recommendations.append('Monitor for freeze-thaw cycles')
        
        # Generate detailed analysis summary
        detailed_analysis = {
            'geological_features': {
                'crack_density': crack_count,
                'fracture_network_strength': fracture_density,
                'surface_roughness': features.get('texture_roughness', 0),
                'bedding_plane_strength': features.get('bedding_plane_strength', 0)
            },
            'environmental_factors': {
                'vegetation_coverage': vegetation,
                'moisture_content': moisture,
                'weathering_degree': weathering,
                'slope_characteristics': {
                    'average_slope': features.get('estimated_slope_angle', 0),
                    'steep_areas_percent': steep_areas,
                    'slope_consistency': features.get('slope_direction_consistency', 0)
                }
            },
            'stability_assessment': {
                'overall_risk_level': risk_level,
                'numerical_risk_score': round(risk_score, 3),
                'primary_concerns': risk_factors[:3],  # Top 3 concerns
                'positive_indicators': stability_indicators
            }
        }
        
        # Add priority recommendations
        if risk_level == 'HIGH':
            recommendations.insert(0, 'URGENT: Implement immediate rockfall monitoring')
            recommendations.insert(1, 'Consider temporary access restrictions')
        elif risk_level == 'MEDIUM':
            recommendations.insert(0, 'Schedule regular monitoring inspections')
        
        return {
            'risk_factors': risk_factors,
            'stability_indicators': stability_indicators,
            'detailed_analysis': detailed_analysis,
            'recommendations': recommendations
        }

# Global analyzer instance
_analyzer_instance = None

def get_analyzer():
    """Get or create the global analyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = TrainedDroneAnalyzer()
    return _analyzer_instance

def analyze_drone_image(image_path):
    """Main function to analyze a drone image"""
    analyzer = get_analyzer()
    return analyzer.analyze_image(image_path)

if __name__ == "__main__":
    # Test the analyzer
    test_image = "training_data/drone_images/WhatsApp Image 2025-09-22 at 13.28.22_1f03489f.jpg"
    if os.path.exists(test_image):
        result = analyze_drone_image(test_image)
        print("🧪 Test Analysis Result:")
        print(json.dumps(result, indent=2))
    else:
        print("❌ Test image not found")