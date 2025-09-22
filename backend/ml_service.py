import sys
import os

# Add paths for rockfall_predictor module
ml_models_path = os.path.join(os.path.dirname(__file__), '..', 'ml_models')
if ml_models_path not in sys.path:
    sys.path.append(ml_models_path)

try:
    from rockfall_predictor import RockfallPredictor as OriginalRockfallPredictor
    # Create a global instance
    predictor_instance = OriginalRockfallPredictor()
except ImportError as e:
    print(f"Warning: Could not import rockfall_predictor: {e}")
    predictor_instance = None
    OriginalRockfallPredictor = None

class RockfallMLService:
    def __init__(self):
        self.predictor = predictor_instance
    
    def predict_rockfall_risk(self, sensor_data=None):
        if self.predictor:
            return self.predictor.predict_rockfall_risk(sensor_data)
        else:
            # Return dummy data if predictor not available
            return {"risk_level": "UNKNOWN", "probability": 0.0}
    
    def generate_risk_map(self, sensor_data_list):
        if self.predictor:
            return self.predictor.generate_risk_map(sensor_data_list)
        else:
            return {"zones": [], "message": "ML service not available"}
    
    def generate_forecast(self):
        if self.predictor:
            return self.predictor.generate_forecast()
        else:
            return {"forecast": [], "message": "ML service not available"}