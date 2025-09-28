# backend/detection/ml_utils.py

import torch
import joblib
import os

# Global placeholders for loaded models
_preprocessing_pipeline = None
_feature_extractor = None
_ensemble_model = None
_image_model = None
_models_loaded = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    global _preprocessing_pipeline, _feature_extractor, _ensemble_model, _image_model, _models_loaded

    if _models_loaded:
        return True

    try:
        # Example paths, adjust if needed
        _preprocessing_pipeline = joblib.load("backend/detection/preprocessing_pipeline.pkl")
        _feature_extractor = torch.load("backend/detection/cnn_model.pth", map_location=get_device())
        _ensemble_model = joblib.load("backend/detection/ensemble_model.pkl")
        _image_model = torch.load("backend/detection/maskrcnn_damage_detection.pth", map_location=get_device())
        _models_loaded = True
        return True
    except Exception as e:
        print(f"Model loading failed: {e}")
        return False

def fused_prediction(feature_extractor, ensemble_model, image_model, tabular_features, image_path, device):
    """
    Dummy function: integrate your existing inference logic here.
    Should return: prediction (0/1), confidence (0.0–1.0)
    """
    # Example:
    prediction = 0
    confidence = 0.75
    return prediction, confidence
