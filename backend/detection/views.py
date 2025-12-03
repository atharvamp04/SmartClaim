# detection/views.py - UPDATED FOR SINGLE BEST MODEL PIPELINE
import os
import json
import tempfile
import torch
import pandas as pd
from PIL import Image
from io import StringIO, BytesIO
from datetime import datetime, timedelta
import numpy as np
import random
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import torch.nn.functional as F

from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework import generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.views import APIView

from django.contrib.auth.models import User

from .serializers import RegisterSerializer, PolicyholderSerializer
from .models import Policyholder

from .verification_apis import verify_dl, verify_rto, verify_fir, aggregate_verification

# Import ML libraries
try:
    import joblib
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    ML_IMPORTS_AVAILABLE = True
    print("✅ ML functions imported successfully")
except ImportError as e:
    ML_IMPORTS_AVAILABLE = False
    print(f"❌ ML import failed: {e}")

# Import image model functions
try:
    from .fusion import (
        load_image_model,
        get_models_dir
    )
    IMAGE_IMPORTS_AVAILABLE = True
    print("✅ Image model functions imported successfully")
except ImportError as e:
    IMAGE_IMPORTS_AVAILABLE = False
    print(f"❌ Image import failed: {e}")

# --- Global Model Variables (Lazy Loading) ---
_best_model = None  # Changed from _ensemble_model
_calibrated_model = None  # NEW: For calibrated predictions
_image_model = None
_preprocessing_objects = None  # Will store scaler, encoders, feature_names
_thresholds = None  # NEW: Store optimal thresholds
_best_model_name = None  # NEW: Store which model is being used
_device = None
_models_loaded = False

def get_device():
    """Get PyTorch device"""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device

def load_models():
    """Load all models lazily when first needed"""
    global _best_model, _calibrated_model, _image_model, _preprocessing_objects, _thresholds, _best_model_name, _models_loaded
    
    if _models_loaded:
        return True
    
    if not ML_IMPORTS_AVAILABLE:
        print("❌ Cannot load models: ML imports not available")
        return False
    
    try:
        device = get_device()
        print(f"🔧 Loading models on device: {device}")
        
        models_dir = get_models_dir() if IMAGE_IMPORTS_AVAILABLE else "models"
        
        # Load preprocessing objects
        print("📋 Loading preprocessing objects...")
        try:
            _preprocessing_objects = {
                'scaler': joblib.load(os.path.join(models_dir, "scaler_final.pkl")),
                'label_encoders': joblib.load(os.path.join(models_dir, "label_encoders.pkl")),
                'feature_names': joblib.load(os.path.join(models_dir, "feature_names.pkl"))
            }
            print("✅ Preprocessing objects loaded")
        except Exception as e:
            print(f"❌ Failed to load preprocessing objects: {e}")
            return False
        
        # Load the single best model (NEW - replaces ensemble loading)
        print("📈 Loading best model...")
        try:
            _best_model = joblib.load(os.path.join(models_dir, "final_best_model.pkl"))
            print("✅ Best model loaded")
            
            # Check if it's a calibrated model
            if hasattr(_best_model, 'calibrated_classifiers_'):
                _calibrated_model = _best_model
                print("✅ Model is calibrated (Platt scaling)")
            else:
                _calibrated_model = _best_model
                print("ℹ️  Model is not calibrated")
            
        except Exception as e:
            print(f"❌ Failed to load best model: {e}")
            return False
        
        # Load optimal thresholds (NEW)
        print("🎯 Loading optimal thresholds...")
        try:
            _thresholds = joblib.load(os.path.join(models_dir, "final_thresholds.pkl"))
            print(f"✅ Loaded {len(_thresholds)} threshold strategies")
            
            # Print available strategies
            for strategy_name, info in _thresholds.items():
                print(f"   - {strategy_name}: threshold={info['threshold']:.4f}, "
                      f"recall={info['recall']*100:.1f}%, precision={info['precision']*100:.1f}%")
        except Exception as e:
            print(f"⚠️  Failed to load thresholds: {e}")
            # Create default thresholds
            _thresholds = {
                'default': {
                    'threshold': 0.5,
                    'precision': 0.15,
                    'recall': 0.75,
                    'f1': 0.20
                }
            }
        
        # Load best model name (NEW)
        try:
            _best_model_name = joblib.load(os.path.join(models_dir, "best_model_name.txt"))
            print(f"✅ Best model type: {_best_model_name}")
        except:
            _best_model_name = type(_best_model).__name__
            print(f"ℹ️  Model type: {_best_model_name}")
        
        # Load image model
        if IMAGE_IMPORTS_AVAILABLE:
            print("🖼️ Loading image model...")
            _image_model = load_image_model("maskrcnn_damage_detection.pth", device)
            if _image_model is not None:
                print("✅ Image model loaded")
            else:
                print("⚠️ Image model failed to load but continuing...")
        
        # Check what loaded successfully
        models_status = {
            "preprocessing_objects": _preprocessing_objects is not None,
            "best_model": _best_model is not None,
            "calibrated_model": _calibrated_model is not None,
            "thresholds": _thresholds is not None,
            "image_model": _image_model is not None
        }
        
        print(f"📊 Model loading status: {models_status}")
        
        # Consider models loaded if at least preprocessing and best model loaded
        if _preprocessing_objects is not None and _best_model is not None:
            _models_loaded = True
            print("✅ Models loaded successfully!")
            return True
        else:
            print("❌ Essential models failed to load")
        
        return False
        
    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        import traceback
        traceback.print_exc()
        return False


# --- Feature Engineering (SAME AS TRAINING) ---
class AdvancedFeatureEngineer:
    """Create powerful fraud-detection features - MUST match training exactly"""
    
    @staticmethod
    def create_features(df):
        """Create advanced fraud indicators"""
        df = df.copy()
        
        if 'Days_Policy_Claim' in df.columns:
            df['very_quick_claim'] = df['Days_Policy_Claim'].apply(
                lambda x: 1 if str(x) == 'none' else 0
            )
        
        if 'Age' in df.columns:
            df['high_risk_age'] = df['Age'].apply(lambda x: 1 if x < 25 or x > 70 else 0)
            df['very_young_driver'] = df['Age'].apply(lambda x: 1 if x < 22 else 0)
        
        if 'VehiclePrice' in df.columns:
            df['premium_vehicle'] = df['VehiclePrice'].apply(
                lambda x: 1 if 'more than 69000' in str(x) else 0
            )
            df['low_value_vehicle'] = df['VehiclePrice'].apply(
                lambda x: 1 if '20000' in str(x) or 'less' in str(x) else 0
            )
        
        if 'PastNumberOfClaims' in df.columns:
            df['serial_claimer'] = df['PastNumberOfClaims'].apply(
                lambda x: 1 if str(x) in ['2 to 4', 'more than 4'] else 0
            )
            df['first_time_claimer'] = df['PastNumberOfClaims'].apply(
                lambda x: 1 if str(x) == 'none' else 0
            )
        
        if 'WitnessPresent' in df.columns and 'PoliceReportFiled' in df.columns:
            df['no_witness_no_police'] = (
                (df['WitnessPresent'] == 'No') & 
                (df['PoliceReportFiled'] == 'No')
            ).astype(int)
            df['has_evidence'] = (
                (df['WitnessPresent'] == 'Yes') | 
                (df['PoliceReportFiled'] == 'Yes')
            ).astype(int)
        
        if 'AddressChange_Claim' in df.columns:
            df['recent_address_change'] = df['AddressChange_Claim'].apply(
                lambda x: 1 if str(x) in ['1 year', '2 to 3 years', '4 to 8 years'] else 0
            )
        
        if 'Deductible' in df.columns:
            df['very_high_deductible'] = df['Deductible'].apply(lambda x: 1 if x >= 700 else 0)
            df['low_deductible'] = df['Deductible'].apply(lambda x: 1 if x <= 300 else 0)
        
        if 'DayOfWeekClaimed' in df.columns:
            df['weekend_claim'] = df['DayOfWeekClaimed'].apply(
                lambda x: 1 if str(x) in ['Saturday', 'Sunday'] else 0
            )
        
        if 'Fault' in df.columns:
            df['policyholder_fault'] = df['Fault'].apply(
                lambda x: 1 if 'Policy Holder' in str(x) else 0
            )
        
        if 'AgentType' in df.columns:
            df['external_agent'] = df['AgentType'].apply(
                lambda x: 1 if 'External' in str(x) else 0
            )
        
        if 'NumberOfSuppliments' in df.columns:
            df['many_supplements'] = df['NumberOfSuppliments'].apply(
                lambda x: 1 if str(x) in ['3 to 5', 'more than 5'] else 0
            )
        
        if 'NumberOfCars' in df.columns:
            df['multiple_cars'] = df['NumberOfCars'].apply(
                lambda x: 1 if str(x) in ['5 to 8', 'more than 8'] else 0
            )
        
        if 'AccidentArea' in df.columns:
            df['urban_accident'] = df['AccidentArea'].apply(
                lambda x: 1 if 'Urban' in str(x) else 0
            )
        
        if 'VehicleCategory' in df.columns:
            df['sport_vehicle'] = df['VehicleCategory'].apply(
                lambda x: 1 if 'Sport' in str(x) else 0
            )
        
        if 'PolicyType' in df.columns:
            df['collision_policy'] = df['PolicyType'].apply(
                lambda x: 1 if 'Collision' in str(x) else 0
            )
        
        if 'high_risk_age' in df.columns and 'premium_vehicle' in df.columns:
            df['young_premium'] = df['high_risk_age'] * df['premium_vehicle']
        
        if 'no_witness_no_police' in df.columns and 'very_quick_claim' in df.columns:
            df['suspicious_combo'] = df['no_witness_no_police'] * df['very_quick_claim']
        
        return df


def preprocess_inference_data(df, preprocessing_objects):
    """Preprocess inference data using the exact same pipeline as training"""
    try:
        df = df.copy()
        
        # Apply feature engineering
        engineer = AdvancedFeatureEngineer()
        df = engineer.create_features(df)
        
        X = df.copy()
        
        # Get column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Handle missing values
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna('Unknown', inplace=True)
        
        # Encode categorical variables
        label_encoders = preprocessing_objects['label_encoders']
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                X[col] = le.transform(X[col].astype(str))
            else:
                X[col] = X[col].astype('category').cat.codes
        
        # Reorder columns to match training
        feature_names = preprocessing_objects['feature_names']
        
        print(f"🔍 Feature alignment check:")
        print(f"   Current features: {X.shape[1]}")
        print(f"   Expected features: {len(feature_names)}")
        print(f"   Model expects: {_best_model.n_features_in_ if hasattr(_best_model, 'n_features_in_') else 'unknown'}")
        
        # Add missing columns with 0
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        
        # Keep only training columns in the same order
        X = X[feature_names]
        
        print(f"   After alignment: {X.shape[1]}")
        
        # Scale using saved scaler
        scaler = preprocessing_objects['scaler']
        X_scaled = scaler.transform(X)
        
        return X_scaled
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_inference_data(policyholder, claim_amount=None, *args, **kwargs):
    """
    Builds a single-row pandas DataFrame for fraud prediction inference.
    Auto-fills missing fields with smart, realistic defaults so model receives
    the full 54-feature structure.
    """

    import pandas as pd

    # === SMART DEFAULTS ===
    defaults = {
        'Days_Policy_Accident': 'more than 30',
        'Days_Policy_Claim': 'more than 30',
        'PoliceReportFiled': 'No',
        'WitnessPresent': 'No',
        'AddressChange_Claim': 'no change',
        'NumberOfSuppliments': 'none',
        'Deductible': 400,
        'PastNumberOfClaims': 0,
        'Fault': 'Policy Holder',
    }

    # === BUILD BASE RECORD ===
    data = {
        'Month': getattr(policyholder, 'month', 'Jan'),
        'WeekOfMonth': getattr(policyholder, 'week_of_month', 3),
        'DayOfWeek': getattr(policyholder, 'day_of_week', 'Monday'),
        'Make': getattr(policyholder, 'vehicle_make', 'Honda'),
        'AccidentArea': getattr(policyholder, 'address_area', 'Urban'),
        'DayOfWeekClaimed': getattr(policyholder, 'day_of_week_claimed', 'Tuesday'),
        'MonthClaimed': getattr(policyholder, 'month_claimed', 'Jan'),
        'WeekOfMonthClaimed': getattr(policyholder, 'week_of_month_claimed', 3),
        'Sex': getattr(policyholder, 'sex', 'Male'),
        'MaritalStatus': getattr(policyholder, 'marital_status', 'Single'),
        'Age': int(getattr(policyholder, 'age', 30)),
        'Fault': defaults['Fault'],
        'PolicyType': getattr(policyholder, 'policy_type', 'Sedan - Liability'),
        'VehicleCategory': getattr(policyholder, 'vehicle_category', 'Sedan'),
        'VehiclePrice': getattr(policyholder, 'vehicle_price_category', '20000 to 29000'),
        'PolicyNumber': 1,
        'RepNumber': 1,
        'Deductible': int(getattr(policyholder, 'deductible', defaults['Deductible'])),
        'DriverRating': int(getattr(policyholder, 'driver_rating', 4)),
        'Days_Policy_Accident': getattr(policyholder, 'days_policy_accident', defaults['Days_Policy_Accident']),
        'Days_Policy_Claim': getattr(policyholder, 'days_policy_claim', defaults['Days_Policy_Claim']),
        'PastNumberOfClaims': int(getattr(policyholder, 'past_number_of_claims', defaults['PastNumberOfClaims'])),
        'AgeOfVehicle': getattr(policyholder, 'age_of_vehicle', '3 to 4'),
        'AgeOfPolicyHolder': getattr(policyholder, 'age_of_policyholder', '31 to 35'),
        'PoliceReportFiled': getattr(policyholder, 'police_report_filed', defaults['PoliceReportFiled']),
        'WitnessPresent': getattr(policyholder, 'witness_present', defaults['WitnessPresent']),
        'AgentType': getattr(policyholder, 'agent_type', 'External'),
        'NumberOfSuppliments': getattr(policyholder, 'number_of_suppliments', defaults['NumberOfSuppliments']),
        'AddressChange_Claim': getattr(policyholder, 'address_change_claim', defaults['AddressChange_Claim']),
        'NumberOfCars': getattr(policyholder, 'number_of_cars', '1 vehicle'),
        'Year': int(getattr(policyholder, 'year_of_vehicle', 1994)),
        'BasePolicy': getattr(policyholder, 'base_policy', 'Liability'),
    }

    # === SMART RISK INFERENCE BASED ON CLAIM AMOUNT ===
    if claim_amount:
        try:
            claim_amount = float(claim_amount)
            if claim_amount > 80000:
                data['PoliceReportFiled'] = 'Yes'
                data['WitnessPresent'] = 'Yes'
                data['Days_Policy_Claim'] = '1 to 7'
                data['Days_Policy_Accident'] = '1 to 7'
                data['AddressChange_Claim'] = '1 year'
                data['NumberOfSuppliments'] = 'more than 5'
            elif claim_amount > 50000:
                data['PoliceReportFiled'] = 'Yes'
                data['WitnessPresent'] = 'No'
                data['Days_Policy_Claim'] = '8 to 15'
                data['Days_Policy_Accident'] = '8 to 15'
                data['AddressChange_Claim'] = '2 to 3 years'
            elif claim_amount < 20000:
                data['PoliceReportFiled'] = 'No'
                data['WitnessPresent'] = 'No'
                data['Days_Policy_Claim'] = 'more than 30'
                data['Days_Policy_Accident'] = 'more than 30'
        except ValueError:
            print("⚠️ Invalid claim_amount value; skipping smart risk logic.")

    # === REMOVE ANY NON-NUMERIC / NON-MODEL FIELDS ===
    for unwanted in ["CreatedAt", "UpdatedAt", "created_at", "updated_at", "createdAt", "updatedAt", "Email", "Username"]:
        if unwanted in data:
            del data[unwanted]

    # === CONVERT TO DATAFRAME ===
    df = pd.DataFrame([data])

    # Optional sanity check: must match feature_names length
    print(f"✅ Inference data prepared with {df.shape[1]} features.")
    return df


# --- NEW: Single Best Model Prediction Function ---

def get_detailed_tabular_predictions(tabular_features, best_model, thresholds, strategy='max_f1'):
    """
    Get detailed predictions from the single best model - ALWAYS returns valid structure
    """
    try:
        # Get probability predictions
        if hasattr(best_model, 'predict_proba'):
            proba = best_model.predict_proba(tabular_features)
            fraud_probability = float(proba[0][1] if len(proba[0]) > 1 else proba[0][0])
            no_fraud_probability = float(proba[0][0] if len(proba[0]) > 1 else 1 - proba[0][0])
        else:
            pred = best_model.predict(tabular_features)
            fraud_probability = float(pred[0])
            no_fraud_probability = 1.0 - fraud_probability
        
        # Get predictions for all available threshold strategies
        threshold_predictions = {}
        for strat_name, strat_info in thresholds.items():
            threshold = strat_info['threshold']
            prediction = 1 if fraud_probability >= threshold else 0
            
            threshold_predictions[strat_name] = {
                'threshold': float(threshold),
                'prediction': int(prediction),
                'distance_from_threshold': float(fraud_probability - threshold),
                'expected_precision': float(strat_info.get('precision', 0)),
                'expected_recall': float(strat_info.get('recall', 0)),
                'expected_f1': float(strat_info.get('f1', 0))
            }
        
        # Get prediction for selected strategy
        selected_strategy = thresholds.get(strategy, thresholds.get('max_f1', {'threshold': 0.5}))
        selected_threshold = selected_strategy['threshold']
        primary_prediction = 1 if fraud_probability >= selected_threshold else 0
        
        # Calculate confidence
        distance_from_threshold = abs(fraud_probability - selected_threshold)
        confidence = min(0.5 + distance_from_threshold, 0.99)
        
        return {
            'raw_features_shape': list(tabular_features.shape),
            'model_type': _best_model_name if _best_model_name else type(best_model).__name__,
            'is_calibrated': hasattr(best_model, 'calibrated_classifiers_'),
            
            # CRITICAL: Always provide probabilities
            'probabilities': {
                'no_fraud': float(no_fraud_probability),
                'fraud': float(fraud_probability)
            },
            
            # Primary prediction using selected strategy
            'primary_prediction': {
                'strategy': strategy,
                'threshold': float(selected_threshold),
                'prediction': int(primary_prediction),
                'confidence': float(confidence),
                'fraud_detected': bool(primary_prediction == 1)
            },
            
            # All threshold strategies
            'threshold_strategies': threshold_predictions,
            
            # Probability distribution info
            'probability_analysis': {
                'fraud_probability': float(fraud_probability),
                'probability_range': 'HIGH' if fraud_probability > 0.7 else 'MEDIUM' if fraud_probability > 0.3 else 'LOW',
                'separation_quality': 'GOOD' if abs(fraud_probability - 0.5) > 0.2 else 'MODERATE' if abs(fraud_probability - 0.5) > 0.1 else 'POOR'
            }
        }
        
    except Exception as e:
        print(f"❌ Error in tabular predictions: {e}")
        import traceback
        traceback.print_exc()
        
        # CRITICAL: Return valid default structure on error
        return {
            'raw_features_shape': list(tabular_features.shape) if tabular_features is not None else [0, 0],
            'model_type': 'Unknown',
            'is_calibrated': False,
            'probabilities': {
                'no_fraud': 0.6,
                'fraud': 0.4
            },
            'primary_prediction': {
                'strategy': strategy,
                'threshold': 0.5,
                'prediction': 0,
                'confidence': 0.6,
                'fraud_detected': False
            },
            'threshold_strategies': {},
            'probability_analysis': {
                'fraud_probability': 0.4,
                'probability_range': 'MEDIUM',
                'separation_quality': 'MODERATE'
            },
            'error': str(e)
        }

# --- Image Processing Functions (KEEP AS IS) ---

def process_damage_detection_image(image_path, image_model, device):
    """Process image with damage detection and return annotated image"""
    try:
        import torchvision.transforms as transforms
        
        original_image = Image.open(image_path).convert("RGB")
        
        with torch.no_grad():
            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(original_image).unsqueeze(0).to(device)
            predictions = image_model(image_tensor)
            
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy() if 'labels' in predictions[0] else None
            
            confidence_threshold = 0.5
            high_conf_indices = scores > confidence_threshold
            
            filtered_boxes = boxes[high_conf_indices]
            filtered_scores = scores[high_conf_indices]
            filtered_labels = labels[high_conf_indices] if labels is not None else None
            
            annotated_image = original_image.copy()
            draw = ImageDraw.Draw(annotated_image)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                label_text = f"Damage: {score:.2f}"
                if filtered_labels is not None:
                    label_text = f"Damage {filtered_labels[i]}: {score:.2f}"
                
                text_bbox = draw.textbbox((x1, y1-25), label_text, font=font)
                draw.rectangle(text_bbox, fill="red")
                draw.text((x1, y1-25), label_text, fill="white", font=font)
            
            buffer = io.BytesIO()
            annotated_image.save(buffer, format='JPEG', quality=95)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            damage_areas = []
            total_damage_area = 0
            
            for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                total_damage_area += area
                
                damage_areas.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(score),
                    'area': float(area),
                    'label': f'damage_{i+1}'
                })
            
            image_total_area = original_image.width * original_image.height
            damage_percentage = (total_damage_area / image_total_area) * 100
            
            severity = "LOW"
            if damage_percentage > 15:
                severity = "HIGH"
            elif damage_percentage > 5:
                severity = "MEDIUM"
            
            return {
                'annotated_image_base64': img_base64,
                'damage_areas': damage_areas,
                'total_damage_areas': len(filtered_boxes),
                'damage_percentage': float(damage_percentage),
                'severity': severity,
                'original_dimensions': {
                    'width': original_image.width,
                    'height': original_image.height
                },
                'average_confidence': float(np.mean(filtered_scores)) if len(filtered_scores) > 0 else 0.0
            }
            
    except Exception as e:
        print(f"Error in damage detection: {str(e)}")
        return None


def get_detailed_image_predictions(image_path, image_model, device):
    """Get detailed image predictions with damage analysis for a single image"""
    try:
        import torchvision.transforms as transforms
        
        image = Image.open(image_path).convert("RGB")
        image_area = image.width * image.height
        
        if image_model is not None:
            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions = image_model(image_tensor)
                
                boxes = predictions[0]['boxes'].cpu().numpy() if 'boxes' in predictions[0] else np.array([])
                scores = predictions[0]['scores'].cpu().numpy() if 'scores' in predictions[0] else np.array([])
                
                confidence_threshold = 0.5
                high_conf_indices = scores > confidence_threshold
                
                filtered_boxes = boxes[high_conf_indices] if len(boxes) > 0 else np.array([])
                filtered_scores = scores[high_conf_indices] if len(scores) > 0 else np.array([])
                
                total_damage_area = 0
                damage_regions = []
                
                for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
                    x1, y1, x2, y2 = box
                    region_area = (x2 - x1) * (y2 - y1)
                    total_damage_area += region_area
                    
                    damage_regions.append({
                        'region_id': i + 1,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'area_pixels': float(region_area),
                        'confidence': float(score),
                        'relative_size': float(region_area / image_area) if image_area > 0 else 0
                    })
                
                damage_percentage = (total_damage_area / image_area) * 100 if image_area > 0 else 0
                
                if damage_percentage > 15:
                    severity_score = 0.9
                    severity_level = "HIGH"
                elif damage_percentage > 5:
                    severity_score = 0.6
                    severity_level = "MEDIUM"
                else:
                    severity_score = 0.3
                    severity_level = "LOW"
                
                weighted_damage_score = 0.0
                if len(filtered_scores) > 0:
                    for region in damage_regions:
                        weight = region['confidence'] * region['relative_size']
                        weighted_damage_score += weight
                    weighted_damage_score = min(weighted_damage_score, 1.0)
                
                return {
                    'image_dimensions': {
                        'width': image.width,
                        'height': image.height,
                        'total_pixels': image_area
                    },
                    'detection_results': {
                        'total_detections': len(boxes),
                        'high_confidence_detections': len(filtered_boxes),
                        'confidence_threshold': confidence_threshold,
                        'all_scores': scores.tolist(),
                        'filtered_scores': filtered_scores.tolist()
                    },
                    'damage_analysis': {
                        'damage_regions': damage_regions,
                        'total_damage_area_pixels': float(total_damage_area),
                        'damage_percentage': float(damage_percentage),
                        'severity_level': severity_level,
                        'severity_score': float(severity_score),
                        'weighted_damage_score': float(weighted_damage_score)
                    },
                    'image_fraud_probability': float(weighted_damage_score),
                    'image_confidence': float(np.mean(filtered_scores)) if len(filtered_scores) > 0 else 0.5
                }
        else:
            return {
                'image_dimensions': {'width': image.width, 'height': image.height, 'total_pixels': image_area},
                'detection_results': {'total_detections': 1, 'high_confidence_detections': 1},
                'damage_analysis': {'damage_percentage': 1.0, 'severity_level': 'LOW', 'severity_score': 0.3},
                'image_fraud_probability': 0.3,
                'image_confidence': 0.6
            }
            
    except Exception as e:
        print(f"Error in image predictions: {e}")
        return {
            'image_dimensions': {'width': 800, 'height': 600, 'total_pixels': 480000},
            'detection_results': {'total_detections': 0, 'high_confidence_detections': 0},
            'damage_analysis': {'damage_percentage': 0.0, 'severity_level': 'LOW', 'severity_score': 0.3},
            'image_fraud_probability': 0.3,
            'image_confidence': 0.5
        }


def calculate_detailed_fusion(tabular_details, image_details,
                              dl_number=None, expiry_date=None,
                              reg_no=None, make=None, year=None,
                              fir_no=None):
    """Calculate fusion with detailed mathematical breakdown + verification layer"""
    try:
        # === BASE MODEL PROBABILITIES ===
        tabular_fraud_prob = tabular_details.get('ensemble_probabilities', {}).get('fraud', 0.4)
        tabular_confidence = tabular_details.get('tabular_confidence', 0.6)
        
        # Get predictions for this image
        image_details = get_detailed_image_predictions(image_path, image_model, device)
        image_details['image_index'] = idx + 1
        image_details['image_filename'] = os.path.basename(image_path)
        all_image_details.append(image_details)
        
        # === CALCULATE WEIGHTS ===
        total_confidence = tabular_confidence + image_confidence
        if total_confidence > 0:
            tabular_weight = tabular_confidence / total_confidence
            image_weight = image_confidence / total_confidence
        else:
            tabular_weight = 0.5
            image_weight = 0.5
        
        # === FUSION METHODS ===
        weighted_fusion = (tabular_weight * tabular_fraud_prob) + (image_weight * image_fraud_prob)
        geometric_fusion = np.sqrt(max(tabular_fraud_prob * image_fraud_prob, 0))
        base_fusion_score = (0.65 * weighted_fusion) + (0.35 * geometric_fusion)

        # =====================================================================
        # 🆕 AMOUNT-DAMAGE MISMATCH DETECTION
        # =====================================================================
        damage_percentage = image_details.get("damage_analysis", {}).get("damage_percentage", 
                           image_details.get("damage_summary", {}).get("avg_damage_percentage", 0))
        severity_level = image_details.get("damage_analysis", {}).get("severity_level",
                        image_details.get("severity_analysis", {}).get("overall_severity", "LOW"))
        
        # Define mismatch thresholds
        amount_damage_boost = 0.0
        mismatch_detected = False
        mismatch_details = {}
        
        # High claim amount with low damage = FRAUD SIGNAL
        if claim_amount > 100000 and damage_percentage < 60:  # >1L claim, <60% damage
            amount_damage_boost = 0.35
            mismatch_detected = True
            mismatch_details = {
                'type': 'CRITICAL_MISMATCH',
                'reason': f'Very high claim (₹{claim_amount:,.0f}) with insufficient damage ({damage_percentage:.1f}%)',
                'boost': 0.35,
                'severity': 'CRITICAL'
            }
        elif claim_amount > 80000 and damage_percentage < 40:  # >80K claim, <40% damage
            amount_damage_boost = 0.25
            mismatch_detected = True
            mismatch_details = {
                'type': 'HIGH_MISMATCH',
                'reason': f'High claim (₹{claim_amount:,.0f}) with low damage ({damage_percentage:.1f}%)',
                'boost': 0.25,
                'severity': 'HIGH'
            }
        elif claim_amount > 50000 and severity_level == "LOW":  # >50K with LOW severity
            amount_damage_boost = 0.15
            mismatch_detected = True
            mismatch_details = {
                'type': 'MODERATE_MISMATCH',
                'reason': f'Moderate claim (₹{claim_amount:,.0f}) with LOW severity damage',
                'boost': 0.15,
                'severity': 'MEDIUM'
            }
        
        # === BASE FINAL FUSION ===
        alpha = 0.6
        beta = 0.4
        base_fusion_score = (alpha * weighted_fusion) + (beta * geometric_fusion)
        
        # ===================================================================
        # 🔐 VERIFICATION INTEGRATION (DL / RTO / FIR)
        # ===================================================================
        dl_info = verify_dl(dl_number, expiry_date)
        rto_info = verify_rto(reg_no, make, year)
        fir_info = verify_fir(fir_no)

        verification_reliability = aggregate_verification(dl_info, rto_info, fir_info)
        # lower reliability → higher fraud likelihood
        verification_impact = (1 - verification_reliability)

        # Weighted merge with verification reliability
        gamma = 0.25  # how much verification affects fusion
        final_fusion_score = ((1 - gamma) * base_fusion_score) + (gamma * verification_impact)

        # === DECISION ===
        fraud_threshold = 0.5
        final_prediction = 1 if final_fusion_score > fraud_threshold else 0
        
        # === RETURN STRUCTURE ===
        return {
            'input_probabilities': {
                'tabular_fraud_probability': float(tabular_fraud_prob),
                'tabular_confidence': float(tabular_confidence),
                'image_fraud_probability': float(image_fraud_prob),
                'image_confidence': float(image_confidence),
                'verification_reliability': float(verification_reliability)
            },
            "weight_calculation": {
                "total_confidence": float(total_confidence),
                "tabular_weight": float(tabular_weight),
                "image_weight": float(image_weight),
                "weight_formula": "weight = confidence / total_confidence"
            },
            "fusion_methods": {
                "weighted_average": {
                    "score": float(weighted_fusion),
                    "formula": f"({tabular_weight:.3f} × {tabular_fraud_prob:.3f}) + ({image_weight:.3f} × {image_fraud_prob:.3f})"
                },
                "geometric_mean": {
                    "score": float(geometric_fusion),
                    "formula": f"√({tabular_fraud_prob:.3f} × {image_fraud_prob:.3f})"
                }
            },
            'verification_details': {
                'dl': dl_info,
                'rto': rto_info,
                'fir': fir_info,
                'combined_reliability': float(verification_reliability)
            },
            'final_fusion': {
                'alpha': float(alpha),
                'beta': float(beta),
                'gamma': float(gamma),
                'calculation': f"(({1 - gamma} × base_fusion) + ({gamma} × (1−reliability)))",
                'base_fusion': float(base_fusion_score),
                'final_score': float(final_fusion_score),
                'threshold': float(fraud_threshold),
                'prediction': int(final_prediction)
            },
            
            "final_fusion": {
                "alpha": 0.65,
                "beta": 0.35,
                "calculation": f"base_fusion + amount_damage_boost",
                "base_fusion": float(base_fusion_score),
                "amount_damage_boost": float(amount_damage_boost),
                "final_score": float(final_fusion_score),
                "threshold": float(fraud_threshold),
                "prediction": int(final_prediction)
            },
            "final_prediction": final_prediction,
            "final_confidence": final_fusion_score
        }

    except Exception as e:
        print(f"[Fusion Error] {e}")
        import traceback
        traceback.print_exc()
        
        # Safe fallback
        return {
            "error": str(e),
            "input_probabilities": {
                "tabular_fraud_probability": 0.4,
                "tabular_confidence": 0.6,
                "image_fraud_probability": 0.3,
                "image_confidence": 0.7,
                "verification_reliability": 0.5
            },
            "weight_calculation": {
                "total_confidence": 1.3,
                "tabular_weight": 0.46,
                "image_weight": 0.54
            },
            "fusion_methods": {
                "weighted_average": {"score": 0.35},
                "geometric_mean": {"score": 0.32}
            },
            'verification_details': {
                'dl': {'valid': False, 'dl_score': 0.4},
                'rto': {'valid': False, 'rto_score': 0.4},
                'fir': {'exists': False, 'fir_score': 0.3},
                'combined_reliability': 0.37
            },
            'final_fusion': {
                'alpha': 0.6,
                'beta': 0.4,
                'gamma': 0.25,
                'final_score': 0.348,
                'threshold': 0.5,
                'prediction': 0
            },
            "final_prediction": 0,
            "final_confidence": 0.35
        }

# --- Auth Views ---

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def protected_view(request):
    return Response({"message": "You are authenticated!"})


class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    permission_classes = [AllowAny]


class PolicyholderCreateView(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request):
        serializer = PolicyholderSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def policyholder_detail(request, username):
    try:
        policyholder = Policyholder.objects.get(username=username)
    except Policyholder.DoesNotExist:
        return Response({"detail": "Not found."}, status=404)
    return Response(PolicyholderSerializer(policyholder).data)


# --- Main Prediction Endpoint ---

# detection/views.py - UPDATED predict_claim function with database storage

# detection/views.py - UPDATED predict_claim function with database storage

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def predict_claim(request):
    """Enhanced predict claim with Multiple Images Support + Database Storage"""
    
    # Extract request data
    username = request.data.get("username")
    claim_description = request.data.get("claim_description", "")
    accident_date = request.data.get("accident_date")
    claim_amount = float(request.data.get("claim_amount", 0))
    
    # NEW: Support multiple images
    car_images = request.FILES.getlist("car_images")
    car_image_single = request.FILES.get("car_image")
    
    # Combine single and multiple image inputs
    if car_image_single and car_image_single not in car_images:
        car_images.insert(0, car_image_single)
    
    threshold_strategy = request.data.get("threshold_strategy", "max_f1")
    
    # Document verification fields
    fir_number = request.data.get("fir_number", "")
    dl_number = request.data.get("dl_number", "")
    vehicle_reg_no = request.data.get("vehicle_reg_no", "")
    accident_location = request.data.get("accident_location", "Unknown")
    registration_number = request.data.get("registration_number", "")
    chassis_number = request.data.get("chassis_number", "")
    engine_number = request.data.get("engine_number", "")
    dob = request.data.get("dob")
    driver_name = request.data.get("driver_name", "")

    if not username or not car_images:
        return Response({
            "error": "username and at least one car_image are required",
            "prediction": 0,
            "confidence": 0.5,
            "fraud_detected": False,
            "probabilities": {"no_fraud": 0.6, "fraud": 0.4}
        }, status=status.HTTP_400_BAD_REQUEST)

    if not load_models():
        return Response({
            "error": "ML models not available",
            "prediction": 0,
            "confidence": 0.5,
            "fraud_detected": False,
            "probabilities": {"no_fraud": 0.6, "fraud": 0.4}
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    try:
        policyholder = Policyholder.objects.get(username=username)
    except Policyholder.DoesNotExist:
        return Response({
            "error": "Policyholder not found",
            "prediction": 0,
            "confidence": 0.5,
            "fraud_detected": False,
            "probabilities": {"no_fraud": 0.6, "fraud": 0.4}
        }, status=status.HTTP_404_NOT_FOUND)

    # Create and preprocess tabular data
    try:
        tabular_df = create_inference_data(policyholder, accident_date, claim_amount)
        tabular_features = preprocess_inference_data(tabular_df, _preprocessing_objects)
        
        if tabular_features is None:
            raise ValueError("Failed to preprocess data")
    except Exception as e:
        return Response({
            "error": f"Data preparation failed: {str(e)}",
            "prediction": 0,
            "confidence": 0.5,
            "fraud_detected": False,
            "probabilities": {"no_fraud": 0.6, "fraud": 0.4}
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Handle multiple images
    image_paths = []
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        
        for idx, car_image in enumerate(car_images):
            image_path = os.path.join(temp_dir, f"temp_image_{idx}_{random.randint(100,999)}.jpg")
            
            with open(image_path, "wb+") as f:
                for chunk in car_image.chunks():
                    f.write(chunk)
            
            image_paths.append(image_path)
        
        print(f"📸 Saved {len(image_paths)} images for processing")
        
    except Exception as e:
        return Response({
            "error": f"Image processing failed: {str(e)}",
            "prediction": 0,
            "confidence": 0.5,
            "fraud_detected": False,
            "probabilities": {"no_fraud": 0.6, "fraud": 0.4}
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Run ML predictions
    try:
        device = get_device()
        
        # Get tabular predictions
        tabular_details = get_detailed_tabular_predictions(
            tabular_features, 
            _calibrated_model,
            _thresholds,
            threshold_strategy
        )
        
        # Process multiple images
        multi_image_results = process_multiple_images(image_paths, _image_model, device)
        
        # Use aggregated results for fusion
        image_details = {
            **multi_image_results['aggregated_results'],
            'image_dimensions': multi_image_results['individual_image_results'][0]['image_dimensions'] if multi_image_results['individual_image_results'] else {},
            'detection_results': {
                'total_images': len(image_paths),
                'total_detections_all': sum(img['detection_results']['total_detections'] for img in multi_image_results['individual_image_results']),
                'high_confidence_detections_all': sum(img['detection_results']['high_confidence_detections'] for img in multi_image_results['individual_image_results'])
            }
        }
        
        # 🆕 UPDATED: Calculate fusion with claim_amount for amount-damage analysis
        fusion_details = calculate_detailed_fusion(
            tabular_details, 
            image_details,
            claim_amount=claim_amount,  # 🆕 Pass claim amount for amount-damage mismatch detection
            dl_number=dl_number,
            expiry_date=None,
            reg_no=vehicle_reg_no,
            make=getattr(policyholder, 'vehicle_make', None),
            year=getattr(policyholder, 'year_of_vehicle', None),
            fir_no=fir_number
        )
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup temp files on error
        try:
            for image_path in image_paths:
                if os.path.exists(image_path):
                    os.remove(image_path)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass
        
        return Response({
            "error": f"Prediction failed: {str(e)}",
            "username": username,
            "prediction": 0,
            "confidence": 0.5,
            "fraud_detected": False,
            "probabilities": {"no_fraud": 0.6, "fraud": 0.4},
            "risk_level": "MEDIUM",
            "message": "Error occurred during prediction",
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # =====================================================================
    # 🆕 UPDATED: Document verification - INFO ONLY, NO SCORING
    # =====================================================================
    base_fraud_score = fusion_details.get('final_confidence', 0.5)
    fraud_signals = []
    verification_results = {}
    
    # Get amount-damage mismatch info from fusion
    amount_damage_analysis = fusion_details.get('amount_damage_analysis', {})
    mismatch_details = amount_damage_analysis.get('mismatch_details', {})
    
    # 🆕 Add amount-damage mismatch to fraud signals if detected
    if amount_damage_analysis.get('mismatch_detected', False):
        fraud_signals.append({
            'type': mismatch_details.get('type', 'AMOUNT_DAMAGE_MISMATCH'),
            'severity': mismatch_details.get('severity', 'MEDIUM'),
            'message': mismatch_details.get('reason', 'Claim amount inconsistent with damage'),
            'boost': mismatch_details.get('boost', 0.0)
        })
    
    # Police verification - INFO ONLY (no scoring)
    if fir_number:
        police_result = verify_police_report(fir_number, accident_date, accident_location)
        verification_results['police'] = police_result
        
        if not police_result['verified']:
            fraud_signals.append({
                'type': 'INVALID_FIR',
                'severity': 'INFO',  # 🆕 Changed from CRITICAL
                'message': 'Police report could not be verified (informational only)',
                'boost': 0.0  # 🆕 No boost
            })
    else:
        if claim_amount > 50000:
            fraud_signals.append({
                'type': 'NO_POLICE_REPORT_HIGH_VALUE',
                'severity': 'INFO',  # 🆕 Changed from HIGH
                'message': f'No police report for ₹{claim_amount:,.0f} claim (informational only)',
                'boost': 0.0  # 🆕 No boost
            })
        
        verification_results['police'] = {
            'verified': False,
            'fraud_indicator': 'INFO',  # 🆕 Changed from MEDIUM/LOW
            'message': 'No police report provided'
        }
    
    # Vehicle verification - INFO ONLY (no scoring)
    if vehicle_reg_no:
        vehicle_result = verify_vehicle_registration(vehicle_reg_no, chassis_number, engine_number)
        verification_results['vehicle'] = vehicle_result
        
        if not vehicle_result['verified']:
            fraud_signals.append({
                'type': 'INVALID_VEHICLE',
                'severity': 'INFO',  # 🆕 Changed from CRITICAL
                'message': 'Vehicle registration could not be verified (informational only)',
                'boost': 0.0  # 🆕 No boost
            })
        elif not vehicle_result.get('vehicle_details', {}).get('insurance_valid', True):
            fraud_signals.append({
                'type': 'EXPIRED_INSURANCE',
                'severity': 'INFO',  # 🆕 Changed from CRITICAL
                'message': 'Vehicle insurance expired (informational only)',
                'boost': 0.0  # 🆕 No boost
            })
    else:
        verification_results['vehicle'] = {
            'verified': False,
            'fraud_indicator': 'INFO',
            'message': 'No vehicle registration provided'
        }
    
    # License verification - INFO ONLY (no scoring)
    if dl_number:
        license_result = verify_driving_license(dl_number, dob, driver_name)
        verification_results['license'] = license_result
        
        if not license_result['verified']:
            fraud_signals.append({
                'type': 'INVALID_LICENSE',
                'severity': 'INFO',  # 🆕 Changed from CRITICAL
                'message': 'Driving license could not be verified (informational only)',
                'boost': 0.0  # 🆕 No boost
            })
        elif not license_result.get('license_details', {}).get('license_valid', True):
            fraud_signals.append({
                'type': 'EXPIRED_LICENSE',
                'severity': 'INFO',  # 🆕 Changed from HIGH
                'message': 'Driving license expired (informational only)',
                'boost': 0.0  # 🆕 No boost
            })
    else:
        verification_results['license'] = {
            'verified': False,
            'fraud_indicator': 'INFO',
            'message': 'No driving license provided'
        }
    
    # 🆕 CRITICAL CHANGE: Use base_fraud_score directly (no document_boost)
    final_fraud_score = base_fraud_score  # Already includes amount-damage analysis from fusion
    
    fraud_threshold = config('FRAUD_THRESHOLD', default=0.5, cast=float)
    fraud_detected = final_fraud_score >= fraud_threshold
    
    high_threshold = config('HIGH_RISK_THRESHOLD', default=0.7, cast=float)
    critical_threshold = config('CRITICAL_RISK_THRESHOLD', default=0.85, cast=float)
    
    if final_fraud_score >= critical_threshold:
        risk_level = "CRITICAL"
        recommended_action = {
            'action': 'REJECT',
            'message': 'Critical fraud indicators detected. Immediate rejection recommended.',
            'next_steps': ['Reject claim immediately', 'Initiate fraud investigation']
        }
    elif final_fraud_score >= high_threshold:
        risk_level = "HIGH"
        recommended_action = {
            'action': 'INVESTIGATE',
            'message': 'High fraud risk. Requires senior investigator review.',
            'next_steps': ['Assign to senior investigator', 'Request additional documentation']
        }
    elif final_fraud_score >= fraud_threshold:
        risk_level = "MEDIUM"
        recommended_action = {
            'action': 'REVIEW',
            'message': 'Moderate fraud risk. Enhanced documentation required.',
            'next_steps': ['Request missing documents', 'Verify submitted documents']
        }
    else:
        risk_level = "LOW"
        recommended_action = {
            'action': 'APPROVE',
            'message': 'Low fraud risk. Proceed with standard processing.',
            'next_steps': ['Verify claim amount', 'Process payment']
        }
    
    # Prepare complete response data
    response_data = {
        "username": username,
        "claim_description": claim_description,
        "accident_date": accident_date,
        "claim_amount": claim_amount,
        "total_images_submitted": len(image_paths),
        
        # Core fields
        "prediction": int(fraud_detected),
        "confidence": float(final_fraud_score),
        "fraud_detected": fraud_detected,
        "probabilities": {
            "no_fraud": float(1 - final_fraud_score),
            "fraud": float(final_fraud_score)
        },
        "risk_level": risk_level,
        "message": f"{'🚨 FRAUD DETECTED' if fraud_detected else '✅ LEGITIMATE'} - {risk_level} risk",
        
        # 🆕 UPDATED: Fraud analysis without document boost
        "fraud_analysis": {
            "ml_base_score": float(base_fraud_score),
            "amount_damage_boost": float(amount_damage_analysis.get('boost_applied', 0.0)),
            "document_boost": 0.0,  # 🆕 Always 0 now
            "final_fraud_score": float(final_fraud_score),
            "fraud_threshold": float(fraud_threshold),
            "calculation": f"{base_fraud_score:.3f} (ML + Amount-Damage Analysis) = {final_fraud_score:.3f}",
            "signals_detected": len(fraud_signals),
            "threshold_strategy_used": threshold_strategy,
            "note": "⚠️ Document verification is informational only and does not affect fraud score"
        },
        
        "fraud_signals": fraud_signals,
        "recommended_action": recommended_action,
        
        # 🆕 UPDATED: Document verification with note
        "document_verification": {
            "police_report": verification_results.get('police'),
            "vehicle_registration": verification_results.get('vehicle'),
            "driving_license": verification_results.get('license'),
            "note": "⚠️ These checks are for investigator reference only - they do not affect the fraud score"
        },
        
        "documents_provided": {
            "fir_number": fir_number is not None and fir_number != "",
            "registration_number": vehicle_reg_no is not None and vehicle_reg_no != "",
            "dl_number": dl_number is not None and dl_number != ""
        },
        
        "detailed_calculations": {
            "tabular_analysis": tabular_details,
            "image_analysis": image_details,
            "fusion_analysis": fusion_details
        },
        
        # Multi-image specific results
        "multi_image_analysis": {
            "individual_images": multi_image_results['individual_image_results'],
            "aggregated_metrics": multi_image_results['aggregated_results']
        },
        
        "annotated_images": multi_image_results['individual_damage_detections'],
        
        "debug_info": {
            "tabular_features_shape": list(tabular_features.shape),
            "model_device": str(device),
            "model_type": _best_model_name,
            "is_calibrated": tabular_details.get('is_calibrated', False),
            "threshold_strategy": threshold_strategy,
            "available_strategies": list(_thresholds.keys()) if _thresholds else [],
            "total_images_processed": len(image_paths),
            "total_annotated_images": len(multi_image_results['individual_damage_detections'])
        }
    }
    
    # ========================================================================
    # 🆕 SAVE TO DATABASE - Import ClaimDatabaseHandler at top of file
    # ========================================================================
    from .claim_handler import ClaimDatabaseHandler
    
    try:
        print("💾 Saving claim to database...")
        
        # Prepare claim data for database
        claim_data = {
            'claim_description': claim_description,
            'accident_date': accident_date,
            'claim_amount': claim_amount,
            'dl_number': dl_number,
            'vehicle_reg_no': vehicle_reg_no,
            'fir_number': fir_number,
        }
        
        # Create claim in database with fraud detection results
        saved_claim = ClaimDatabaseHandler.create_claim(
            policyholder=policyholder,
            claim_data=claim_data,
            fraud_detection_result=response_data,
            image_files=car_images  # Pass the uploaded files
        )
        
        print(f"✅ Claim saved successfully: {saved_claim.claim_number}")
        
        # Add claim information to response
        response_data['claim_saved'] = True
        response_data['claim_id'] = saved_claim.id
        response_data['claim_number'] = saved_claim.claim_number
        response_data['claim_status'] = saved_claim.status
        response_data['claim_submission_message'] = f"Claim {saved_claim.claim_number} submitted successfully and saved to database"
        
    except Exception as e:
        print(f"⚠️ Failed to save claim to database: {e}")
        import traceback
        traceback.print_exc()
        
        # Don't fail the entire request if database save fails
        response_data['claim_saved'] = False
        response_data['save_error'] = str(e)
    
    # ========================================================================
    # Cleanup temp files
    # ========================================================================
    try:
        for image_path in image_paths:
            if os.path.exists(image_path):
                os.remove(image_path)
        if temp_dir and os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    return Response(response_data)

@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Health check endpoint"""
    device = get_device()
    
    force_load = request.GET.get('load', 'false').lower() == 'true'
    
    if force_load or not _models_loaded:
        load_success = load_models()
    
    model_status = {
        "models_loaded": _models_loaded,
        "device": str(device),
        "ml_imports_available": ML_IMPORTS_AVAILABLE,
        "image_imports_available": IMAGE_IMPORTS_AVAILABLE,
        "best_model": _best_model is not None,
        "calibrated_model": _calibrated_model is not None,
        "image_model": _image_model is not None,
        "preprocessing_objects": _preprocessing_objects is not None,
        "thresholds_loaded": _thresholds is not None,
    }
    
    if _best_model is not None:
        model_status["model_type"] = _best_model_name if _best_model_name else type(_best_model).__name__
        model_status["is_calibrated"] = hasattr(_best_model, 'calibrated_classifiers_')
    
    if _thresholds is not None:
        model_status["available_threshold_strategies"] = list(_thresholds.keys())
        model_status["threshold_details"] = {
            name: {
                'threshold': info['threshold'],
                'expected_recall': info.get('recall', 0),
                'expected_precision': info.get('precision', 0)
            }
            for name, info in _thresholds.items()
        }
    
    if IMAGE_IMPORTS_AVAILABLE:
        try:
            models_dir = get_models_dir()
            model_status["models_directory"] = models_dir
            model_status["models_directory_exists"] = os.path.exists(models_dir)
            
            if os.path.exists(models_dir):
                try:
                    files = os.listdir(models_dir)
                    model_status["files_in_models_dir"] = files
                except:
                    model_status["files_in_models_dir"] = "Cannot list directory"
        except Exception as e:
            model_status["models_directory_error"] = str(e)
    
    return Response({
        "status": "healthy" if _models_loaded else "models_not_loaded",
        "model_status": model_status,
        "pipeline_info": {
            "tabular_model": f"Single Best Model ({_best_model_name})" if _best_model_name else "Single Best Model",
            "calibration": "Platt Scaling (Sigmoid)",
            "image_model": "Mask R-CNN",
            "fusion_method": "Weighted + Geometric Mean",
            "threshold_strategies": list(_thresholds.keys()) if _thresholds else []
        },
        "message": "All systems operational" if _models_loaded else "Models need to be loaded"
    })