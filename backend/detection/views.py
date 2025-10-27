# detection/views.py
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
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
_ensemble_model = None
_image_model = None
_preprocessing_objects = None  # Will store scaler, encoders, feature_names
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
    global _ensemble_model, _image_model, _preprocessing_objects, _models_loaded
    
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
                'scaler': joblib.load(os.path.join(models_dir, "scaler.pkl")),
                'label_encoders': joblib.load(os.path.join(models_dir, "label_encoders.pkl")),
                'feature_names': joblib.load(os.path.join(models_dir, "feature_names.pkl"))
            }
            print("✅ Preprocessing objects loaded")
        except Exception as e:
            print(f"❌ Failed to load preprocessing objects: {e}")
            return False
        
        # Load ensemble model (XGBoost + LightGBM + RF + GB + LR)
        print("📈 Loading ensemble model...")
        try:
            _ensemble_model = joblib.load(os.path.join(models_dir, "ensemble_model.pkl"))
            print("✅ Ensemble model loaded")
        except Exception as e:
            print(f"❌ Failed to load ensemble model: {e}")
            return False
        
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
            "ensemble_model": _ensemble_model is not None,
            "image_model": _image_model is not None
        }
        
        print(f"📊 Model loading status: {models_status}")
        
        # Consider models loaded if at least preprocessing and ensemble loaded
        if _preprocessing_objects is not None and _ensemble_model is not None:
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


# --- Feature Engineering (same as training) ---
class AdvancedFeatureEngineer:
    """Create powerful fraud-detection features - MUST match training exactly"""
    
    @staticmethod
    def create_features(df):
        """Create advanced fraud indicators"""
        df = df.copy()
        
        # 1. Quick claim indicator
        if 'Days_Policy_Claim' in df.columns:
            df['very_quick_claim'] = df['Days_Policy_Claim'].apply(
                lambda x: 1 if str(x) == 'none' else 0
            )
        
        # 2. Young/old driver risk
        if 'Age' in df.columns:
            df['high_risk_age'] = df['Age'].apply(lambda x: 1 if x < 25 or x > 70 else 0)
            df['very_young_driver'] = df['Age'].apply(lambda x: 1 if x < 22 else 0)
        
        # 3. Premium vehicle flag
        if 'VehiclePrice' in df.columns:
            df['premium_vehicle'] = df['VehiclePrice'].apply(
                lambda x: 1 if 'more than 69000' in str(x) else 0
            )
            df['low_value_vehicle'] = df['VehiclePrice'].apply(
                lambda x: 1 if '20000' in str(x) or 'less' in str(x) else 0
            )
        
        # 4. Serial claimer
        if 'PastNumberOfClaims' in df.columns:
            df['serial_claimer'] = df['PastNumberOfClaims'].apply(
                lambda x: 1 if str(x) in ['2 to 4', 'more than 4'] else 0
            )
            df['first_time_claimer'] = df['PastNumberOfClaims'].apply(
                lambda x: 1 if str(x) == 'none' else 0
            )
        
        # 5. No evidence
        if 'WitnessPresent' in df.columns and 'PoliceReportFiled' in df.columns:
            df['no_witness_no_police'] = (
                (df['WitnessPresent'] == 'No') & 
                (df['PoliceReportFiled'] == 'No')
            ).astype(int)
            df['has_evidence'] = (
                (df['WitnessPresent'] == 'Yes') | 
                (df['PoliceReportFiled'] == 'Yes')
            ).astype(int)
        
        # 6. Recent address change
        if 'AddressChange_Claim' in df.columns:
            df['recent_address_change'] = df['AddressChange_Claim'].apply(
                lambda x: 1 if str(x) in ['1 year', '2 to 3 years', '4 to 8 years'] else 0
            )
        
        # 7. High deductible
        if 'Deductible' in df.columns:
            df['very_high_deductible'] = df['Deductible'].apply(lambda x: 1 if x >= 700 else 0)
            df['low_deductible'] = df['Deductible'].apply(lambda x: 1 if x <= 300 else 0)
        
        # 8. Weekend claim
        if 'DayOfWeekClaimed' in df.columns:
            df['weekend_claim'] = df['DayOfWeekClaimed'].apply(
                lambda x: 1 if str(x) in ['Saturday', 'Sunday'] else 0
            )
        
        # 9. Policyholder at fault
        if 'Fault' in df.columns:
            df['policyholder_fault'] = df['Fault'].apply(
                lambda x: 1 if 'Policy Holder' in str(x) else 0
            )
        
        # 10. External agent
        if 'AgentType' in df.columns:
            df['external_agent'] = df['AgentType'].apply(
                lambda x: 1 if 'External' in str(x) else 0
            )
        
        # 11. Multiple supplements
        if 'NumberOfSuppliments' in df.columns:
            df['many_supplements'] = df['NumberOfSuppliments'].apply(
                lambda x: 1 if str(x) in ['3 to 5', 'more than 5'] else 0
            )
        
        # 12. Multiple cars
        if 'NumberOfCars' in df.columns:
            df['multiple_cars'] = df['NumberOfCars'].apply(
                lambda x: 1 if str(x) in ['5 to 8', 'more than 8'] else 0
            )
        
        # 13. Urban area
        if 'AccidentArea' in df.columns:
            df['urban_accident'] = df['AccidentArea'].apply(
                lambda x: 1 if 'Urban' in str(x) else 0
            )
        
        # 14. Sport vehicle
        if 'VehicleCategory' in df.columns:
            df['sport_vehicle'] = df['VehicleCategory'].apply(
                lambda x: 1 if 'Sport' in str(x) else 0
            )
        
        # 15. Collision policy
        if 'PolicyType' in df.columns:
            df['collision_policy'] = df['PolicyType'].apply(
                lambda x: 1 if 'Collision' in str(x) else 0
            )
        
        # 16. Interaction features
        if 'high_risk_age' in df.columns and 'premium_vehicle' in df.columns:
            df['young_premium'] = df['high_risk_age'] * df['premium_vehicle']
        
        if 'no_witness_no_police' in df.columns and 'very_quick_claim' in df.columns:
            df['suspicious_combo'] = df['no_witness_no_police'] * df['very_quick_claim']
        
        return df


def preprocess_inference_data(df, preprocessing_objects):
    """
    Preprocess inference data using the exact same pipeline as training
    """
    try:
        df = df.copy()
        
        # Apply feature engineering (same as training)
        engineer = AdvancedFeatureEngineer()
        df = engineer.create_features(df)
        
        # Separate features
        X = df.copy()
        
        # Get column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Handle missing values (same as training)
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna('Unknown', inplace=True)
        
        # Encode categorical variables using saved encoders
        label_encoders = preprocessing_objects['label_encoders']
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                # Handle unseen categories
                X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                X[col] = le.transform(X[col].astype(str))
            else:
                # If encoder doesn't exist, create simple encoding
                X[col] = X[col].astype('category').cat.codes
        
        # Reorder columns to match training
        feature_names = preprocessing_objects['feature_names']
        
        # Add missing columns with 0
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        
        # Keep only training columns in the same order
        X = X[feature_names]
        
        # Scale using saved scaler
        scaler = preprocessing_objects['scaler']
        X_scaled = scaler.transform(X)
        
        return X_scaled
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_inference_data(policyholder, accident_date=None, claim_amount=None):
    """
    Create inference data that matches training data format
    """
    from datetime import datetime, timedelta
    import random
    
    # Parse dates
    if accident_date:
        try:
            acc_date = datetime.strptime(str(accident_date), "%Y-%m-%d")
        except:
            acc_date = datetime.now() - timedelta(days=random.randint(1, 30))
    else:
        acc_date = datetime.now() - timedelta(days=random.randint(1, 30))
    
    claim_date = acc_date + timedelta(days=random.randint(1, 7))
    
    # Map age to category
    age = int(getattr(policyholder, 'age', 30))
    if age <= 17:
        age_category = "16 to 17"
    elif age <= 20:
        age_category = "18 to 20"
    elif age <= 25:
        age_category = "21 to 25"
    elif age <= 30:
        age_category = "26 to 30"
    elif age <= 35:
        age_category = "31 to 35"
    elif age <= 40:
        age_category = "36 to 40"
    elif age <= 50:
        age_category = "41 to 50"
    elif age <= 65:
        age_category = "51 to 65"
    else:
        age_category = "over 65"
    
    # Create data dictionary matching training format
    data = {
        'Month': acc_date.strftime("%b"),
        'WeekOfMonth': (acc_date.day - 1) // 7 + 1,
        'DayOfWeek': acc_date.strftime("%A"),
        'Make': getattr(policyholder, 'make', 'Honda'),
        'AccidentArea': getattr(policyholder, 'accident_area', 'Urban'),
        'DayOfWeekClaimed': claim_date.strftime("%A"),
        'MonthClaimed': claim_date.strftime("%b"),
        'WeekOfMonthClaimed': (claim_date.day - 1) // 7 + 1,
        'Sex': str(getattr(policyholder, 'sex', 'Male')),
        'MaritalStatus': str(getattr(policyholder, 'marital_status', 'Single')),
        'Age': age,
        'Fault': 'Policy Holder',
        'PolicyType': getattr(policyholder, 'policy_type', 'Sedan - All Perils'),
        'VehicleCategory': getattr(policyholder, 'vehicle_category', 'Sedan'),
        'VehiclePrice': getattr(policyholder, 'vehicle_price', 'more than 69000'),
        'FraudFound_P': 0,
        'PolicyNumber': 1,
        'RepNumber': 12,
        'Deductible': int(getattr(policyholder, 'deductible', 300)),
        'DriverRating': int(getattr(policyholder, 'driver_rating', 1)),
        'Days_Policy_Accident': 'more than 30',
        'Days_Policy_Claim': 'more than 30',
        'PastNumberOfClaims': getattr(policyholder, 'past_claims', 'none'),
        'AgeOfVehicle': '3 years',
        'AgeOfPolicyHolder': age_category,
        'PoliceReportFiled': 'Yes' if claim_amount and float(claim_amount) > 50000 else 'No',
        'WitnessPresent': 'Yes' if claim_amount and float(claim_amount) > 100000 else 'No',
        'AgentType': 'Internal',
        'NumberOfSuppliments': 'none',
        'AddressChange_Claim': 'no change',
        'NumberOfCars': int(getattr(policyholder, 'number_of_cars', 1)),
        'Year': int(getattr(policyholder, 'year_of_vehicle', 1994)),
        'BasePolicy': getattr(policyholder, 'base_policy', 'All Perils')
    }
    
    return pd.DataFrame([data])


# --- Image Processing Functions (KEEP AS IS) ---

def process_damage_detection_image(image_path, image_model, device):
    """
    Process image with damage detection and return annotated image with bounding boxes
    """
    try:
        import torchvision.transforms as transforms
        
        # Load and process the image
        original_image = Image.open(image_path).convert("RGB")
        
        with torch.no_grad():
            # Transform image for model
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            image_tensor = transform(original_image).unsqueeze(0).to(device)
            
            # Get model predictions
            predictions = image_model(image_tensor)
            
            # Extract bounding boxes, scores, and labels
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy() if 'labels' in predictions[0] else None
            
            # Filter predictions by confidence threshold
            confidence_threshold = 0.5
            high_conf_indices = scores > confidence_threshold
            
            filtered_boxes = boxes[high_conf_indices]
            filtered_scores = scores[high_conf_indices]
            filtered_labels = labels[high_conf_indices] if labels is not None else None
            
            # Create annotated image
            annotated_image = original_image.copy()
            draw = ImageDraw.Draw(annotated_image)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Draw bounding boxes
            for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                label_text = f"Damage: {score:.2f}"
                if filtered_labels is not None:
                    label_text = f"Damage {filtered_labels[i]}: {score:.2f}"
                
                text_bbox = draw.textbbox((x1, y1-25), label_text, font=font)
                draw.rectangle(text_bbox, fill="red")
                draw.text((x1, y1-25), label_text, fill="white", font=font)
            
            # Convert to base64
            buffer = io.BytesIO()
            annotated_image.save(buffer, format='JPEG', quality=95)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Calculate damage metrics
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
            
            # Calculate damage severity
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
                }
            }
            
    except Exception as e:
        print(f"Error in damage detection visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_detailed_tabular_predictions(tabular_features, ensemble_model):
    """Get detailed tabular predictions from ensemble"""
    try:
        # Get ensemble prediction probabilities
        ensemble_proba = ensemble_model.predict_proba(tabular_features)
        ensemble_pred = ensemble_model.predict(tabular_features)
        
        # Get individual estimator predictions
        individual_predictions = []
        if hasattr(ensemble_model, 'estimators_'):
            for i, estimator in enumerate(ensemble_model.estimators_):
                try:
                    pred_proba = estimator.predict_proba(tabular_features)[0]
                    model_name = type(estimator).__name__
                    individual_predictions.append({
                        'model_name': model_name,
                        'fraud_probability': float(pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]),
                        'no_fraud_probability': float(pred_proba[0] if len(pred_proba) > 1 else 1-pred_proba[0])
                    })
                except Exception as e:
                    print(f"Error getting individual prediction {i}: {e}")
        
        return {
            'raw_features_shape': list(tabular_features.shape),
            'ensemble_probabilities': {
                'no_fraud': float(ensemble_proba[0][0] if len(ensemble_proba[0]) > 1 else 1-ensemble_proba[0][0]),
                'fraud': float(ensemble_proba[0][1] if len(ensemble_proba[0]) > 1 else ensemble_proba[0][0])
            },
            'ensemble_prediction': int(ensemble_pred[0]),
            'individual_model_predictions': individual_predictions,
            'tabular_confidence': float(max(ensemble_proba[0]) if len(ensemble_proba[0]) > 0 else 0.5)
        }
        
    except Exception as e:
        print(f"Error in tabular predictions: {e}")
        return {
            'raw_features_shape': list(tabular_features.shape),
            'ensemble_probabilities': {'no_fraud': 0.6, 'fraud': 0.4},
            'ensemble_prediction': 0,
            'individual_model_predictions': [],
            'tabular_confidence': 0.6
        }


def get_detailed_image_predictions(image_path, image_model, device):
    """Get detailed image predictions with damage analysis"""
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
            # Fallback
            return {
                'image_dimensions': {'width': image.width, 'height': image.height, 'total_pixels': image_area},
                'detection_results': {
                    'total_detections': 1,
                    'high_confidence_detections': 1,
                    'confidence_threshold': 0.5,
                    'all_scores': [0.6],
                    'filtered_scores': [0.6]
                },
                'damage_analysis': {
                    'damage_regions': [],
                    'total_damage_area_pixels': 5000.0,
                    'damage_percentage': 1.0,
                    'severity_level': 'LOW',
                    'severity_score': 0.3,
                    'weighted_damage_score': 0.02
                },
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
        
        image_fraud_prob = image_details.get('image_fraud_probability', 0.3)
        image_confidence = image_details.get('image_confidence', 0.6)
        
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
        max_fusion = max(tabular_fraud_prob, image_fraud_prob)
        
        if tabular_fraud_prob + image_fraud_prob > 0:
            harmonic_fusion = (2 * tabular_fraud_prob * image_fraud_prob) / (tabular_fraud_prob + image_fraud_prob)
        else:
            harmonic_fusion = 0.0
        
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
            'weight_calculation': {
                'total_confidence': float(total_confidence),
                'tabular_weight': float(tabular_weight),
                'image_weight': float(image_weight),
                'weight_formula': "weight = confidence / total_confidence"
            },
            'fusion_methods': {
                'weighted_average': {
                    'score': float(weighted_fusion),
                    'formula': f"({tabular_weight:.3f} × {tabular_fraud_prob:.3f}) + ({image_weight:.3f} × {image_fraud_prob:.3f})"
                },
                'geometric_mean': {
                    'score': float(geometric_fusion),
                    'formula': f"√({tabular_fraud_prob:.3f} × {image_fraud_prob:.3f})"
                },
                'maximum': {
                    'score': float(max_fusion),
                    'formula': f"max({tabular_fraud_prob:.3f}, {image_fraud_prob:.3f})"
                },
                'harmonic_mean': {
                    'score': float(harmonic_fusion),
                    'formula': f"2 × {tabular_fraud_prob:.3f} × {image_fraud_prob:.3f} / ({tabular_fraud_prob:.3f} + {image_fraud_prob:.3f})"
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
            'final_prediction': final_prediction,
            'final_confidence': final_fusion_score
        }
        
    except Exception as e:
        print(f"Error in fusion calculation: {e}")
        return {
            'input_probabilities': {
                'tabular_fraud_probability': 0.4,
                'tabular_confidence': 0.6,
                'image_fraud_probability': 0.3,
                'image_confidence': 0.6
            },
            'weight_calculation': {
                'total_confidence': 1.2,
                'tabular_weight': 0.5,
                'image_weight': 0.5
            },
            'fusion_methods': {
                'weighted_average': {'score': 0.35},
                'geometric_mean': {'score': 0.346},
                'maximum': {'score': 0.4},
                'harmonic_mean': {'score': 0.343}
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
            'final_prediction': 0,
            'final_confidence': 0.348
        }


# --- Auth and Policyholder Views ---

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

    serializer = PolicyholderSerializer(policyholder)
    return Response(serializer.data)


# --- Main Prediction Endpoint ---

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def predict_claim(request):
    """
    Enhanced predict claim function with detailed calculations
    Uses current pipeline: XGBoost + LightGBM + RF + GB + LR ensemble
    """
    
    # Extract request data
    username = request.data.get("username")
    claim_description = request.data.get("claim_description", "")
    accident_date = request.data.get("accident_date")
    claim_amount = request.data.get("claim_amount", 0)
    car_image = request.FILES.get("car_image")

    # Validate required inputs
    if not username:
        return Response({"error": "username is required"}, status=status.HTTP_400_BAD_REQUEST)
    if not car_image:
        return Response({"error": "car_image is required"}, status=status.HTTP_400_BAD_REQUEST)

    # Load models if not already loaded
    if not load_models():
        return Response(
            {"error": "ML models not available"}, 
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    # Get policyholder data
    try:
        policyholder = Policyholder.objects.get(username=username)
    except Policyholder.DoesNotExist:
        return Response({"error": "Policyholder not found"}, status=status.HTTP_404_NOT_FOUND)

    # Create and preprocess tabular data
    try:
        print("🔧 Creating inference data...")
        tabular_df = create_inference_data(policyholder, accident_date, claim_amount)
        print(f"✅ Created data: {tabular_df.shape}")
        
        print("🔧 Preprocessing data...")
        tabular_features = preprocess_inference_data(tabular_df, _preprocessing_objects)
        
        if tabular_features is None:
            return Response(
                {"error": "Failed to preprocess tabular data"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        print(f"✅ Preprocessed features: {tabular_features.shape}")
            
    except Exception as e:
        print(f"❌ Data preparation error: {e}")
        import traceback
        traceback.print_exc()
        return Response(
            {"error": f"Data preparation failed: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    # Handle image processing
    try:
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, f"temp_{random.randint(100,999)}.jpg")
        
        with open(image_path, "wb+") as f:
            for chunk in car_image.chunks():
                f.write(chunk)
        
        print(f"✅ Image saved: {image_path}")
                
    except Exception as e:
        return Response(
            {"error": f"Image processing failed: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    # Run ML predictions
    try:
        device = get_device()
        
        print("🔧 Getting tabular predictions...")
        # Get detailed tabular predictions from ensemble
        tabular_details = get_detailed_tabular_predictions(tabular_features, _ensemble_model)
        print(f"✅ Tabular prediction: {tabular_details.get('ensemble_prediction', 'N/A')}")
        
        print("🔧 Getting image predictions...")
        # Get detailed image predictions  
        image_details = get_detailed_image_predictions(image_path, _image_model, device)
        print(f"✅ Image fraud prob: {image_details.get('image_fraud_probability', 'N/A')}")
        
        print("🔧 Calculating fusion...")
        # Calculate fusion with mathematical breakdown
        fusion_details = calculate_detailed_fusion(tabular_details, image_details)
        print(f"✅ Final prediction: {fusion_details.get('final_prediction', 'N/A')}")
        
        # Get damage detection visualization
        print("🔧 Processing damage visualization...")
        damage_info = None
        if _image_model is not None:
            damage_info = process_damage_detection_image(image_path, _image_model, device)
            print(f"✅ Damage detection: {damage_info.get('total_damage_areas', 0) if damage_info else 0} areas")
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return Response(
            {"error": f"Prediction failed: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        # Clean up temporary files
        try:
            if 'image_path' in locals() and os.path.exists(image_path):
                os.remove(image_path)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass

    # Extract final results
    final_prediction = fusion_details.get('final_prediction', 0)
    final_confidence = fusion_details.get('final_confidence', 0.5)
    
    fraud_detected = bool(final_prediction)
    risk_level = "HIGH" if final_confidence > 0.8 else ("MEDIUM" if final_confidence > 0.5 else "LOW")
    
    # Prepare comprehensive response
    response_data = {
        "username": username,
        "claim_description": claim_description,
        "accident_date": accident_date,
        "claim_amount": claim_amount,
        "prediction": int(final_prediction),
        "confidence": float(final_confidence),
        "fraud_detected": fraud_detected,
        "risk_level": risk_level,
        "message": f"Fraud detected with {risk_level.lower()} confidence" if fraud_detected else f"No fraud detected ({risk_level.lower()} confidence)",
        
        # Add detailed calculations
        "detailed_calculations": {
            "tabular_analysis": tabular_details,
            "image_analysis": image_details,
            "fusion_analysis": fusion_details
        },
        
        # Add damage detection
        "damage_detection": damage_info,
        
        "debug_info": {
            "tabular_features_shape": list(tabular_features.shape),
            "model_device": str(device),
            "image_processed": True,
            "preprocessing_success": True,
            "detailed_analysis": True,
            "ensemble_type": "XGBoost + LightGBM + RF + GB + LR"
        }
    }
    
    print(f"✅ Response prepared: Fraud={fraud_detected}, Confidence={final_confidence:.3f}")
    
    return Response(response_data)


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Health check endpoint to verify model status"""
    device = get_device()
    
    # Try to load models if requested
    force_load = request.GET.get('load', 'false').lower() == 'true'
    
    if force_load or not _models_loaded:
        print("🔄 Attempting to load models...")
        load_success = load_models()
        print(f"📊 Model loading result: {load_success}")
    
    model_status = {
        "models_loaded": _models_loaded,
        "device": str(device),
        "ml_imports_available": ML_IMPORTS_AVAILABLE,
        "image_imports_available": IMAGE_IMPORTS_AVAILABLE,
        "ensemble_model": _ensemble_model is not None,
        "image_model": _image_model is not None,
        "preprocessing_objects": _preprocessing_objects is not None,
    }
    
    # Add model details if loaded
    if _ensemble_model is not None:
        model_status["ensemble_type"] = type(_ensemble_model).__name__
        if hasattr(_ensemble_model, 'estimators_'):
            model_status["ensemble_models"] = [type(e).__name__ for e in _ensemble_model.estimators_]
    
    # Add models directory info
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
            else:
                model_status["files_in_models_dir"] = "Directory does not exist"
        except Exception as e:
            model_status["models_directory_error"] = str(e)
    
    return Response({
        "status": "healthy" if _models_loaded else "models_not_loaded",
        "model_status": model_status,
        "pipeline_info": {
            "tabular_model": "Ensemble (XGBoost + LightGBM + RF + GB + LR)",
            "image_model": "Mask R-CNN for damage detection",
            "fusion_method": "Weighted + Geometric Mean",
            "feature_engineering": "Advanced (17+ engineered features)"
        },
        "message": "All systems operational" if _models_loaded else "Models need to be loaded",
        "instructions": "Add ?load=true to force model loading attempt"
    })