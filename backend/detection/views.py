# detection/views.py - UPDATED WITH YOLO INTEGRATION FOR PARTS + DAMAGE DETECTION

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
from decimal import Decimal
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework import generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.views import APIView

from django.contrib.auth.models import User
from django.db import connection

from .serializers import RegisterSerializer, PolicyholderSerializer
from .models import Policyholder


# Import ML libraries
try:
    import joblib
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    ML_IMPORTS_AVAILABLE = True
    print("✅ ML functions imported successfully")
except ImportError as e:
    ML_IMPORTS_AVAILABLE = False
    print(f"❌ ML import failed: {e}")

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO imported successfully")
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"❌ YOLO import failed: {e}")

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
_best_model = None
_calibrated_model = None
_image_model = None  # CNN for damage percentage
_yolo_parts_model = None  # NEW: YOLO for car parts
_yolo_damage_model = None  # NEW: YOLO for damage types
_preprocessing_objects = None
_thresholds = None
_best_model_name = None
_device = None
_models_loaded = False

# NEW: Part pricing database (to be replaced with actual DB queries)
# Add this mapping dictionary at the top of views.py

YOLO_TO_DB_PART_MAPPING = {
    # Bumpers
    'back bumper': 'rear bumper',
    'back-bumper': 'rear bumper',
    'backbumper': 'rear bumper',
    'front-bumper': 'front bumper',
    'frontbumper': 'front bumper',
    
    # Doors
    'back door': 'rear door',
    'back-door': 'rear door',
    'backdoor': 'rear door',
    'front-door': 'front door',
    'frontdoor': 'front door',
    
    # Lights
    'head light': 'headlight',
    'head-light': 'headlight',
    'tail light': 'taillight',
    'tail-light': 'taillight',
    'fog-light': 'fog light',
    'foglight': 'fog light',
    
    # Windows
    'wind shield': 'windshield',
    'wind-shield': 'windshield',
    'rear-windshield': 'rear windshield',
    'side-window': 'side window',
    
    # Mirrors
    'side-mirror': 'side mirror',
    'sidemirror': 'side mirror',
    'rear-view-mirror': 'rearview mirror',
    'rearview-mirror': 'rearview mirror',
    
    # Wheels
    'wheel-rim': 'wheel rim',
    'wheelrim': 'wheel rim',
    
    # Body parts
    'quarter-panel': 'quarter panel',
    'quarterpanel': 'quarter panel',
    'trunk-lid': 'trunk lid',
    'trunkLid': 'trunk lid',
    'side-skirt': 'side skirt',
    'sideskirt': 'side skirt',
    
    # Other
    'shock-absorber': 'shock absorber',
    'turn-signal': 'turn signal',
    'brake-light': 'brake light',
    'wheel-cover': 'wheel cover',
}


def normalize_part_name_with_mapping(yolo_part_name):
    """
    Normalize YOLO part name and map to database equivalent
    
    Args:
        yolo_part_name: Part name from YOLO detection
        
    Returns:
        Database-compatible part name
    """
    if not yolo_part_name:
        return 'default'
    
    # Step 1: Basic normalization
    normalized = normalize_string(yolo_part_name)
    
    # Step 2: Check mapping dictionary
    if normalized in YOLO_TO_DB_PART_MAPPING:
        mapped_name = YOLO_TO_DB_PART_MAPPING[normalized]
        print(f"🔄 Mapped: '{yolo_part_name}' → '{mapped_name}'")
        return mapped_name
    
    # Step 3: Return normalized name if no mapping exists
    return normalized


# Update your get_part_price_from_db function:
def get_part_price_from_db(vehicle_make, vehicle_model, part_name):
    """
    Fetch part price from database catalog
    NOW WITH YOLO MAPPING SUPPORT
    """
    try:
        make_normalized = normalize_string(vehicle_make)
        model_normalized = normalize_string(vehicle_model)
        
        # NEW: Use mapping function
        part_normalized = normalize_part_name_with_mapping(part_name)
        
        print(f"🔍 Looking up: {make_normalized} / {model_normalized} / {part_normalized}")
        
        query = """
        SELECT 
            mk.make_name,
            m.model_name,
            p.part_name,
            pp.base_price,
            pp.labor_cost,
            pp.gst_percentage,
            (pp.base_price + pp.labor_cost) as subtotal,
            (pp.base_price + pp.labor_cost) * (1 + pp.gst_percentage/100) as total_price
        FROM parts_pricing pp
        JOIN vehicle_makes mk ON pp.make_id = mk.id
        JOIN vehicle_models m ON pp.model_id = m.id
        JOIN car_parts p ON pp.part_id = p.id
        WHERE pp.is_active = true
        AND mk.make_name_normalized = %s
        AND m.model_name_normalized = %s
        AND p.part_name_normalized = %s
        AND (pp.effective_to IS NULL OR pp.effective_to >= CURRENT_DATE)
        ORDER BY pp.effective_from DESC
        LIMIT 1
        """
        
        with connection.cursor() as cursor:
            cursor.execute(query, [make_normalized, model_normalized, part_normalized])
            row = cursor.fetchone()
            
            if row:
                print(f"✅ Found: {row[0]} {row[1]} - {row[2]} = ₹{row[7]:.2f}")
                return {
                    'make': row[0],
                    'model': row[1],
                    'part': row[2],
                    'base_price': float(row[3]),
                    'labor_cost': float(row[4]),
                    'gst_percentage': float(row[5]),
                    'subtotal': float(row[6]),
                    'total_price': float(row[7]),
                    'source': 'database',
                    'matched': True
                }
        
        # If not found, try defaults
        print(f"⚠️ No exact match, trying defaults...")
        return get_default_part_price(make_normalized, model_normalized, part_normalized)
        
    except Exception as e:
        logger.error(f"Database lookup failed: {e}")
        import traceback
        traceback.print_exc()
        return get_fallback_price(part_normalized)

def get_default_part_price(make_normalized, model_normalized, part_normalized):
    """
    Try to find default pricing when exact match fails
    Priority: 1) Same make, similar model  2) Same make, default part  3) Fallback
    """
    try:
        # Try: Same make, any model, same part
        query = """
        SELECT 
            AVG(pp.base_price) as avg_base_price,
            AVG(pp.labor_cost) as avg_labor_cost,
            MAX(pp.gst_percentage) as gst_percentage
        FROM parts_pricing pp
        JOIN vehicle_makes mk ON pp.make_id = mk.id
        JOIN car_parts p ON pp.part_id = p.id
        WHERE pp.is_active = true
        AND mk.make_name_normalized = %s
        AND p.part_name_normalized = %s
        """
        
        with connection.cursor() as cursor:
            cursor.execute(query, [make_normalized, part_normalized])
            row = cursor.fetchone()
            
            if row and row[0]:
                base_price = float(row[0])
                labor_cost = float(row[1]) if row[1] else 1000
                gst = float(row[2]) if row[2] else 18.0
                subtotal = base_price + labor_cost
                total = subtotal * (1 + gst/100)
                
                return {
                    'make': make_normalized,
                    'model': 'average',
                    'part': part_normalized,
                    'base_price': base_price,
                    'labor_cost': labor_cost,
                    'gst_percentage': gst,
                    'subtotal': subtotal,
                    'total_price': total,
                    'source': 'make_average',
                    'matched': False
                }
        
        # If still not found, use fallback
        return get_fallback_price(part_normalized)
        
    except Exception as e:
        logger.error(f"Default price lookup failed: {e}")
        return get_fallback_price(part_normalized)


def get_fallback_price(part_normalized):
    """
    Hardcoded fallback prices when database lookup fails completely
    """
    FALLBACK_PRICES = {
        'front bumper': 5000,
        'rear bumper': 4500,
        'hood': 8000,
        'front door': 6000,
        'rear door': 5500,
        'fender': 4000,
        'headlight': 3000,
        'taillight': 2500,
        'side mirror': 1500,
        'windshield': 7000,
        'rear windshield': 5000,
        'side window': 2000,
        'fog light': 1200,
        'grille': 1800,
        'wheel rim': 3000,
        'tire': 3500,
        'trunk lid': 5500,
        'quarter panel': 6500,
        'default': 3000
    }
    
    base_price = FALLBACK_PRICES.get(part_normalized, FALLBACK_PRICES['default'])
    labor_cost = base_price * 0.3  # 30% of base price
    gst = 18.0
    subtotal = base_price + labor_cost
    total = subtotal * (1 + gst/100)
    
    return {
        'make': 'unknown',
        'model': 'unknown',
        'part': part_normalized,
        'base_price': float(base_price),
        'labor_cost': float(labor_cost),
        'gst_percentage': gst,
        'subtotal': float(subtotal),
        'total_price': float(total),
        'source': 'fallback',
        'matched': False
    }


def get_damage_severity_multiplier_from_db(damage_type):
    """
    Fetch damage severity multiplier from database
    """
    try:
        damage_normalized = normalize_string(damage_type)
        
        query = """
        SELECT cost_multiplier, severity_level, description
        FROM damage_types
        WHERE is_active = true
        AND damage_name_normalized = %s
        LIMIT 1
        """
        
        with connection.cursor() as cursor:
            cursor.execute(query, [damage_normalized])
            row = cursor.fetchone()
            
            if row:
                return {
                    'multiplier': float(row[0]),
                    'severity_level': row[1],
                    'description': row[2],
                    'damage_type': damage_type,
                    'source': 'database',
                    'matched': True
                }
        
        # If exact match not found, try partial match
        query_partial = """
        SELECT cost_multiplier, severity_level, description, damage_name
        FROM damage_types
        WHERE is_active = true
        AND damage_name_normalized LIKE %s
        ORDER BY cost_multiplier DESC
        LIMIT 1
        """
        
        with connection.cursor() as cursor:
            cursor.execute(query_partial, [f'%{damage_normalized}%'])
            row = cursor.fetchone()
            
            if row:
                return {
                    'multiplier': float(row[0]),
                    'severity_level': row[1],
                    'description': row[2],
                    'damage_type': row[3],
                    'source': 'database_partial',
                    'matched': False
                }
        
        # Fallback
        return get_fallback_damage_multiplier(damage_normalized)
        
    except Exception as e:
        logger.error(f"Database damage lookup failed: {e}")
        return get_fallback_damage_multiplier(damage_normalized)


def get_fallback_damage_multiplier(damage_normalized):
    """
    Hardcoded fallback damage multipliers
    """
    FALLBACK_MULTIPLIERS = {
        'scratch': 0.20,
        'minor scratch': 0.15,
        'deep scratch': 0.35,
        'dent': 0.40,
        'minor dent': 0.25,
        'major dent': 0.60,
        'crack': 0.60,
        'broken': 1.00,
        'shattered': 1.00,
        'bent': 0.70,
        'damaged': 0.50,
        'smashed': 1.00,
        'torn': 0.45,
        'chipped': 0.15,
        'missing': 1.00,
        'default': 0.50
    }
    
    # Try to find closest match
    for key, value in FALLBACK_MULTIPLIERS.items():
        if key in damage_normalized or damage_normalized in key:
            return {
                'multiplier': value,
                'severity_level': 'MEDIUM',
                'description': f'Fallback for {damage_normalized}',
                'damage_type': key,
                'source': 'fallback',
                'matched': False
            }
    
    return {
        'multiplier': FALLBACK_MULTIPLIERS['default'],
        'severity_level': 'MEDIUM',
        'description': 'Default fallback multiplier',
        'damage_type': 'unknown',
        'source': 'fallback',
        'matched': False
    }


def get_vehicle_info(make, model):
    """
    Get vehicle information from database
    """
    try:
        make_normalized = normalize_string(make)
        model_normalized = normalize_string(model)
        
        query = """
        SELECT 
            mk.make_name,
            m.model_name,
            m.body_type,
            m.year_from,
            m.year_to
        FROM vehicle_models m
        JOIN vehicle_makes mk ON m.make_id = mk.id
        WHERE mk.make_name_normalized = %s
        AND m.model_name_normalized = %s
        AND m.is_active = true
        LIMIT 1
        """
        
        with connection.cursor() as cursor:
            cursor.execute(query, [make_normalized, model_normalized])
            row = cursor.fetchone()
            
            if row:
                return {
                    'make': row[0],
                    'model': row[1],
                    'body_type': row[2],
                    'year_from': row[3],
                    'year_to': row[4],
                    'found': True
                }
        
        return {
            'make': make,
            'model': model,
            'body_type': 'Unknown',
            'year_from': None,
            'year_to': None,
            'found': False
        }
        
    except Exception as e:
        logger.error(f"Vehicle info lookup failed: {e}")
        return {
            'make': make,
            'model': model,
            'found': False,
            'error': str(e)
        }


def get_all_parts_for_vehicle(make, model):
    """
    Get all available parts and their prices for a specific vehicle
    """
    try:
        make_normalized = normalize_string(make)
        model_normalized = normalize_string(model)
        
        query = """
        SELECT 
            p.part_name,
            p.part_category,
            pp.base_price,
            pp.labor_cost,
            pp.gst_percentage,
            (pp.base_price + pp.labor_cost) * (1 + pp.gst_percentage/100) as total_price
        FROM parts_pricing pp
        JOIN vehicle_makes mk ON pp.make_id = mk.id
        JOIN vehicle_models m ON pp.model_id = m.id
        JOIN car_parts p ON pp.part_id = p.id
        WHERE pp.is_active = true
        AND mk.make_name_normalized = %s
        AND m.model_name_normalized = %s
        ORDER BY p.part_category, p.part_name
        """
        
        with connection.cursor() as cursor:
            cursor.execute(query, [make_normalized, model_normalized])
            rows = cursor.fetchall()
            
            parts_list = []
            for row in rows:
                parts_list.append({
                    'part_name': row[0],
                    'category': row[1],
                    'base_price': float(row[2]),
                    'labor_cost': float(row[3]),
                    'gst_percentage': float(row[4]),
                    'total_price': float(row[5])
                })
            
            return {
                'make': make,
                'model': model,
                'total_parts': len(parts_list),
                'parts': parts_list
            }
        
    except Exception as e:
        logger.error(f"Parts listing failed: {e}")
        return {
            'make': make,
            'model': model,
            'total_parts': 0,
            'parts': [],
            'error': str(e)
        }


def get_device():
    """Get PyTorch device"""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_models():
    """Load all models including YOLO"""
    global _best_model, _calibrated_model, _image_model, _yolo_parts_model, _yolo_damage_model
    global _preprocessing_objects, _thresholds, _best_model_name, _models_loaded
    
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
        
        # Load tabular model
        print("📈 Loading best model...")
        try:
            _best_model = joblib.load(os.path.join(models_dir, "final_best_model.pkl"))
            print("✅ Best model loaded")
            
            if hasattr(_best_model, 'calibrated_classifiers_'):
                _calibrated_model = _best_model
                print("✅ Model is calibrated (Platt scaling)")
            else:
                _calibrated_model = _best_model
                print("ℹ️  Model is not calibrated")
            
        except Exception as e:
            print(f"❌ Failed to load best model: {e}")
            return False
        
        # Load optimal thresholds
        print("🎯 Loading optimal thresholds...")
        try:
            _thresholds = joblib.load(os.path.join(models_dir, "final_thresholds.pkl"))
            print(f"✅ Loaded {len(_thresholds)} threshold strategies")
        except Exception as e:
            print(f"⚠️  Failed to load thresholds: {e}")
            _thresholds = {
                'default': {
                    'threshold': 0.5,
                    'precision': 0.15,
                    'recall': 0.75,
                    'f1': 0.20
                }
            }
        
        # Load CNN image model (for damage percentage)
        if IMAGE_IMPORTS_AVAILABLE:
            print("🖼️ Loading CNN image model...")
            _image_model = load_image_model("maskrcnn_damage_detection.pth", device)
            if _image_model is not None:
                print("✅ CNN image model loaded")
            else:
                print("⚠️ CNN image model failed to load")
        
        # NEW: Load YOLO models
        if YOLO_AVAILABLE:
            print("🚗 Loading YOLO car parts model...")
            try:
                parts_model_path = os.path.join(models_dir, "yolov8_car_parts.pt")
                if os.path.exists(parts_model_path):
                    _yolo_parts_model = YOLO(parts_model_path)
                    print("✅ YOLO parts model loaded")
                else:
                    print(f"⚠️ YOLO parts model not found at {parts_model_path}")
            except Exception as e:
                print(f"❌ Failed to load YOLO parts model: {e}")
            
            print("🔨 Loading YOLO damage model...")
            try:
                damage_model_path = os.path.join(models_dir, "yolov8_car_damage.pt")
                if os.path.exists(damage_model_path):
                    _yolo_damage_model = YOLO(damage_model_path)
                    print("✅ YOLO damage model loaded")
                else:
                    print(f"⚠️ YOLO damage model not found at {damage_model_path}")
            except Exception as e:
                print(f"❌ Failed to load YOLO damage model: {e}")
        
        # Check what loaded successfully
        models_status = {
            "preprocessing_objects": _preprocessing_objects is not None,
            "best_model": _best_model is not None,
            "calibrated_model": _calibrated_model is not None,
            "thresholds": _thresholds is not None,
            "cnn_image_model": _image_model is not None,
            "yolo_parts_model": _yolo_parts_model is not None,
            "yolo_damage_model": _yolo_damage_model is not None
        }
        
        print(f"📊 Model loading status: {models_status}")
        
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


# --- KEEP ALL EXISTING FEATURE ENGINEERING AND PREPROCESSING FUNCTIONS ---
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
        
        engineer = AdvancedFeatureEngineer()
        df = engineer.create_features(df)
        
        X = df.copy()
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna('Unknown', inplace=True)
        
        label_encoders = preprocessing_objects['label_encoders']
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                X[col] = le.transform(X[col].astype(str))
            else:
                X[col] = X[col].astype('category').cat.codes
        
        feature_names = preprocessing_objects['feature_names']
        
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        
        X = X[feature_names]
        
        scaler = preprocessing_objects['scaler']
        X_scaled = scaler.transform(X)
        
        return X_scaled
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_inference_data(policyholder, claim_amount=None, *args, **kwargs):
    """Builds a single-row pandas DataFrame for fraud prediction inference"""
    import pandas as pd

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

    for unwanted in ["CreatedAt", "UpdatedAt", "created_at", "updated_at", "createdAt", "updatedAt", "Email", "Username"]:
        if unwanted in data:
            del data[unwanted]

    df = pd.DataFrame([data])
    print(f"✅ Inference data prepared with {df.shape[1]} features.")
    return df


# KEEP EXISTING TABULAR PREDICTION FUNCTION
def get_detailed_tabular_predictions(tabular_features, best_model, thresholds, strategy='max_f1'):
    """Get detailed predictions from the single best model"""
    try:
        if hasattr(best_model, 'predict_proba'):
            proba = best_model.predict_proba(tabular_features)
            fraud_probability = float(proba[0][1] if len(proba[0]) > 1 else proba[0][0])
            no_fraud_probability = float(proba[0][0] if len(proba[0]) > 1 else 1 - proba[0][0])
        else:
            pred = best_model.predict(tabular_features)
            fraud_probability = float(pred[0])
            no_fraud_probability = 1.0 - fraud_probability
        
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
        
        selected_strategy = thresholds.get(strategy, thresholds.get('max_f1', {'threshold': 0.5}))
        selected_threshold = selected_strategy['threshold']
        primary_prediction = 1 if fraud_probability >= selected_threshold else 0
        
        distance_from_threshold = abs(fraud_probability - selected_threshold)
        confidence = min(0.5 + distance_from_threshold, 0.99)
        
        return {
            'raw_features_shape': list(tabular_features.shape),
            'model_type': _best_model_name if _best_model_name else type(best_model).__name__,
            'is_calibrated': hasattr(best_model, 'calibrated_classifiers_'),
            'probabilities': {
                'no_fraud': float(no_fraud_probability),
                'fraud': float(fraud_probability)
            },
            'primary_prediction': {
                'strategy': strategy,
                'threshold': float(selected_threshold),
                'prediction': int(primary_prediction),
                'confidence': float(confidence),
                'fraud_detected': bool(primary_prediction == 1)
            },
            'threshold_strategies': threshold_predictions,
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


# ============================================================================
# NEW: YOLO INTEGRATION FUNCTIONS
# ============================================================================

def normalize_string(text):
    """
    Normalize text for database matching
    Handles YOLO's hyphenated format (e.g., 'front-bumper' → 'front bumper')
    """
    if not text:
        return 'default'
    
    # Remove extra whitespace, convert to lowercase
    normalized = str(text).lower().strip()
    
    # Replace hyphens with spaces (CRITICAL for YOLO compatibility)
    normalized = normalized.replace('-', ' ')
    
    # Replace underscores with spaces
    normalized = normalized.replace('_', ' ')
    
    # Replace multiple spaces with single space
    import re
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove special characters except spaces
    normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
    
    return normalized


# Test cases to verify it works:
if __name__ == "__main__":
    test_cases = [
        ('front-bumper', 'front bumper'),
        ('Front-Bumper', 'front bumper'),
        ('front_bumper', 'front bumper'),
        ('FRONT BUMPER', 'front bumper'),
        ('rear-door', 'rear door'),
        ('side-mirror', 'side mirror'),
        ('headlight', 'headlight'),
        ('fog-light', 'fog light'),
    ]
    
    print("Testing normalize_string():")
    print("-" * 50)
    for input_text, expected in test_cases:
        result = normalize_string(input_text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_text}' → '{result}' (expected: '{expected}')")

def get_part_price(vehicle_make, vehicle_model, part_name):
    """
    UPDATED VERSION - Fetches from database instead of hardcoded dict
    Compatible with existing code - returns just the total price as float
    """
    price_data = get_part_price_from_db(vehicle_make, vehicle_model, part_name)
    return price_data['total_price']


def get_damage_severity_multiplier(damage_type):
    """
    UPDATED VERSION - Fetches from database instead of hardcoded dict
    Compatible with existing code - returns just the multiplier as float
    """
    damage_data = get_damage_severity_multiplier_from_db(damage_type)
    return damage_data['multiplier']



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
        weighted_fusion = (tabular_weight * tabular_fraud_prob) + (cnn_weight * cnn_fraud_prob)
        geometric_fusion = np.sqrt(max(tabular_fraud_prob * cnn_fraud_prob, 0))
        base_fusion_score = (0.65 * weighted_fusion) + (0.35 * geometric_fusion)
        
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
                "cnn_weight": float(cnn_weight)
            },
            "fusion_methods": {
                "weighted_average": {
                    "score": float(weighted_fusion),
                    "formula": f"({tabular_weight:.3f} × {tabular_fraud_prob:.3f}) + ({cnn_weight:.3f} × {cnn_fraud_prob:.3f})"
                },
                "geometric_mean": {
                    "score": float(geometric_fusion)
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
                "base_fusion": float(base_fusion_score),
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
        
        return {
            "error": str(e),
            "input_probabilities": {
                "tabular_fraud_probability": 0.4,
                "tabular_confidence": 0.6,
                "cnn_fraud_probability": 0.3,
                "cnn_confidence": 0.7
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

##################

def deduplicate_detections(yolo_results_all_images, confidence_threshold=0.5):
    """
    Deduplicate YOLO detections to prevent charging for the same part multiple times
    
    Strategy:
    1. Group by image and part
    2. For each part in an image, keep only the HIGHEST severity damage
    3. Filter out low confidence detections
    """
    
    deduplicated_results = []
    deduplication_stats = {
        'total_detections': 0,
        'after_deduplication': 0,
        'removed_duplicates': 0,
        'removed_low_confidence': 0,
        'deduplication_enabled': True
    }
    
    for img_idx, yolo_result in enumerate(yolo_results_all_images):
        if not yolo_result.get('detection_successful', False):
            deduplicated_results.append(yolo_result)
            continue
        
        assignments = yolo_result.get('assignments', [])
        deduplication_stats['total_detections'] += len(assignments)
        
        # Filter by confidence first
        high_confidence_assignments = [
            a for a in assignments 
            if a.get('damage_confidence', 0) >= confidence_threshold
        ]
        
        removed_low_conf = len(assignments) - len(high_confidence_assignments)
        deduplication_stats['removed_low_confidence'] += removed_low_conf
        
        if removed_low_conf > 0:
            print(f"Image #{img_idx + 1}: Removed {removed_low_conf} low-confidence detections")
        
        # Group by part name
        part_damages = defaultdict(list)
        
        for assignment in high_confidence_assignments:
            part_name = assignment.get('assigned_part', 'UNASSIGNED')
            if part_name:  # Only process assigned parts
                part_damages[part_name].append(assignment)
        
        # For each part, keep only the highest severity damage
        deduplicated_assignments = []
        
        for part_name, damages in part_damages.items():
            if len(damages) == 1:
                # Only one detection for this part - keep it
                deduplicated_assignments.append(damages[0])
            else:
                # Multiple detections - need to deduplicate
                print(f"Image #{img_idx + 1}, {part_name}: Found {len(damages)} detections, deduplicating...")
                
                # Strategy: Keep the damage with highest severity multiplier
                best_damage = max(damages, key=lambda d: (
                    get_severity_priority(d.get('damage_type', '')),
                    d.get('damage_confidence', 0)
                ))
                
                deduplicated_assignments.append(best_damage)
                
                # Count removed damages
                removed_damages = [d for d in damages if d != best_damage]
                deduplication_stats['removed_duplicates'] += len(removed_damages)
        
        # Create deduplicated result
        deduplicated_result = yolo_result.copy()
        deduplicated_result['assignments'] = deduplicated_assignments
        deduplicated_result['original_assignment_count'] = len(assignments)
        deduplicated_result['deduplicated_assignment_count'] = len(deduplicated_assignments)
        deduplicated_result['deduplication_applied'] = True
        
        deduplicated_results.append(deduplicated_result)
        deduplication_stats['after_deduplication'] += len(deduplicated_assignments)
    
    print(f"✅ Deduplication complete: {deduplication_stats}")
    
    return deduplicated_results, deduplication_stats

def get_severity_priority(damage_type):
    """
    Return priority score for damage types (higher = more severe)
    Used to determine which damage to keep when multiple detected on same part
    """
    damage_lower = str(damage_type).lower()
    
    # Critical damages (100% cost)
    if any(word in damage_lower for word in ['broken', 'shattered', 'missing', 'smashed']):
        return 100
    
    # High severity (70-80% cost)
    if any(word in damage_lower for word in ['bent', 'major', 'severe']):
        return 70
    
    # Medium severity (40-60% cost)
    if any(word in damage_lower for word in ['crack', 'dent', 'damaged', 'chip']):
        return 50
    
    # Low severity (15-35% cost)
    if any(word in damage_lower for word in ['scratch', 'minor', 'corrosion']):
        return 20
    
    # Default
    return 40


def group_damages_by_part(yolo_results_all_images):
    """
    Group all damages by part across all images
    Useful for reporting and validation
    
    Returns:
        dict: {part_name: [list of damages across all images]}
    """
    part_damages = defaultdict(list)
    
    for img_idx, yolo_result in enumerate(yolo_results_all_images):
        if not yolo_result.get('detection_successful', False):
            continue
        
        for assignment in yolo_result.get('assignments', []):
            part_name = assignment.get('assigned_part', 'UNASSIGNED')
            if part_name and part_name != 'UNASSIGNED':
                part_damages[part_name].append({
                    'image_index': img_idx + 1,
                    'damage_type': assignment.get('damage_type', 'unknown'),
                    'confidence': assignment.get('damage_confidence', 0),
                    'assignment': assignment
                })
    
    return dict(part_damages)


def validate_claim_breakdown(breakdown):
    """
    Validate claim breakdown to detect suspicious patterns
    """
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'suspicious_patterns': [],
        'statistics': {}
    }
    
    if not breakdown:
        return validation_result
    
    # Group by image and part
    image_parts = defaultdict(lambda: defaultdict(list))
    
    for item in breakdown:
        img_idx = item['image_index']
        part = item['part']
        image_parts[img_idx][part].append(item)
    
    # Check for duplicates
    for img_idx, parts in image_parts.items():
        for part, damages in parts.items():
            if len(damages) > 1:
                validation_result['is_valid'] = False
                validation_result['warnings'].append(
                    f"Image #{img_idx}: Part '{part}' charged {len(damages)} times"
                )
    
    # Statistics
    validation_result['statistics'] = {
        'total_items': len(breakdown),
        'total_cost': sum(item.get('damage_cost', 0) for item in breakdown),
        'avg_confidence': sum(item.get('confidence', 0) for item in breakdown) / len(breakdown) if breakdown else 0,
        'unique_parts': len(set(item.get('part', 'unknown') for item in breakdown)),
        'low_confidence_count': sum(1 for item in breakdown if item.get('confidence', 0) < 0.5)
    }
    
    return validation_result
##################

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

# ============================================================================
# UPDATED MAIN PREDICTION ENDPOINT
# ============================================================================

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def predict_claim(request):
    """
    Enhanced predict claim with YOLO + Deduplication + CNN + XGBoost Fusion
    """
    
    username = request.data.get("username")
    claim_description = request.data.get("claim_description", "")
    accident_date = request.data.get("accident_date")

    car_images = request.FILES.getlist("car_images")
    car_image_single = request.FILES.get("car_image")
    
    if car_image_single and car_image_single not in car_images:
        car_images.insert(0, car_image_single)
    
    threshold_strategy = request.data.get("threshold_strategy", "max_f1")
    
    # Deduplication settings
    enable_deduplication = request.data.get("enable_deduplication", "true").lower() == "true"
    confidence_threshold = float(request.data.get("confidence_threshold", 0.5))
    
    # Document verification fields
    fir_number = request.data.get("fir_number", "")
    dl_number = request.data.get("dl_number", "")
    vehicle_reg_no = request.data.get("vehicle_reg_no", "")

    if not username or not car_images:
        return Response({
            "error": "username and at least one car_image are required"
        }, status=status.HTTP_400_BAD_REQUEST)

    if not load_models():
        return Response({
            "error": "ML models not available"
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    try:
        policyholder = Policyholder.objects.get(username=username)
    except Policyholder.DoesNotExist:
        return Response({
            "error": "Policyholder not found"
        }, status=status.HTTP_404_NOT_FOUND)

    # ========================================================================
    # GET VEHICLE INFO: Prioritize form data, fallback to policyholder profile
    # ========================================================================
    vehicle_make_from_form = request.data.get("vehicle_make", "").strip()
    vehicle_model_from_form = request.data.get("vehicle_model", "").strip()

    if vehicle_make_from_form and vehicle_model_from_form:
        vehicle_make = vehicle_make_from_form
        vehicle_model = vehicle_model_from_form
        print(f"✅ Using vehicle from form: {vehicle_make} {vehicle_model}")
    else:
        vehicle_make = getattr(policyholder, 'vehicle_make', 'default')
        vehicle_model = getattr(policyholder, 'vehicle_model', 'default')
        print(f"📋 Using vehicle from profile: {vehicle_make} {vehicle_model}")

    # Warn if using defaults
    if vehicle_make == 'default' or vehicle_model == 'default':
        print(f"⚠️  WARNING: Using default vehicle - pricing may be inaccurate!")
    
    print(f"🚗 Final Vehicle for Pricing: {vehicle_make} {vehicle_model}")

    # Save images temporarily
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
            "error": f"Image processing failed: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # ========================================================================
    # STEP 1: Process images with YOLO + CNN
    # ========================================================================
    try:
        device = get_device()
        
        print("🚀 Running YOLO + CNN pipeline...")
        multi_model_results = process_multiple_images_with_yolo(
            image_paths,
            _yolo_parts_model,
            _yolo_damage_model,
            _image_model,
            device
        )
        
        yolo_results_all = multi_model_results['yolo_results_all_images']
        cnn_aggregated = multi_model_results['cnn_aggregated_results']
        cnn_damage_viz = multi_model_results['cnn_damage_visualizations']
        
        # Get average CNN damage percentage
        avg_cnn_damage = cnn_aggregated.get('damage_summary', {}).get('avg_damage_percentage', 0)
        
        print(f"✅ YOLO: Processed {len(yolo_results_all)} images")
        print(f"✅ CNN: Average damage {avg_cnn_damage:.2f}%")
        
        # Log raw detection counts before deduplication
        total_raw_detections = sum(
            len(r.get('assignments', [])) 
            for r in yolo_results_all 
            if r.get('detection_successful', False)
        )
        print(f"📊 Raw detections: {total_raw_detections}")
        
    except Exception as e:
        print(f"❌ YOLO/CNN processing failed: {e}")
        import traceback
        traceback.print_exc()
        
        return Response({
            "error": f"Image analysis failed: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    # ========================================================================
    # STEP 2: Calculate claim amount with DEDUPLICATION
    # ========================================================================
    try:
        print(f"💰 Calculating claim amount for {vehicle_make} {vehicle_model}...")
        print(f"⚙️  Deduplication: {'ENABLED' if enable_deduplication else 'DISABLED'}")
        print(f"⚙️  Confidence threshold: {confidence_threshold}")
        
        # Call the deduplication function (defined in views.py)
        yolo_claim_calc = calculate_claim_amount_from_yolo_deduplicated(
            yolo_results_all,
            vehicle_make,
            vehicle_model,
            avg_cnn_damage,
            enable_deduplication=enable_deduplication,
            confidence_threshold=confidence_threshold
        )
        
        calculated_claim_amount = yolo_claim_calc['final_calculated_amount']
        
        print(f"✅ Calculated claim amount: ₹{calculated_claim_amount:,.2f}")
        
        # Log deduplication results
        if enable_deduplication:
            dedup_stats = yolo_claim_calc.get('deduplication_stats', {})
            print(f"🔍 Deduplication results:")
            print(f"   Before: {dedup_stats.get('total_detections', 0)} detections")
            print(f"   After: {dedup_stats.get('after_deduplication', 0)} detections")
            print(f"   Removed duplicates: {dedup_stats.get('removed_duplicates', 0)}")
            print(f"   Removed low confidence: {dedup_stats.get('removed_low_confidence', 0)}")
        
        # Check validation warnings
        validation = yolo_claim_calc.get('validation', {})
        if not validation.get('is_valid', True):
            print(f"⚠️  Validation warnings: {validation.get('warnings', [])}")
        
    except Exception as e:
        print(f"❌ Claim calculation failed: {e}")
        import traceback
        traceback.print_exc()
        
        calculated_claim_amount = 30000  # Fallback
        yolo_claim_calc = {
            'final_calculated_amount': calculated_claim_amount,
            'error': str(e),
            'deduplication_stats': {}
        }
    
    # ========================================================================
    # STEP 3: Run tabular fraud detection with calculated amount
    # ========================================================================
    try:
        tabular_df = create_inference_data(policyholder, calculated_claim_amount)
        tabular_features = preprocess_inference_data(tabular_df, _preprocessing_objects)
        
        if tabular_features is None:
            raise ValueError("Failed to preprocess data")
        
        tabular_details = get_detailed_tabular_predictions(
            tabular_features, 
            _calibrated_model,
            _thresholds,
            threshold_strategy
        )
        
        print("✅ Tabular fraud detection complete")
        
    except Exception as e:
        print(f"❌ Tabular fraud detection failed: {e}")
        import traceback
        traceback.print_exc()
        
        return Response({
            "error": f"Fraud detection failed: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    # ========================================================================
    # STEP 4: Fusion analysis
    # ========================================================================
    try:
        fusion_details = calculate_detailed_fusion_with_yolo(
            tabular_details,
            cnn_aggregated,
            yolo_claim_calc,
            dl_number=dl_number,
            expiry_date=None,
            reg_no=vehicle_reg_no,
            make=vehicle_make,
            year=getattr(policyholder, 'year_of_vehicle', None),
            fir_no=fir_number
        )
        
        print("✅ Fusion analysis complete")
        
    except Exception as e:
        print(f"❌ Fusion failed: {e}")
        import traceback
        traceback.print_exc()
        
        fusion_details = {
            'final_prediction': 0,
            'final_confidence': 0.5,
            'error': str(e)
        }
    
    # ========================================================================
    # STEP 5: Build response with deduplication info
    # ========================================================================
    final_fraud_score = fusion_details.get('final_confidence', 0.5)
    fraud_detected = fusion_details.get('final_prediction', 0) == 1
    
    fraud_threshold = config('FRAUD_THRESHOLD', default=0.5, cast=float)
    high_threshold = config('HIGH_RISK_THRESHOLD', default=0.7, cast=float)
    critical_threshold = config('CRITICAL_RISK_THRESHOLD', default=0.85, cast=float)
    
    if final_fraud_score >= critical_threshold:
        risk_level = "CRITICAL"
        recommended_action = {
            'action': 'REJECT',
            'message': 'Critical fraud indicators detected.',
            'next_steps': ['Reject claim', 'Initiate investigation']
        }
    elif final_fraud_score >= high_threshold:
        risk_level = "HIGH"
        recommended_action = {
            'action': 'INVESTIGATE',
            'message': 'High fraud risk detected.',
            'next_steps': ['Senior review required', 'Request additional docs']
        }
    elif final_fraud_score >= fraud_threshold:
        risk_level = "MEDIUM"
        recommended_action = {
            'action': 'REVIEW',
            'message': 'Moderate fraud risk.',
            'next_steps': ['Enhanced verification needed']
        }
    else:
        risk_level = "LOW"
        recommended_action = {
            'action': 'APPROVE',
            'message': 'Low fraud risk.',
            'next_steps': ['Process payment']
        }
    
    # Format breakdown for display (function defined in views.py)
    formatted_breakdown = format_claim_breakdown_for_display(
        yolo_claim_calc.get('detailed_breakdown', []),
        group_by_image=True
    )
    
    # Get deduplication summary (function defined in views.py)
    dedup_summary = get_deduplication_summary(
        yolo_claim_calc.get('deduplication_stats', {})
    )
    
    response_data = {
        "username": username,
        "claim_description": claim_description,
        "accident_date": accident_date,
        "total_images_submitted": len(image_paths),
        
        # Calculated claim amount
        "calculated_claim_amount": float(calculated_claim_amount),
        "claim_calculation_details": yolo_claim_calc,
        
        # Formatted breakdown for display
        "formatted_breakdown": formatted_breakdown,
        "deduplication_summary": dedup_summary,
        
        # Fraud detection results
        "prediction": int(fraud_detected),
        "confidence": float(final_fraud_score),
        "fraud_detected": fraud_detected,
        "probabilities": {
            "no_fraud": float(1 - final_fraud_score),
            "fraud": float(final_fraud_score)
        },
        "risk_level": risk_level,
        "message": f"{'🚨 FRAUD DETECTED' if fraud_detected else '✅ LEGITIMATE'} - {risk_level} risk",
        
        "fraud_analysis": {
            "ml_base_score": float(final_fraud_score),
            "fraud_threshold": float(fraud_threshold)
        },
        
        "recommended_action": recommended_action,
        
        # Detailed results
        "yolo_detection_results": {
            "all_images": yolo_results_all,
            "total_parts_detected": sum(len(r.get('parts_detected', [])) for r in yolo_results_all),
            "total_damages_detected": sum(len(r.get('damages_detected', [])) for r in yolo_results_all),
            "total_assignments": sum(len(r.get('assignments', [])) for r in yolo_results_all),
        },
        
        "annotated_images": cnn_damage_viz,
        
        "multi_image_analysis": {
            "individual_images": multi_model_results.get('cnn_results_all_images', []),
            "aggregated_metrics": cnn_aggregated
        },
        
        "detailed_calculations": {
            "tabular_analysis": tabular_details,
            "fusion_analysis": fusion_details
        },
        
        "debug_info": {
            "yolo_parts_model_loaded": _yolo_parts_model is not None,
            "yolo_damage_model_loaded": _yolo_damage_model is not None,
            "cnn_model_loaded": _image_model is not None,
            "device": str(device),
            "deduplication_applied": enable_deduplication,
            "validation_status": yolo_claim_calc.get('validation', {}).get('is_valid', True)
        }
    }
    
    # ========================================================================
    # STEP 6: Save to database
    # ========================================================================
    try:
        from .claim_handler import ClaimDatabaseHandler
        
        claim_data = {
            'claim_description': claim_description,
            'accident_date': accident_date,
            'claim_amount': calculated_claim_amount,
            'dl_number': dl_number,
            'vehicle_reg_no': vehicle_reg_no,
            'fir_number': fir_number,
        }
        
        saved_claim = ClaimDatabaseHandler.create_claim(
            policyholder=policyholder,
            claim_data=claim_data,
            fraud_detection_result=response_data,
            image_files=car_images
        )
        
        response_data['claim_saved'] = True
        response_data['claim_id'] = saved_claim.id
        response_data['claim_number'] = saved_claim.claim_number
        response_data['claim_status'] = saved_claim.status
        
        print(f"✅ Claim saved: {saved_claim.claim_number}")
        
    except Exception as e:
        print(f"⚠️ Database save failed: {e}")
        import traceback
        traceback.print_exc()
        response_data['claim_saved'] = False
        response_data['save_error'] = str(e)
    
    # Cleanup
    try:
        for image_path in image_paths:
            if os.path.exists(image_path):
                os.remove(image_path)
        if temp_dir and os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")
    
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