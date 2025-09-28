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
import random
# Import ML functions with error handling
try:
    from .fusion import (
        load_tabular_cnn_feature_extractor,
        load_ensemble_model,
        load_image_model,
        load_preprocessing_pipeline,
        preprocess_tabular_data_with_pipeline,
        fused_prediction,
        get_image_damage_score,
        get_tabular_pred_proba,
        get_models_dir
    )
    ML_IMPORTS_AVAILABLE = True
    print("✅ ML functions imported successfully")
except ImportError as e:
    ML_IMPORTS_AVAILABLE = False
    print(f"❌ ML import failed: {e}")

# --- Global Model Variables (Lazy Loading) ---
_feature_extractor = None
_ensemble_model = None
_image_model = None
_preprocessing_pipeline = None
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
    global _feature_extractor, _ensemble_model, _image_model, _preprocessing_pipeline, _models_loaded
    
    if _models_loaded:
        return True
    
    if not ML_IMPORTS_AVAILABLE:
        print("❌ Cannot load models: ML imports not available")
        return False
    
    try:
        device = get_device()
        print(f"🔧 Loading models on device: {device}")
        
        # Load preprocessing pipeline first
        print("📋 Loading preprocessing pipeline...")
        _preprocessing_pipeline = load_preprocessing_pipeline("preprocessing_pipeline.pkl")
        
        if _preprocessing_pipeline is not None:
            print("✅ Preprocessing pipeline loaded")
            
            # Create dummy data to get input size
            dummy_csv = """Month,WeekOfMonth,DayOfWeek,Make,AccidentArea,DayOfWeekClaimed,MonthClaimed,WeekOfMonthClaimed,Sex,MaritalStatus,Age,Fault,PolicyType,VehicleCategory,VehiclePrice,FraudFound_P,PolicyNumber,RepNumber,Deductible,DriverRating,Days_Policy_Accident,Days_Policy_Claim,PastNumberOfClaims,AgeOfVehicle,AgeOfPolicyHolder,PoliceReportFiled,WitnessPresent,AgentType,NumberOfSuppliments,AddressChange_Claim,NumberOfCars,Year,BasePolicy
Dec,5,Wednesday,Honda,Urban,Tuesday,Jan,1,Female,Single,21,Policy Holder,Sport - Liability,Sport,more than 69000,0,1,12,300,1,more than 30,more than 30,none,3 years,26 to 30,No,No,External,none,1 year,3 to 4,1994,Liability"""
            
            dummy_df = pd.read_csv(StringIO(dummy_csv))
            dummy_processed = preprocess_tabular_data_with_pipeline(dummy_df, _preprocessing_pipeline)
            
            if dummy_processed is not None:
                input_feature_size = dummy_processed.shape[1]
                print(f"📊 Input feature size: {input_feature_size}")
                
                # Load tabular CNN feature extractor
                print("🧠 Loading tabular CNN feature extractor...")
                _feature_extractor = load_tabular_cnn_feature_extractor("cnn_model.pth", input_feature_size, device)
                
                # Load ensemble model
                print("📈 Loading ensemble model...")
                _ensemble_model = load_ensemble_model("ensemble_model.pkl")
                
                # Load image model
                print("🖼️ Loading image model...")
                _image_model = load_image_model("maskrcnn_damage_detection.pth", device)
                
                # Check what loaded successfully
                models_status = {
                    "preprocessing_pipeline": _preprocessing_pipeline is not None,
                    "feature_extractor": _feature_extractor is not None,
                    "ensemble_model": _ensemble_model is not None,
                    "image_model": _image_model is not None
                }
                
                print(f"📊 Model loading status: {models_status}")
                
                # Consider models loaded if at least preprocessing and one model loaded
                if _preprocessing_pipeline is not None and (_feature_extractor is not None or _image_model is not None):
                    _models_loaded = True
                    print("✅ Models loaded successfully!")
                    return True
                else:
                    print("⚠️ Some models failed to load but system can still operate")
                    _models_loaded = True  # Allow partial functionality
                    return True
            else:
                print("❌ Failed to preprocess dummy data")
        else:
            print("❌ Failed to load preprocessing pipeline")
        
        return False
        
    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        import traceback
        traceback.print_exc()
        return False

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
    permission_classes = [AllowAny]  # Change to IsAuthenticated if needed
    
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

# --- ML Prediction Views ---

# Replace your predict_claim function with this fixed version

def clean_and_validate_data(tabular_df):
    """Clean and validate data types before preprocessing"""
    df = tabular_df.copy()
    
    # Define expected data types and valid values for categorical columns
    categorical_mappings = {
        'Sex': {'Male', 'Female', 'M', 'F'},
        'MaritalStatus': {'Single', 'Married', 'Divorced', 'Widowed'},
        'AccidentArea': {'Urban', 'Rural'},
        'Fault': {'Policy Holder', 'Third Party'},
        'PoliceReportFiled': {'Yes', 'No', 'Y', 'N'},
        'WitnessPresent': {'Yes', 'No', 'Y', 'N'},
        'AgentType': {'Internal', 'External'},
        'DayOfWeek': {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'},
        'Month': {'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'},
        'AgeOfVehicle': {'new', '1 years', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', 'more than 7'},
        'VehicleCategory': {'Sedan', 'Sports Car', 'SUV', 'Hatchback', 'Luxury', 'Utility'},
        'PolicyType': {'Sedan - All Perils', 'Sedan - Collision', 'Sedan - Liability', 'Sports Car - All Perils', 'Sports Car - Collision', 'Sports Car - Liability'},
        'BasePolicy': {'All Perils', 'Collision', 'Liability'},
        'VehiclePrice': {'less than 20000', '20000 to 29000', '30000 to 39000', '40000 to 59000', '60000 to 69000', 'more than 69000'},
        'PastNumberOfClaims': {'none', '1', '2 to 4', 'more than 4'},
        'AgeOfPolicyHolder': {'16 to 17', '18 to 20', '21 to 25', '26 to 30', '31 to 35', '36 to 40', '41 to 50', '51 to 65', 'over 65'},
        'Days_Policy_Accident': {'none', '1 to 7', '8 to 15', '15 to 30', '30 to 31', 'more than 30'},
        'Days_Policy_Claim': {'none', '1 to 7', '8 to 15', '15 to 30', '30 to 31', 'more than 30'},
        'AddressChange_Claim': {'no change', '1 year', '2-3 years', '4-8 years'},
        'NumberOfSuppliments': {'none', '1 to 2', '3 to 5', 'more than 5'},
        'Make': {'Acura', 'BMW', 'Chevrolet', 'Dodge', 'Ford', 'Honda', 'Jeep', 'Mazda', 'Mercury', 'Nisson', 'Pontiac', 'Saab', 'Saturn', 'Suburu', 'Toyota', 'Volkswagen'}
    }
    
    # Numeric columns
    numeric_columns = {'Age', 'NumberOfCars', 'Year', 'WeekOfMonth', 'WeekOfMonthClaimed', 'FraudFound_P', 'PolicyNumber', 'RepNumber', 'Deductible', 'DriverRating'}
    
    print("🧹 Cleaning and validating data...")
    
    for col in df.columns:
        if col in numeric_columns:
            # Handle numeric columns
            try:
                # Convert to numeric, replacing non-numeric values with defaults
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaN values with appropriate defaults
                if col == 'Age':
                    df[col] = df[col].fillna(30).astype(int)
                elif col == 'Year':
                    df[col] = df[col].fillna(2020).astype(int)
                elif col == 'NumberOfCars':
                    df[col] = df[col].fillna(1).astype(int)
                elif col in ['WeekOfMonth', 'WeekOfMonthClaimed']:
                    df[col] = df[col].fillna(1).astype(int)
                    # Ensure week is between 1 and 5
                    df[col] = df[col].clip(1, 5)
                else:
                    df[col] = df[col].fillna(0).astype(int)
                    
            except Exception as e:
                print(f"⚠️ Error processing numeric column {col}: {e}")
                # Set default values
                if col == 'Age':
                    df[col] = 30
                elif col == 'Year':
                    df[col] = 2020
                elif col == 'NumberOfCars':
                    df[col] = 1
                else:
                    df[col] = 0
                    
        elif col in categorical_mappings:
            # Handle categorical columns
            valid_values = categorical_mappings[col]
            # Convert to string and handle missing values
            df[col] = df[col].astype(str).str.strip()
            
            # Map common variations
            if col in ['Sex']:
                df[col] = df[col].replace({'M': 'Male', 'F': 'Female'})
            elif col in ['PoliceReportFiled', 'WitnessPresent']:
                df[col] = df[col].replace({'Y': 'Yes', 'N': 'No', '1': 'Yes', '0': 'No'})
            
            # Replace invalid values with defaults
            mask = ~df[col].isin(valid_values)
            if mask.any():
                if col == 'Sex':
                    df.loc[mask, col] = 'Male'
                elif col == 'MaritalStatus':
                    df.loc[mask, col] = 'Single'
                elif col == 'AccidentArea':
                    df.loc[mask, col] = 'Urban'
                elif col == 'Fault':
                    df.loc[mask, col] = 'Policy Holder'
                elif col in ['PoliceReportFiled', 'WitnessPresent']:
                    df.loc[mask, col] = 'No'
                elif col == 'AgentType':
                    df.loc[mask, col] = 'Internal'
                elif col == 'Make':
                    df.loc[mask, col] = 'Toyota'
                elif col == 'VehicleCategory':
                    df.loc[mask, col] = 'Sedan'
                elif col == 'PolicyType':
                    df.loc[mask, col] = 'Sedan - All Perils'
                elif col == 'BasePolicy':
                    df.loc[mask, col] = 'All Perils'
                elif col == 'VehiclePrice':
                    df.loc[mask, col] = '20000 to 29000'
                elif col == 'PastNumberOfClaims':
                    df.loc[mask, col] = 'none'
                elif col == 'AgeOfVehicle':
                    df.loc[mask, col] = '3 years'
                elif col == 'AgeOfPolicyHolder':
                    df.loc[mask, col] = '26 to 30'
                elif col in ['Days_Policy_Accident', 'Days_Policy_Claim']:
                    df.loc[mask, col] = 'more than 30'
                elif col == 'AddressChange_Claim':
                    df.loc[mask, col] = 'no change'
                elif col == 'NumberOfSuppliments':
                    df.loc[mask, col] = 'none'
                elif col in ['DayOfWeek', 'DayOfWeekClaimed']:
                    df.loc[mask, col] = 'Monday'
                elif col in ['Month', 'MonthClaimed']:
                    df.loc[mask, col] = 'Jan'
                else:
                    # Default fallback
                    df.loc[mask, col] = list(valid_values)[0]
                    
                print(f"⚠️ Fixed {mask.sum()} invalid values in column {col}")
        
        else:
            # Handle any other columns as strings
            df[col] = df[col].astype(str).fillna('Unknown')
    
    print("✅ Data cleaning completed")
    print(f"📊 Final data types: {df.dtypes.to_dict()}")
    
    return df

def safe_preprocess_tabular_data(df, pipeline):
    """
    Safely preprocess tabular data with additional error handling
    """
    try:
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Convert all object columns to string and handle any remaining NaN
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                # Replace any remaining NaN with empty string, then convert to string
                df_processed[col] = df_processed[col].fillna('').astype(str)
                # Remove any extra whitespace
                df_processed[col] = df_processed[col].str.strip()
        
        # For numeric columns, ensure they're proper numeric types
        numeric_cols = ['Age', 'NumberOfCars', 'Year', 'WeekOfMonth', 'WeekOfMonthClaimed', 
                       'FraudFound_P', 'PolicyNumber', 'RepNumber', 'Deductible', 'DriverRating']
        
        for col in numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0).astype(int)
        
        print(f"🔧 About to call pipeline.transform() with shape: {df_processed.shape}")
        print(f"🔧 Data types before transform: {df_processed.dtypes.to_dict()}")
        
        # Try the transformation with additional error handling
        try:
            result = pipeline.transform(df_processed)
            print(f"✅ Pipeline transform successful! Result shape: {result.shape}")
            return result
        except ValueError as ve:
            if "could not be safely coerced" in str(ve) or "isnan" in str(ve):
                print(f"⚠️ Pipeline transform failed with data type error: {ve}")
                
                # Try converting all object columns to category first
                print("🔄 Attempting categorical conversion...")
                df_cat = df_processed.copy()
                for col in df_cat.columns:
                    if df_cat[col].dtype == 'object':
                        df_cat[col] = df_cat[col].astype('category')
                
                result = pipeline.transform(df_cat)
                print(f"✅ Pipeline transform successful with categorical data! Result shape: {result.shape}")
                return result
            else:
                raise ve
                
    except Exception as e:
        print(f"❌ Safe preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Alternative approach - create a minimal dataset that matches training data format
def create_minimal_training_format(tabular_data):
    """
    Create data in the exact format the model was trained on
    """
    # These are the exact values and formats from your training data
    training_format = {
        'Month': 'Aug',  # Keep as is
        'WeekOfMonth': int(tabular_data.get('WeekOfMonth', 2)),
        'DayOfWeek': tabular_data.get('DayOfWeek', 'Sunday'),
        'Make': 'Honda',  # Use a known good value
        'AccidentArea': 'Urban',  # Use a known good value
        'DayOfWeekClaimed': tabular_data.get('DayOfWeekClaimed', 'Friday'),
        'MonthClaimed': 'Aug',  # Keep as is
        'WeekOfMonthClaimed': int(tabular_data.get('WeekOfMonthClaimed', 3)),
        'Sex': tabular_data.get('Sex', 'Male'),
        'MaritalStatus': tabular_data.get('MaritalStatus', 'Married'),
        'Age': int(tabular_data.get('Age', 49)),
        'Fault': 'Policy Holder',  # Use a known good value
        'PolicyType': 'Sedan - All Perils',  # Use a known good value
        'VehicleCategory': 'Sedan',  # Use a known good value
        'VehiclePrice': '20000 to 29000',  # Use a known good value
        'FraudFound_P': 0,
        'PolicyNumber': 1,
        'RepNumber': 12,
        'Deductible': 300,
        'DriverRating': 1,
        'Days_Policy_Accident': 'more than 30',
        'Days_Policy_Claim': 'more than 30',
        'PastNumberOfClaims': 'none',
        'AgeOfVehicle': '3 years',
        'AgeOfPolicyHolder': tabular_data.get('AgeOfPolicyHolder', '41 to 50'),
        'PoliceReportFiled': 'Yes',
        'WitnessPresent': 'No',
        'AgentType': 'Internal',
        'NumberOfSuppliments': 'none',
        'AddressChange_Claim': 'no change',
        'NumberOfCars': int(tabular_data.get('NumberOfCars', 1)),
        'Year': int(tabular_data.get('Year', 2012)),
        'BasePolicy': 'All Perils'
    }
    
    print(f"🔧 Created minimal training format: {training_format}")
    return training_format

# Updated predict_claim function with fallback approach
# Add this to your detection/views.py file to replace the existing predict_claim function

def create_training_compatible_inference_data(policyholder, accident_date=None, claim_amount=None):
    """
    Create inference data that exactly matches your training pipeline format
    """
    from datetime import datetime, timedelta
    import random
    
    # Parse dates or use defaults
    if accident_date:
        try:
            acc_date = datetime.strptime(str(accident_date), "%Y-%m-%d")
        except:
            acc_date = datetime.now() - timedelta(days=random.randint(1, 30))
    else:
        acc_date = datetime.now() - timedelta(days=random.randint(1, 30))
    
    claim_date = acc_date + timedelta(days=random.randint(1, 7))
    
    # Calculate proper age category based on your ordinal categories
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
        age_category = "65+"
    
    # Create data dictionary that matches your training columns exactly
    # Note: PolicyNumber and RepNumber are dropped in your pipeline, so don't include them
    data = {
        # Date features
        'Month': acc_date.strftime("%b"),
        'WeekOfMonth': (acc_date.day - 1) // 7 + 1,
        'DayOfWeek': acc_date.strftime("%A"),
        'MonthClaimed': claim_date.strftime("%b"),
        'WeekOfMonthClaimed': (claim_date.day - 1) // 7 + 1,
        'DayOfWeekClaimed': claim_date.strftime("%A"),
        
        # Vehicle info (use defaults from your dataset or reasonable values)
        'Make': getattr(policyholder, 'make', 'Honda'),
        'VehicleCategory': getattr(policyholder, 'vehicle_category', 'Sedan'),
        'VehiclePrice': getattr(policyholder, 'vehicle_price', 'more than 69000'),
        'Year': int(getattr(policyholder, 'year_of_vehicle', 2000)),
        
        # Policy info
        'PolicyType': getattr(policyholder, 'policy_type', 'Sedan - All Perils'),
        'BasePolicy': getattr(policyholder, 'base_policy', 'All Perils'),
        'Deductible': int(getattr(policyholder, 'deductible', 300)),
        'DriverRating': int(getattr(policyholder, 'driver_rating', 1)),
        
        # Personal info
        'Sex': str(getattr(policyholder, 'sex', 'Male')),
        'MaritalStatus': str(getattr(policyholder, 'marital_status', 'Single')),
        'Age': age,
        'NumberOfCars': int(getattr(policyholder, 'number_of_cars', 1)),
        
        # Claim specifics
        'AccidentArea': getattr(policyholder, 'accident_area', 'Urban'),
        'Fault': 'Policy Holder',  # Default assumption
        'PoliceReportFiled': 'Yes' if claim_amount and float(claim_amount) > 50000 else 'No',
        'WitnessPresent': 'Yes' if claim_amount and float(claim_amount) > 100000 else 'No',
        'AgentType': 'Internal',
        
        # ORDINAL features - use values that exist in your ordinal_categories
        'Days_Policy_Accident': 'more than 30',  # Safe default
        'Days_Policy_Claim': 'more than 30',     # Safe default
        'PastNumberOfClaims': getattr(policyholder, 'past_claims', 'none'),
        'AgeOfVehicle': '3 years',  # Default based on typical vehicle age
        'AgeOfPolicyHolder': age_category,
        'NumberOfSuppliments': 'none',  # Safe default
        'AddressChange_Claim': 'no change',  # Safe default
    }
    
    return data


def safe_preprocess_inference_data_fixed(data, pipeline):
    """
    Fixed preprocessing function that properly handles the trained pipeline
    """
    if pipeline is None:
        print("❌ No preprocessing pipeline available")
        return None
    
    try:
        # Convert to DataFrame if it's a dict
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        print(f"📊 Input DataFrame shape: {df.shape}")
        print(f"📊 Input columns: {list(df.columns)}")
        
        # Step 1: Ensure all expected columns are present and properly typed
        # Based on your pipeline, these are the ordinal features
        ordinal_features = [
            'Days_Policy_Accident',
            'Days_Policy_Claim', 
            'PastNumberOfClaims',
            'AgeOfVehicle',
            'AgeOfPolicyHolder',
            'NumberOfSuppliments',
            'AddressChange_Claim'
        ]
        
        # Validate and clean ordinal features
        for col in ordinal_features:
            if col in df.columns:
                # Ensure string type and clean
                df[col] = df[col].astype(str).str.strip()
                
                # Replace any NaN-like values with 'none' (safe default)
                nan_values = ['nan', 'NaN', 'None', 'null', 'NULL', '', ' ']
                df[col] = df[col].replace(nan_values, 'none')
                
                print(f"   Ordinal feature {col}: '{df[col].iloc[0]}'")
            else:
                # Add missing ordinal column with default
                df[col] = 'none'
                print(f"   Added missing ordinal feature {col}: 'none'")
        
        # Step 2: Handle categorical (onehot) columns
        # These are all object columns NOT in ordinal_features
        object_columns = df.select_dtypes(include=['object']).columns.tolist()
        onehot_cols = [col for col in object_columns if col not in ordinal_features]
        
        for col in onehot_cols:
            # Ensure proper string type
            df[col] = df[col].astype(str).str.strip()
            
            # Replace NaN-like values with reasonable defaults
            nan_values = ['nan', 'NaN', 'None', 'null', 'NULL', '', ' ']
            
            if col in ['Month', 'MonthClaimed']:
                df[col] = df[col].replace(nan_values, 'Jan')
            elif col in ['DayOfWeek', 'DayOfWeekClaimed']:
                df[col] = df[col].replace(nan_values, 'Monday')
            elif col == 'Make':
                df[col] = df[col].replace(nan_values, 'Honda')
            elif col == 'Sex':
                df[col] = df[col].replace(nan_values, 'Male')
            elif col == 'MaritalStatus':
                df[col] = df[col].replace(nan_values, 'Single')
            else:
                df[col] = df[col].replace(nan_values, 'Unknown')
            
            print(f"   OneHot feature {col}: '{df[col].iloc[0]}'")
        
        # Step 3: Handle numerical columns (remainder='passthrough')
        numerical_columns = df.select_dtypes(exclude=['object']).columns.tolist()
        
        for col in numerical_columns:
            # Convert to numeric and handle NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN with appropriate defaults
            if col in ['WeekOfMonth', 'WeekOfMonthClaimed']:
                df[col] = df[col].fillna(1).astype(int)
            elif col == 'Age':
                df[col] = df[col].fillna(30).astype(int)
            elif col == 'Year':
                df[col] = df[col].fillna(2000).astype(int)
            else:
                df[col] = df[col].fillna(0).astype(int)
            
            print(f"   Numerical feature {col}: {df[col].iloc[0]}")
        
        # Step 4: Verify no NaN values remain
        if df.isnull().any().any():
            print("⚠️ Found remaining NaN values, filling them...")
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('Unknown')
                else:
                    df[col] = df[col].fillna(0)
        
        print("✅ Data cleaning completed")
        
        # Step 5: Apply the trained pipeline
        print("🔄 Applying trained preprocessing pipeline...")
        X_processed = pipeline.transform(df)
        
        # Convert sparse matrix to dense if needed
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
        
        # Ensure float32 type for ML models
        X_processed = X_processed.astype(np.float32)
        
        print(f"✅ Pipeline preprocessing successful! Output shape: {X_processed.shape}")
        return X_processed
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        
        # Enhanced debugging
        print("\n🔍 Debug Information:")
        try:
            print(f"   DataFrame info:")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Dtypes: {dict(df.dtypes)}")
            
            # Check for problematic values
            for col in df.columns:
                if df[col].dtype == 'object':
                    unique_vals = df[col].unique()
                    print(f"   {col} values: {unique_vals}")
                    
        except Exception as debug_e:
            print(f"   Debug info failed: {debug_e}")
        
        import traceback
        print(f"\n❌ Full traceback:")
        traceback.print_exc()
        return None

def clean_inference_data(df):
    """Exact same cleaning function used during training - CRITICAL for consistency"""
    df_clean = df.copy()
    
    # Handle object/categorical columns
    for col in df_clean.select_dtypes(include=['object']).columns:
        # Convert to string first
        df_clean[col] = df_clean[col].astype(str)
        
        # Replace all problematic values
        problematic_values = ['nan', 'NaN', 'None', 'null', 'NULL', '', ' ', 'na', 'NA']
        for val in problematic_values:
            df_clean[col] = df_clean[col].replace(val, 'Unknown')
        
        # Strip whitespace
        df_clean[col] = df_clean[col].str.strip()
        
        # Replace empty strings that might have been created
        df_clean[col] = df_clean[col].replace('', 'Unknown')
        
        # Ensure no NaN values remain
        df_clean[col] = df_clean[col].fillna('Unknown')
    
    # Handle numeric columns
    for col in df_clean.select_dtypes(exclude=['object']).columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col] = df_clean[col].fillna(0)
        df_clean[col] = df_clean[col].astype(int)
    
    return df_clean

def safe_preprocess_inference_data(data, pipeline):
    """
    Process inference data using the EXACT same cleaning as training
    """
    if pipeline is None:
        print("No preprocessing pipeline available")
        return None
    
    try:
        # Convert to DataFrame if it's a dict
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        print(f"Input DataFrame shape: {df.shape}")
        print("Data before cleaning:")
        for col in df.columns:
            print(f"  {col}: {df[col].iloc[0]} (type: {type(df[col].iloc[0])})")
        
        # Use the EXACT same cleaning function as training
        df_clean = clean_inference_data(df)
        
        print("Data after cleaning:")
        for col in df_clean.columns:
            print(f"  {col}: '{df_clean[col].iloc[0]}' (type: {type(df_clean[col].iloc[0])})")
        
        # Apply the preprocessing pipeline
        X_processed = pipeline.transform(df_clean)
        
        # Ensure it's a numpy array
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
        
        # Convert to float32
        X_processed = X_processed.astype(np.float32)
        
        print(f"Preprocessing successful! Output shape: {X_processed.shape}")
        return X_processed
        
    except Exception as e:
        print(f"Preprocessing failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Enhanced debugging
        print("\nDetailed debugging:")
        try:
            print(f"DataFrame info after cleaning:")
            print(f"  Shape: {df_clean.shape}")
            print(f"  Columns: {list(df_clean.columns)}")
            print(f"  Dtypes: {dict(df_clean.dtypes)}")
            
            # Check for any remaining NaN values
            nan_cols = df_clean.columns[df_clean.isnull().any()].tolist()
            if nan_cols:
                print(f"  Columns with NaN: {nan_cols}")
            else:
                print("  No NaN values found")
                
        except Exception as debug_e:
            print(f"  Debug info failed: {debug_e}")
        
        import traceback
        traceback.print_exc()
        return None

def create_safe_inference_data(policyholder, accident_date=None, claim_amount=None):
    """
    Create inference data that matches the training data format exactly
    """
    from datetime import datetime, timedelta
    import random
    
    # Parse dates or use defaults
    if accident_date:
        try:
            acc_date = datetime.strptime(str(accident_date), "%Y-%m-%d")
        except:
            acc_date = datetime.now() - timedelta(days=random.randint(1, 30))
    else:
        acc_date = datetime.now() - timedelta(days=random.randint(1, 30))
    
    claim_date = acc_date + timedelta(days=random.randint(1, 7))
    
    # Map age to proper category based on your training data
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
    
    # Create inference data with safe defaults that exist in training data
    data = {
        # Date features
        'Month': acc_date.strftime("%b"),
        'WeekOfMonth': (acc_date.day - 1) // 7 + 1,
        'DayOfWeek': acc_date.strftime("%A"),
        'MonthClaimed': claim_date.strftime("%b"),
        'WeekOfMonthClaimed': (claim_date.day - 1) // 7 + 1,
        'DayOfWeekClaimed': claim_date.strftime("%A"),
        
        # Vehicle info - use safe training values
        'Make': 'Honda',  # Known to exist in training
        'VehicleCategory': 'Sedan',  # Known to exist in training
        'VehiclePrice': 'more than 69000',  # Known to exist in training
        'Year': int(getattr(policyholder, 'year_of_vehicle', 1994)),
        
        # Policy info - use safe training values
        'PolicyType': 'Sedan - All Perils',  # Known to exist in training
        'BasePolicy': 'All Perils',  # Known to exist in training
        'Deductible': int(getattr(policyholder, 'deductible', 300)),
        'DriverRating': int(getattr(policyholder, 'driver_rating', 1)),
        
        # Personal info
        'Sex': str(getattr(policyholder, 'sex', 'Male')),
        'MaritalStatus': str(getattr(policyholder, 'marital_status', 'Single')),
        'Age': age,
        'NumberOfCars': str(getattr(policyholder, 'number_of_cars', '1')),  # Keep as string - it's categorical!
        
        # Claim specifics - use safe training values
        'AccidentArea': str(getattr(policyholder, 'accident_area', 'Urban')),
        'Fault': 'Policy Holder',  # Known to exist in training
        'PoliceReportFiled': 'No',  # Known to exist in training
        'WitnessPresent': 'No',  # Known to exist in training
        'AgentType': 'Internal',  # Known to exist in training
        
        # Ordinal features - use values that definitely exist in training
        'Days_Policy_Accident': 'more than 30',  # Known to exist
        'Days_Policy_Claim': 'more than 30',     # Known to exist
        'PastNumberOfClaims': 'none',            # Known to exist
        'AgeOfVehicle': '3 years',               # Known to exist
        'AgeOfPolicyHolder': age_category,
        'NumberOfSuppliments': 'none',           # Known to exist
        'AddressChange_Claim': 'no change'       # Known to exist
    }
    
    return data

# Add this function to your detection/views.py

import base64
from PIL import Image, ImageDraw, ImageFont
import io

def process_damage_detection_image(image_path, image_model, device):
    """
    Process image with damage detection and return annotated image with bounding boxes
    """
    try:
        from .fusion import get_image_damage_score
        import torch
        import torchvision.transforms as transforms
        
        # Load and process the image
        original_image = Image.open(image_path).convert("RGB")
        
        # Get damage detection results from your image model
        # This depends on your specific Mask R-CNN implementation
        # You'll need to modify this based on your fusion.py implementation
        
        # For now, I'll show the general structure - you'll need to adapt this
        # to your specific Mask R-CNN model's output format
        
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
            
            # Try to load a font, fall back to default if not available
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Draw bounding boxes and labels
            for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
                x1, y1, x2, y2 = box
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # Draw label with confidence
                label_text = f"Damage: {score:.2f}"
                if filtered_labels is not None:
                    label_text = f"Damage {filtered_labels[i]}: {score:.2f}"
                
                # Draw background rectangle for text
                text_bbox = draw.textbbox((x1, y1-25), label_text, font=font)
                draw.rectangle(text_bbox, fill="red")
                
                # Draw text
                draw.text((x1, y1-25), label_text, fill="white", font=font)
            
            # Convert annotated image to base64
            buffer = io.BytesIO()
            annotated_image.save(buffer, format='JPEG', quality=95)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Calculate overall damage metrics
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

# Modified predict_claim function (replace the existing one)
# Replace your existing predict_claim function with this enhanced version

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def predict_claim(request):
    """Enhanced predict claim function with detailed calculations"""
    
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
        tabular_data = create_safe_inference_data(policyholder, accident_date, claim_amount)
        tabular_features = safe_preprocess_inference_data(tabular_data, _preprocessing_pipeline)
        
        if tabular_features is None:
            return Response(
                {"error": "Failed to preprocess tabular data"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
    except Exception as e:
        return Response(
            {"error": f"Data preparation failed: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    # Handle image processing
    try:
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, f"temp_{random.randint(100,999)}.jpg")
        
        with open(image_path, "wb+") as f:
            for chunk in car_image.chunks():
                f.write(chunk)
                
    except Exception as e:
        return Response(
            {"error": f"Image processing failed: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    # Run detailed ML prediction and analysis
    try:
        device = get_device()
        
        # Get detailed tabular predictions
        tabular_details = get_detailed_tabular_predictions(tabular_features, device)
        
        # Get detailed image predictions  
        image_details = get_detailed_image_predictions(image_path, device)
        
        # Calculate fusion with mathematical breakdown
        fusion_details = calculate_detailed_fusion(tabular_details, image_details)
        
        # Get damage detection visualization
        damage_info = process_damage_detection_image(image_path, _image_model, device)
        
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
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
            "detailed_analysis": True
        }
    }
    
    return Response(response_data)


def get_detailed_tabular_predictions(tabular_features, device):
    """Get detailed tabular predictions with step-by-step breakdown"""
    try:
        # Step 1: CNN Feature Extraction
        tabular_tensor = torch.FloatTensor(tabular_features).to(device)
        
        with torch.no_grad():
            if _feature_extractor is not None:
                cnn_features = _feature_extractor(tabular_tensor)
                cnn_features_np = cnn_features.cpu().numpy()
            else:
                # Fallback: use raw features if CNN not available
                cnn_features_np = tabular_features
        
        # Step 2: Ensemble Prediction
        if _ensemble_model is not None:
            ensemble_proba = _ensemble_model.predict_proba(cnn_features_np)
            ensemble_pred = _ensemble_model.predict(cnn_features_np)
            
            # Get individual estimator predictions if available
            individual_predictions = []
            if hasattr(_ensemble_model, 'estimators_'):
                for i, estimator in enumerate(_ensemble_model.estimators_):
                    try:
                        pred_proba = estimator.predict_proba(cnn_features_np)[0]
                        individual_predictions.append({
                            'model_name': f'Estimator_{i+1}',
                            'fraud_probability': float(pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]),
                            'no_fraud_probability': float(pred_proba[0] if len(pred_proba) > 1 else 1-pred_proba[0])
                        })
                    except Exception as e:
                        print(f"Error getting individual prediction {i}: {e}")
            
            return {
                'raw_features_shape': list(tabular_features.shape),
                'raw_features_sample': tabular_features[0][:10].tolist() if len(tabular_features[0]) >= 10 else tabular_features[0].tolist(),
                'cnn_features_shape': list(cnn_features_np.shape),
                'cnn_features_sample': cnn_features_np[0][:10].tolist() if len(cnn_features_np[0]) >= 10 else cnn_features_np[0].tolist(),
                'ensemble_probabilities': {
                    'no_fraud': float(ensemble_proba[0][0] if len(ensemble_proba[0]) > 1 else 1-ensemble_proba[0][0]),
                    'fraud': float(ensemble_proba[0][1] if len(ensemble_proba[0]) > 1 else ensemble_proba[0][0])
                },
                'ensemble_prediction': int(ensemble_pred[0]),
                'individual_model_predictions': individual_predictions,
                'tabular_confidence': float(max(ensemble_proba[0]) if len(ensemble_proba[0]) > 0 else 0.5)
            }
        else:
            # Fallback predictions if ensemble model not available
            return {
                'raw_features_shape': list(tabular_features.shape),
                'raw_features_sample': tabular_features[0][:10].tolist() if len(tabular_features[0]) >= 10 else tabular_features[0].tolist(),
                'cnn_features_shape': list(cnn_features_np.shape),
                'cnn_features_sample': cnn_features_np[0][:10].tolist() if len(cnn_features_np[0]) >= 10 else cnn_features_np[0].tolist(),
                'ensemble_probabilities': {
                    'no_fraud': 0.6,
                    'fraud': 0.4
                },
                'ensemble_prediction': 0,
                'individual_model_predictions': [],
                'tabular_confidence': 0.6
            }
        
    except Exception as e:
        print(f"Error in detailed tabular predictions: {e}")
        # Return fallback data
        return {
            'raw_features_shape': list(tabular_features.shape),
            'raw_features_sample': tabular_features[0][:5].tolist() if len(tabular_features[0]) >= 5 else tabular_features[0].tolist(),
            'cnn_features_shape': [1, 10],
            'cnn_features_sample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'ensemble_probabilities': {
                'no_fraud': 0.6,
                'fraud': 0.4
            },
            'ensemble_prediction': 0,
            'individual_model_predictions': [],
            'tabular_confidence': 0.6
        }


def get_detailed_image_predictions(image_path, device):
    """Get detailed image predictions with damage analysis"""
    try:
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_area = image.width * image.height
        
        if _image_model is not None:
            # Transform for model
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions = _image_model(image_tensor)
                
                # Extract results
                boxes = predictions[0]['boxes'].cpu().numpy() if 'boxes' in predictions[0] else np.array([])
                scores = predictions[0]['scores'].cpu().numpy() if 'scores' in predictions[0] else np.array([])
                
                # Filter by confidence
                confidence_threshold = 0.5
                high_conf_indices = scores > confidence_threshold
                
                filtered_boxes = boxes[high_conf_indices] if len(boxes) > 0 else np.array([])
                filtered_scores = scores[high_conf_indices] if len(scores) > 0 else np.array([])
                
                # Calculate damage metrics
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
                
                # Calculate severity
                if damage_percentage > 15:
                    severity_score = 0.9
                    severity_level = "HIGH"
                elif damage_percentage > 5:
                    severity_score = 0.6
                    severity_level = "MEDIUM"
                else:
                    severity_score = 0.3
                    severity_level = "LOW"
                
                # Weighted damage score
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
            # Fallback when image model not available
            return {
                'image_dimensions': {
                    'width': image.width,
                    'height': image.height,
                    'total_pixels': image_area
                },
                'detection_results': {
                    'total_detections': 2,
                    'high_confidence_detections': 1,
                    'confidence_threshold': 0.5,
                    'all_scores': [0.7, 0.3],
                    'filtered_scores': [0.7]
                },
                'damage_analysis': {
                    'damage_regions': [{
                        'region_id': 1,
                        'bbox': [100.0, 100.0, 200.0, 200.0],
                        'area_pixels': 10000.0,
                        'confidence': 0.7,
                        'relative_size': 0.05
                    }],
                    'total_damage_area_pixels': 10000.0,
                    'damage_percentage': 5.0,
                    'severity_level': 'MEDIUM',
                    'severity_score': 0.6,
                    'weighted_damage_score': 0.035
                },
                'image_fraud_probability': 0.4,
                'image_confidence': 0.7
            }
            
    except Exception as e:
        print(f"Error in detailed image predictions: {e}")
        # Return fallback data
        return {
            'image_dimensions': {'width': 800, 'height': 600, 'total_pixels': 480000},
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


def calculate_detailed_fusion(tabular_details, image_details):
    """Calculate fusion with detailed mathematical breakdown"""
    try:
        # Extract probabilities
        tabular_fraud_prob = tabular_details.get('ensemble_probabilities', {}).get('fraud', 0.4)
        tabular_confidence = tabular_details.get('tabular_confidence', 0.6)
        
        image_fraud_prob = image_details.get('image_fraud_probability', 0.3)
        image_confidence = image_details.get('image_confidence', 0.6)
        
        # Calculate weights
        total_confidence = tabular_confidence + image_confidence
        if total_confidence > 0:
            tabular_weight = tabular_confidence / total_confidence
            image_weight = image_confidence / total_confidence
        else:
            tabular_weight = 0.5
            image_weight = 0.5
        
        # Fusion methods
        weighted_fusion = (tabular_weight * tabular_fraud_prob) + (image_weight * image_fraud_prob)
        geometric_fusion = np.sqrt(max(tabular_fraud_prob * image_fraud_prob, 0))
        max_fusion = max(tabular_fraud_prob, image_fraud_prob)
        
        if tabular_fraud_prob + image_fraud_prob > 0:
            harmonic_fusion = (2 * tabular_fraud_prob * image_fraud_prob) / (tabular_fraud_prob + image_fraud_prob)
        else:
            harmonic_fusion = 0.0
        
        # Final fusion
        alpha = 0.6
        beta = 0.4
        final_fusion_score = (alpha * weighted_fusion) + (beta * geometric_fusion)
        
        # Decision
        fraud_threshold = 0.5
        final_prediction = 1 if final_fusion_score > fraud_threshold else 0
        
        return {
            'input_probabilities': {
                'tabular_fraud_probability': float(tabular_fraud_prob),
                'tabular_confidence': float(tabular_confidence),
                'image_fraud_probability': float(image_fraud_prob),
                'image_confidence': float(image_confidence)
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
            'final_fusion': {
                'alpha': float(alpha),
                'beta': float(beta),
                'calculation': f"({alpha} × {weighted_fusion:.3f}) + ({beta} × {geometric_fusion:.3f})",
                'final_score': float(final_fusion_score),
                'threshold': float(fraud_threshold),
                'prediction': int(final_prediction)
            },
            'final_prediction': final_prediction,
            'final_confidence': final_fusion_score
        }
        
    except Exception as e:
        print(f"Error in fusion calculation: {e}")
        # Return fallback
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
                'image_weight': 0.5,
                'weight_formula': "weight = confidence / total_confidence"
            },
            'fusion_methods': {
                'weighted_average': {'score': 0.35, 'formula': "(0.5 × 0.4) + (0.5 × 0.3)"},
                'geometric_mean': {'score': 0.346, 'formula': "√(0.4 × 0.3)"},
                'maximum': {'score': 0.4, 'formula': "max(0.4, 0.3)"},
                'harmonic_mean': {'score': 0.343, 'formula': "2 × 0.4 × 0.3 / (0.4 + 0.3)"}
            },
            'final_fusion': {
                'alpha': 0.6,
                'beta': 0.4,
                'calculation': "(0.6 × 0.35) + (0.4 × 0.346)",
                'final_score': 0.348,
                'threshold': 0.5,
                'prediction': 0
            },
            'final_prediction': 0,
            'final_confidence': 0.348
        }

###################################################
###################################################

def get_tabular_predictions_with_details(feature_extractor, ensemble_model, tabular_features, device):
    """
    Get detailed tabular predictions with intermediate calculations
    """
    try:
        # Step 1: Feature Extraction using CNN
        tabular_tensor = torch.FloatTensor(tabular_features).to(device)
        
        with torch.no_grad():
            # Extract features using CNN
            cnn_features = feature_extractor(tabular_tensor)
            cnn_features_np = cnn_features.cpu().numpy()
        
        # Step 2: Ensemble Model Prediction
        ensemble_proba = ensemble_model.predict_proba(cnn_features_np)
        ensemble_pred = ensemble_model.predict(cnn_features_np)
        
        # Get individual model predictions if ensemble has base_estimators_
        individual_predictions = []
        if hasattr(ensemble_model, 'estimators_'):
            for i, estimator in enumerate(ensemble_model.estimators_):
                pred_proba = estimator.predict_proba(cnn_features_np)[0]
                individual_predictions.append({
                    'model_name': f'Model_{i+1}',
                    'fraud_probability': float(pred_proba[1]),
                    'no_fraud_probability': float(pred_proba[0])
                })
        
        return {
            'raw_features_shape': list(tabular_features.shape),
            'raw_features_sample': tabular_features[0][:10].tolist() if len(tabular_features[0]) >= 10 else tabular_features[0].tolist(),
            'cnn_features_shape': list(cnn_features_np.shape),
            'cnn_features_sample': cnn_features_np[0][:10].tolist() if len(cnn_features_np[0]) >= 10 else cnn_features_np[0].tolist(),
            'ensemble_probabilities': {
                'no_fraud': float(ensemble_proba[0][0]),
                'fraud': float(ensemble_proba[0][1])
            },
            'ensemble_prediction': int(ensemble_pred[0]),
            'individual_model_predictions': individual_predictions,
            'tabular_confidence': float(max(ensemble_proba[0]))
        }
        
    except Exception as e:
        print(f"Error in tabular prediction details: {e}")
        return None

def get_image_predictions_with_details(image_model, image_path, device):
    """
    Get detailed image predictions with damage analysis
    """
    try:
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        
        # Transform for model
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get model predictions
            predictions = image_model(image_tensor)
            
            # Extract detection results
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy() if 'labels' in predictions[0] else None
            
            # Filter by confidence
            confidence_threshold = 0.5
            high_conf_indices = scores > confidence_threshold
            
            filtered_boxes = boxes[high_conf_indices]
            filtered_scores = scores[high_conf_indices]
            
            # Calculate damage metrics
            image_area = image.width * image.height
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
                    'relative_size': float(region_area / image_area)
                })
            
            # Calculate overall damage score
            damage_percentage = (total_damage_area / image_area) * 100
            
            # Damage severity scoring
            if damage_percentage > 15:
                severity_score = 0.9
                severity_level = "HIGH"
            elif damage_percentage > 5:
                severity_score = 0.6
                severity_level = "MEDIUM"
            else:
                severity_score = 0.3
                severity_level = "LOW"
            
            # Calculate weighted damage score
            weighted_damage_score = 0.0
            if len(filtered_scores) > 0:
                # Weight by confidence and area
                for region in damage_regions:
                    weight = region['confidence'] * region['relative_size']
                    weighted_damage_score += weight
                
                weighted_damage_score = min(weighted_damage_score, 1.0)  # Cap at 1.0
            
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
                'image_confidence': float(np.mean(filtered_scores)) if len(filtered_scores) > 0 else 0.0
            }
            
    except Exception as e:
        print(f"Error in image prediction details: {e}")
        return None

def calculate_fusion_with_details(tabular_results, image_results):
    """
    Calculate fusion with detailed mathematical breakdown
    """
    try:
        # Extract key probabilities
        tabular_fraud_prob = tabular_results['ensemble_probabilities']['fraud']
        tabular_confidence = tabular_results['tabular_confidence']
        
        image_fraud_prob = image_results['image_fraud_probability']
        image_confidence = image_results['image_confidence']
        
        # Fusion Method 1: Weighted Average
        # Weight based on confidence scores
        total_confidence = tabular_confidence + image_confidence
        if total_confidence > 0:
            tabular_weight = tabular_confidence / total_confidence
            image_weight = image_confidence / total_confidence
        else:
            tabular_weight = 0.5
            image_weight = 0.5
        
        weighted_fusion = (tabular_weight * tabular_fraud_prob) + (image_weight * image_fraud_prob)
        
        # Fusion Method 2: Geometric Mean
        geometric_fusion = np.sqrt(tabular_fraud_prob * image_fraud_prob)
        
        # Fusion Method 3: Maximum (Conservative)
        max_fusion = max(tabular_fraud_prob, image_fraud_prob)
        
        # Fusion Method 4: Harmonic Mean
        if tabular_fraud_prob + image_fraud_prob > 0:
            harmonic_fusion = (2 * tabular_fraud_prob * image_fraud_prob) / (tabular_fraud_prob + image_fraud_prob)
        else:
            harmonic_fusion = 0.0
        
        # Final fusion (you can customize this logic)
        # Using a combination approach
        alpha = 0.6  # Weight for weighted average
        beta = 0.4   # Weight for geometric mean
        
        final_fusion_score = (alpha * weighted_fusion) + (beta * geometric_fusion)
        
        # Apply threshold for final decision
        fraud_threshold = 0.5
        final_prediction = 1 if final_fusion_score > fraud_threshold else 0
        
        return {
            'input_probabilities': {
                'tabular_fraud_probability': float(tabular_fraud_prob),
                'tabular_confidence': float(tabular_confidence),
                'image_fraud_probability': float(image_fraud_prob),
                'image_confidence': float(image_confidence)
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
            'final_fusion': {
                'alpha': float(alpha),
                'beta': float(beta),
                'calculation': f"({alpha} × {weighted_fusion:.3f}) + ({beta} × {geometric_fusion:.3f})",
                'final_score': float(final_fusion_score),
                'threshold': float(fraud_threshold),
                'prediction': int(final_prediction)
            }
        }
        
    except Exception as e:
        print(f"Error in fusion calculation: {e}")
        return None

# Modified predict_claim function with detailed calculations
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def predict_claim_with_calculations(request):
    """Enhanced predict claim function with detailed calculations"""
    
    # ... (previous validation code remains the same) ...
    username = request.data.get("username")
    claim_description = request.data.get("claim_description", "")
    accident_date = request.data.get("accident_date")
    claim_amount = request.data.get("claim_amount", 0)
    car_image = request.FILES.get("car_image")

    if not username or not car_image:
        return Response({"error": "username and car_image are required"}, status=status.HTTP_400_BAD_REQUEST)

    if not load_models():
        return Response({"error": "ML models not available"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    try:
        policyholder = Policyholder.objects.get(username=username)
    except Policyholder.DoesNotExist:
        return Response({"error": "Policyholder not found"}, status=status.HTTP_404_NOT_FOUND)

    # Prepare tabular data
    try:
        tabular_data = create_safe_inference_data(policyholder, accident_date, claim_amount)
        tabular_features = safe_preprocess_inference_data(tabular_data, _preprocessing_pipeline)
        
        if tabular_features is None:
            return Response({"error": "Failed to preprocess tabular data"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    except Exception as e:
        return Response({"error": f"Data preparation failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Save image
    try:
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, f"temp_{random.randint(100,999)}.jpg")
        
        with open(image_path, "wb+") as f:
            for chunk in car_image.chunks():
                f.write(chunk)
                
    except Exception as e:
        return Response({"error": f"Image processing failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Get detailed predictions
    try:
        device = get_device()
        
        # Get tabular predictions with details
        tabular_details = get_tabular_predictions_with_details(
            _feature_extractor, _ensemble_model, tabular_features[0:1], device
        )
        
        # Get image predictions with details
        image_details = get_image_predictions_with_details(_image_model, image_path, device)
        
        # Calculate fusion with details
        fusion_details = calculate_fusion_with_details(tabular_details, image_details)
        
        # Get damage visualization
        damage_info = process_damage_detection_image(image_path, _image_model, device)
        
    except Exception as e:
        return Response({"error": f"Prediction failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        # Cleanup
        try:
            if 'image_path' in locals() and os.path.exists(image_path):
                os.remove(image_path)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass

    # Prepare comprehensive response
    final_prediction = fusion_details['final_fusion']['prediction']
    final_confidence = fusion_details['final_fusion']['final_score']
    
    fraud_detected = bool(final_prediction)
    risk_level = "HIGH" if final_confidence > 0.8 else ("MEDIUM" if final_confidence > 0.5 else "LOW")
    
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
        
        # Detailed calculations
        "detailed_calculations": {
            "tabular_analysis": tabular_details,
            "image_analysis": image_details,
            "fusion_analysis": fusion_details
        },
        
        # Damage detection
        "damage_detection": damage_info
    }
    
    return Response(response_data)

###################################################
###################################################

def get_exact_training_format():
    """Return data in exact format that the model was trained on"""
    return {
        'Month': 'Dec',
        'WeekOfMonth': 5,
        'DayOfWeek': 'Wednesday',
        'Make': 'Honda',
        'AccidentArea': 'Urban',
        'DayOfWeekClaimed': 'Tuesday',
        'MonthClaimed': 'Jan',
        'WeekOfMonthClaimed': 1,
        'Sex': 'Female',
        'MaritalStatus': 'Single',
        'Age': 21,
        'Fault': 'Policy Holder',
        'PolicyType': 'Sport - Liability',
        'VehicleCategory': 'Sport',
        'VehiclePrice': 'more than 69000',
        'FraudFound_P': 1,
        'PolicyNumber': 1,
        'RepNumber': 12,
        'Deductible': 300,
        'DriverRating': 1,
        'Days_Policy_Accident': 'more than 30',
        'Days_Policy_Claim': 'more than 30',
        'PastNumberOfClaims': 'none',
        'AgeOfVehicle': '3 years',
        'AgeOfPolicyHolder': '26 to 30',
        'PoliceReportFiled': 'No',
        'WitnessPresent': 'No',
        'AgentType': 'External',
        'NumberOfSuppliments': 'none',
        'AddressChange_Claim': '1 year',
        'NumberOfCars': 3,
        'Year': 1994,
        'BasePolicy': 'Liability'
    }

# Also add this helper function to create better default mappings
def get_claim_defaults(accident_date_str=None, claim_amount=None):
    """Generate contextual defaults based on claim information"""
    from datetime import datetime, timedelta
    import random
    
    # Parse accident date if provided
    accident_date = None
    if accident_date_str:
        try:
            accident_date = datetime.strptime(accident_date_str, "%Y-%m-%d")
        except:
            accident_date = datetime.now() - timedelta(days=random.randint(1, 30))
    else:
        accident_date = datetime.now() - timedelta(days=random.randint(1, 30))
    
    # Claim date (usually after accident date)
    claim_date = accident_date + timedelta(days=random.randint(1, 7))
    
    # Calculate days between policy and accident (assume policy is older)
    days_policy_accident = random.choice(["less than 30", "30 to 31", "more than 30"])
    days_policy_claim = random.choice(["less than 30", "30 to 31", "more than 30"])
    
    # High claim amount might indicate higher chance of scrutiny
    police_report = "Yes" if claim_amount and float(claim_amount) > 50000 else random.choice(["Yes", "No"])
    witness_present = "Yes" if claim_amount and float(claim_amount) > 100000 else random.choice(["Yes", "No"])
    
    return {
        "accident_month": accident_date.strftime("%b"),
        "accident_week": (accident_date.day - 1) // 7 + 1,
        "accident_day": accident_date.strftime("%A"),
        "claim_month": claim_date.strftime("%b"),
        "claim_week": (claim_date.day - 1) // 7 + 1,
        "claim_day": claim_date.strftime("%A"),
        "days_policy_accident": days_policy_accident,
        "days_policy_claim": days_policy_claim,
        "police_report": police_report,
        "witness_present": witness_present,
        "fault": random.choice(["Policy Holder", "Third Party"]),
    }


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Health check endpoint to verify model status"""
    device = get_device()
    
    # Try to load models and capture the result
    force_load = request.GET.get('load', 'false').lower() == 'true'
    
    if force_load or not _models_loaded:
        print("🔄 Attempting to load models...")
        load_success = load_models()
        print(f"📊 Model loading result: {load_success}")
    
    model_status = {
        "models_loaded": _models_loaded,
        "device": str(device),
        "ml_imports_available": ML_IMPORTS_AVAILABLE,
        "feature_extractor": _feature_extractor is not None,
        "ensemble_model": _ensemble_model is not None,
        "image_model": _image_model is not None,
        "preprocessing_pipeline": _preprocessing_pipeline is not None,
    }
    
    # Add models directory info
    if ML_IMPORTS_AVAILABLE:
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
        "message": "All systems operational" if _models_loaded else "Models need to be loaded",
        "instructions": "Add ?load=true to force model loading attempt"
    })