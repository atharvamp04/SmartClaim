# detection/views.py
import os
import json
import tempfile
import torch
import pandas as pd
from PIL import Image
from io import StringIO, BytesIO
from datetime import datetime, timedelta


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

def create_inference_data(policyholder, accident_date=None, claim_amount=None):
    """
    Create properly formatted data for inference that matches training data structure
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
    
    # Calculate age category
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
    
    # Create the properly formatted data
    data = {
        # Date-related features
        'Month': acc_date.strftime("%b"),
        'WeekOfMonth': (acc_date.day - 1) // 7 + 1,
        'DayOfWeek': acc_date.strftime("%A"),
        'MonthClaimed': claim_date.strftime("%b"),
        'WeekOfMonthClaimed': (claim_date.day - 1) // 7 + 1,
        'DayOfWeekClaimed': claim_date.strftime("%A"),
        
        # Vehicle information (use defaults or from policyholder)
        'Make': getattr(policyholder, 'make', 'Honda'),
        'VehicleCategory': getattr(policyholder, 'vehicle_category', 'Sedan'),
        'VehiclePrice': getattr(policyholder, 'vehicle_price', '20000 to 29000'),
        'Year': int(getattr(policyholder, 'year_of_vehicle', 2015)),
        'AgeOfVehicle': '3 years',  # Could calculate from Year if needed
        
        # Policy information
        'PolicyType': getattr(policyholder, 'policy_type', 'Sedan - All Perils'),
        'BasePolicy': getattr(policyholder, 'base_policy', 'All Perils'),
        'Deductible': int(getattr(policyholder, 'deductible', 300)),
        'DriverRating': int(getattr(policyholder, 'driver_rating', 1)),
        
        # Personal information
        'Sex': str(getattr(policyholder, 'sex', 'Male')),
        'MaritalStatus': str(getattr(policyholder, 'marital_status', 'Married')),
        'Age': age,
        'AgeOfPolicyHolder': age_category,
        'NumberOfCars': int(getattr(policyholder, 'number_of_cars', 1)),
        
        # Claim specifics (use contextual defaults)
        'AccidentArea': 'Urban',  # Could be random.choice(['Urban', 'Rural'])
        'Fault': 'Policy Holder',  # Could vary based on claim
        'PoliceReportFiled': 'Yes' if claim_amount and float(claim_amount) > 10000 else 'No',
        'WitnessPresent': 'Yes' if claim_amount and float(claim_amount) > 50000 else 'No',
        'AgentType': 'Internal',
        
        # Policy history
        'Days_Policy_Accident': 'more than 30',
        'Days_Policy_Claim': 'more than 30',
        'PastNumberOfClaims': getattr(policyholder, 'past_claims', 'none'),
        'NumberOfSuppliments': 'none',
        'AddressChange_Claim': 'no change'
    }
    
    return data

def safe_preprocess_inference_data(tabular_data, pipeline):
    """
    Safely preprocess inference data with comprehensive error handling
    """
    try:
        # Create DataFrame
        if isinstance(tabular_data, dict):
            df = pd.DataFrame([tabular_data])
        else:
            df = tabular_data.copy()
        
        print(f"📊 Input data shape: {df.shape}")
        print(f"📊 Input columns: {df.columns.tolist()}")
        
        # Ensure all expected columns exist (add missing with defaults)
        expected_columns = [
            'Month', 'WeekOfMonth', 'DayOfWeek', 'Make', 'AccidentArea', 
            'DayOfWeekClaimed', 'MonthClaimed', 'WeekOfMonthClaimed', 'Sex', 
            'MaritalStatus', 'Age', 'Fault', 'PolicyType', 'VehicleCategory', 
            'VehiclePrice', 'Deductible', 'DriverRating', 'Days_Policy_Accident', 
            'Days_Policy_Claim', 'PastNumberOfClaims', 'AgeOfVehicle', 
            'AgeOfPolicyHolder', 'PoliceReportFiled', 'WitnessPresent', 
            'AgentType', 'NumberOfSuppliments', 'AddressChange_Claim', 
            'NumberOfCars', 'Year', 'BasePolicy'
        ]
        
        # Add missing columns with safe defaults
        for col in expected_columns:
            if col not in df.columns:
                if col in ['Age', 'Year', 'NumberOfCars', 'WeekOfMonth', 'WeekOfMonthClaimed', 'Deductible', 'DriverRating']:
                    df[col] = 30 if col == 'Age' else (2015 if col == 'Year' else (1 if col in ['NumberOfCars', 'WeekOfMonth', 'WeekOfMonthClaimed', 'DriverRating'] else 300))
                else:
                    # Categorical defaults
                    defaults = {
                        'Month': 'Jan', 'DayOfWeek': 'Monday', 'Make': 'Honda', 'AccidentArea': 'Urban',
                        'DayOfWeekClaimed': 'Monday', 'MonthClaimed': 'Jan', 'Sex': 'Male', 
                        'MaritalStatus': 'Married', 'Fault': 'Policy Holder', 'PolicyType': 'Sedan - All Perils',
                        'VehicleCategory': 'Sedan', 'VehiclePrice': '20000 to 29000', 
                        'Days_Policy_Accident': 'more than 30', 'Days_Policy_Claim': 'more than 30',
                        'PastNumberOfClaims': 'none', 'AgeOfVehicle': '3 years', 
                        'AgeOfPolicyHolder': '26 to 30', 'PoliceReportFiled': 'No', 
                        'WitnessPresent': 'No', 'AgentType': 'Internal', 
                        'NumberOfSuppliments': 'none', 'AddressChange_Claim': 'no change',
                        'BasePolicy': 'All Perils'
                    }
                    df[col] = defaults.get(col, 'Unknown')
        
        # Data type cleaning
        categorical_cols = [col for col in df.columns if col not in ['Age', 'Year', 'NumberOfCars', 'WeekOfMonth', 'WeekOfMonthClaimed', 'Deductible', 'DriverRating']]
        numeric_cols = ['Age', 'Year', 'NumberOfCars', 'WeekOfMonth', 'WeekOfMonthClaimed', 'Deductible', 'DriverRating']
        
        # Clean categorical columns
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Clean numeric columns
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(30 if col == 'Age' else (2015 if col == 'Year' else 1))
                df[col] = df[col].astype(int)
        
        print(f"🧹 Data cleaned, shape: {df.shape}")
        
        # Transform using pipeline
        result = pipeline.transform(df)
        print(f"✅ Preprocessing successful! Output shape: {result.shape}")
        
        return result
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def predict_claim(request):
    """Enhanced predict claim function with improved data preprocessing"""
    
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
            {"error": "ML models not available. Please check server configuration."}, 
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    # Get policyholder data
    try:
        policyholder = Policyholder.objects.get(username=username)
    except Policyholder.DoesNotExist:
        return Response({"error": "Policyholder not found"}, status=status.HTTP_404_NOT_FOUND)

    # Create properly formatted tabular data
    try:
        tabular_data = create_inference_data(policyholder, accident_date, claim_amount)
        print(f"🔧 Created inference data: {tabular_data}")
        
        # Preprocess the data
        tabular_features = safe_preprocess_inference_data(tabular_data, _preprocessing_pipeline)
        
        if tabular_features is None:
            return Response(
                {"error": "Failed to preprocess tabular data"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        print(f"✅ Tabular features ready, shape: {tabular_features.shape}")
        
    except Exception as e:
        print(f"❌ Data preparation failed: {str(e)}")
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
        
        # Save uploaded image
        with open(image_path, "wb+") as f:
            for chunk in car_image.chunks():
                f.write(chunk)
        
        print(f"📷 Image saved: {image_path}")
        
    except Exception as e:
        return Response(
            {"error": f"Image processing failed: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    # Run ML prediction
    try:
        device = get_device()
        print(f"🔮 Running prediction on device: {device}")
        
        prediction, confidence = fused_prediction(
            _feature_extractor, 
            _ensemble_model, 
            _image_model, 
            tabular_features[0:1], 
            image_path, 
            device
        )
        
        print(f"📊 Prediction result - Fraud: {prediction}, Confidence: {confidence}")
        
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

    # Prepare response
    fraud_detected = bool(prediction)
    risk_level = "HIGH" if confidence > 0.8 else ("MEDIUM" if confidence > 0.5 else "LOW")
    
    return Response({
        "username": username,
        "claim_description": claim_description,
        "accident_date": accident_date,
        "claim_amount": claim_amount,
        "prediction": int(prediction),
        "confidence": float(confidence),
        "fraud_detected": fraud_detected,
        "risk_level": risk_level,
        "message": f"Fraud detected with {risk_level.lower()} confidence" if fraud_detected else f"No fraud detected ({risk_level.lower()} confidence)",
        "debug_info": {
            "tabular_features_shape": list(tabular_features.shape),
            "model_device": str(device),
            "image_processed": True,
            "policyholder_data_fields": len([k for k, v in tabular_data.items() if v is not None])
        }
    })

# Add this test function to verify your pipeline works
# @api_view(['GET'])
# @permission_classes([AllowAny])
# def test_preprocessing(request):
#     """Test endpoint to verify preprocessing pipeline"""
    
#     if not load_models():
#         return Response({"error": "Models not loaded"}, status=500)
    
#     # Create test data
#     test_data = {
#         'Month': 'Aug',
#         'WeekOfMonth': 2,
#         'DayOfWeek': 'Sunday',
#         'Make': 'Honda',
#         'AccidentArea': 'Urban',
#         'DayOfWeekClaimed': 'Friday',
#         'MonthClaimed': 'Aug',
#         'WeekOfMonthClaimed': 3,
#         'Sex': 'Male',
#         'MaritalStatus': 'Married',
#         'Age': 49,
#         'Fault': 'Policy Holder',
#         'PolicyType': 'Sedan - All Perils',
#         'VehicleCategory': 'Sedan',
#         'VehiclePrice': '20000 to 29000',
#         'Deductible': 300,
#         'DriverRating': 1,
#         'Days_Policy_Accident': 'more than 30',
#         'Days_Policy_Claim': 'more than 30',
#         'PastNumberOfClaims': 'none',
#         'AgeOfVehicle': '3 years',
#         'AgeOfPolicyHolder': '41 to 50',
#         'PoliceReportFiled': 'Yes',
#         'WitnessPresent': 'No',
#         'AgentType': 'Internal',
#         'NumberOfSuppliments': 'none',
#         'AddressChange_Claim': 'no change',
#         'NumberOfCars': 1,
#         'Year': 2012,
#         'BasePolicy': 'All Perils'
#     }
    
#     # Test preprocessing
#     try:
#         result = safe_preprocess_inference_data(test_data, _preprocessing_pipeline)
        
#         if result is not None:
#             return Response({
#                 "status": "success",
#                 "message": "Preprocessing pipeline working correctly",
#                 "input_shape": [1, len(test_data)],
#                 "output_shape": list(result.shape),
#                 "sample_features": result[0][:10].tolist() if len(result[0]) >= 10 else result[0].tolist()
#             })
#         else:
#             return Response({
#                 "status": "error", 
#                 "message": "Preprocessing failed"
#             }, status=500)
            
#     except Exception as e:
#         return Response({
#             "status": "error",
#             "message": f"Test failed: {str(e)}"
#         }, status=500)
#     """Predict claim using policyholder data and uploaded image"""
#     username = request.data.get("username")
#     claim_description = request.data.get("claim_description")
#     accident_date = request.data.get("accident_date")
#     claim_amount = request.data.get("claim_amount")
#     car_image = request.FILES.get("car_image")

#     # Validate required inputs
#     if not username:
#         return Response({"error": "username is required"}, status=status.HTTP_400_BAD_REQUEST)
#     if not car_image:
#         return Response({"error": "car_image is required"}, status=status.HTTP_400_BAD_REQUEST)

#     # Check if models are available
#     if not load_models():
#         return Response(
#             {"error": "ML models not available. Please check server configuration."}, 
#             status=status.HTTP_503_SERVICE_UNAVAILABLE
#         )

#     # Get tabular data from Policyholder model
#     try:
#         policyholder = Policyholder.objects.get(username=username)
#     except Policyholder.DoesNotExist:
#         return Response({"error": "Policyholder not found"}, status=status.HTTP_404_NOT_FOUND)

#     # Create basic tabular data structure
#     try:
#         basic_data = {
#             'Sex': str(getattr(policyholder, 'sex', 'Male')),
#             'MaritalStatus': str(getattr(policyholder, 'marital_status', 'Married') or 'Married'),
#             'Age': int(getattr(policyholder, 'age', 49)),
#             'NumberOfCars': int(getattr(policyholder, 'number_of_cars', 1)),
#             'Year': int(getattr(policyholder, 'year_of_vehicle', 2012) or 2012),
#             'WeekOfMonth': 2,  # From your current data
#             'WeekOfMonthClaimed': 3,  # From your current data
#             'DayOfWeek': 'Sunday',
#             'DayOfWeekClaimed': 'Friday',
#             'AgeOfPolicyHolder': (
#                 "16 to 17" if policyholder.age <= 17 else
#                 "18 to 20" if policyholder.age <= 20 else
#                 "21 to 25" if policyholder.age <= 25 else
#                 "26 to 30" if policyholder.age <= 30 else
#                 "31 to 35" if policyholder.age <= 35 else
#                 "36 to 40" if policyholder.age <= 40 else
#                 "41 to 50" if policyholder.age <= 50 else
#                 "51 to 65" if policyholder.age <= 65 else
#                 "over 65"
#             )
#         }
        
#         # Create training format with safe defaults
#         tabular_data = create_minimal_training_format(basic_data)
        
#     except Exception as e:
#         print(f"❌ Error creating tabular_data: {e}")
#         return Response(
#             {"error": f"Error extracting policyholder data: {str(e)}"}, 
#             status=status.HTTP_500_INTERNAL_SERVER_ERROR
#         )

#     # Convert to DataFrame and preprocess
#     try:
#         tabular_df = pd.DataFrame([tabular_data])
#         print(f"📊 DataFrame created with shape: {tabular_df.shape}")
        
#         # Try safe preprocessing first
#         tabular_features = safe_preprocess_tabular_data(tabular_df, _preprocessing_pipeline)
        
#         if tabular_features is None:
#             print("❌ Safe preprocessing failed, trying fallback...")
#             # Fallback: try with just the exact dummy data that worked during model loading
#             dummy_csv = """Month,WeekOfMonth,DayOfWeek,Make,AccidentArea,DayOfWeekClaimed,MonthClaimed,WeekOfMonthClaimed,Sex,MaritalStatus,Age,Fault,PolicyType,VehicleCategory,VehiclePrice,FraudFound_P,PolicyNumber,RepNumber,Deductible,DriverRating,Days_Policy_Accident,Days_Policy_Claim,PastNumberOfClaims,AgeOfVehicle,AgeOfPolicyHolder,PoliceReportFiled,WitnessPresent,AgentType,NumberOfSuppliments,AddressChange_Claim,NumberOfCars,Year,BasePolicy
# Dec,5,Wednesday,Honda,Urban,Tuesday,Jan,1,Female,Single,21,Policy Holder,Sport - Liability,Sport,more than 69000,0,1,12,300,1,more than 30,more than 30,none,3 years,26 to 30,No,No,External,none,1 year,3 to 4,1994,Liability"""
            
#             fallback_df = pd.read_csv(StringIO(dummy_csv))
#             # Update key fields with actual user data
#             fallback_df.loc[0, 'Sex'] = tabular_data['Sex']
#             fallback_df.loc[0, 'MaritalStatus'] = tabular_data['MaritalStatus']
#             fallback_df.loc[0, 'Age'] = tabular_data['Age']
#             fallback_df.loc[0, 'AgeOfPolicyHolder'] = tabular_data['AgeOfPolicyHolder']
#             fallback_df.loc[0, 'NumberOfCars'] = tabular_data['NumberOfCars']
#             fallback_df.loc[0, 'Year'] = tabular_data['Year']
            
#             print("🔄 Trying fallback preprocessing...")
#             tabular_features = preprocess_tabular_data_with_pipeline(fallback_df, _preprocessing_pipeline)
        
#         if tabular_features is None:
#             return Response(
#                 {"error": "Failed to preprocess tabular data - all methods failed"}, 
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )
        
#         print(f"✅ Preprocessing successful! Features shape: {tabular_features.shape}")
        
#     except Exception as e:
#         print(f"❌ Data preprocessing failed: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return Response(
#             {"error": f"Data preprocessing failed: {str(e)}"}, 
#             status=status.HTTP_500_INTERNAL_SERVER_ERROR
#         )

#     # Save uploaded image temporarily
#     try:
#         temp_dir = tempfile.mkdtemp()
#         image_path = os.path.join(temp_dir, f"temp_{car_image.name}")
        
#         with open(image_path, "wb+") as f:
#             for chunk in car_image.chunks():
#                 f.write(chunk)
                
#         print(f"📷 Image saved: {image_path}")
        
#     except Exception as e:
#         print(f"❌ Failed to process image: {str(e)}")
#         return Response(
#             {"error": f"Failed to process image: {str(e)}"}, 
#             status=status.HTTP_500_INTERNAL_SERVER_ERROR
#         )

#     # Run ML prediction
#     try:
#         device = get_device()
#         print(f"🔮 Running prediction on device: {device}")
        
#         prediction, confidence = fused_prediction(
#             _feature_extractor, 
#             _ensemble_model, 
#             _image_model, 
#             tabular_features[0:1], 
#             image_path, 
#             device
#         )
        
#         print(f"📊 Prediction result - Fraud: {prediction}, Confidence: {confidence}")
        
#     except Exception as e:
#         print(f"❌ Prediction failed: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return Response(
#             {"error": f"Prediction failed: {str(e)}"}, 
#             status=status.HTTP_500_INTERNAL_SERVER_ERROR
#         )
#     finally:
#         # Clean up temporary file
#         try:
#             if os.path.exists(image_path):
#                 os.remove(image_path)
#             if os.path.exists(temp_dir):
#                 os.rmdir(temp_dir)
#         except:
#             pass

#     # Return response
#     return Response({
#         "username": username,
#         "claim_description": claim_description,
#         "accident_date": accident_date,
#         "claim_amount": claim_amount,
#         "prediction": int(prediction),
#         "confidence": float(confidence),
#         "fraud_detected": bool(prediction),
#         "message": "Fraud detected" if prediction else "No fraud detected",
#         "debug_info": {
#             "tabular_features_shape": list(tabular_features.shape),
#             "model_device": str(device),
#             "image_processed": True
#         }
#     })

# Even simpler approach - create exact training format
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