# detection/fusion.py
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision.transforms import functional as F
from PIL import Image
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from io import StringIO
from django.conf import settings

# Conditional import for joblib/sklearn
try:
    import joblib
    import warnings
    # Suppress sklearn version warnings for model compatibility
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    SKLEARN_AVAILABLE = True
    print("✅ sklearn/joblib available")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    print(f"⚠️ sklearn/joblib not available: {e}")
    print("   Models requiring sklearn will return None")
    
    # Create dummy joblib for compatibility
    class DummyJoblib:
        @staticmethod
        def load(filepath):
            print(f"Warning: sklearn not available, cannot load {filepath}")
            return None
    
    joblib = DummyJoblib()

# -------------------------------
# 1. Define Tabular CNN Model Architecture
# -------------------------------
class TabularCNN(nn.Module):
    def __init__(self, input_size):
        super(TabularCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# -------------------------------
# 2. Get model directory path
# -------------------------------
def get_models_dir():
    """Get the absolute path to the models directory"""
    try:
        # Try to get Django BASE_DIR first
        base_dir = getattr(settings, 'BASE_DIR', None)
        if base_dir is None:
            raise AttributeError("BASE_DIR not set")
    except (AttributeError, ImportError):
        # Fallback for standalone execution
        # Assume we're in backend/detection/, so go up two levels to get to backend/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    models_dir = os.path.join(base_dir, 'models')
    return models_dir

# -------------------------------
# 3. Load Tabular CNN Feature Extractor (with error handling)
# -------------------------------
def load_tabular_cnn_feature_extractor(model_filename, input_size, device):
    """Load CNN model with proper error handling"""
    try:
        models_dir = get_models_dir()
        model_path = os.path.join(models_dir, model_filename)
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
        
        model = TabularCNN(input_size)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        # Remove last Linear(64->1) and Sigmoid layers to get feature extractor
        feature_extractor = nn.Sequential(*list(model.model.children())[:-2])
        print(f"✅ Loaded tabular CNN feature extractor from {model_filename}")
        return feature_extractor
    except Exception as e:
        print(f"Error loading tabular CNN model: {e}")
        return None

# -------------------------------
# 4. Load Ensemble Model (sklearn)
# -------------------------------
def load_ensemble_model(pkl_filename):
    """Load ensemble model with error handling"""
    if not SKLEARN_AVAILABLE:
        print(f"⚠️ Cannot load {pkl_filename}: sklearn not available")
        return None
        
    try:
        models_dir = get_models_dir()
        pkl_path = os.path.join(models_dir, pkl_filename)
        
        if not os.path.exists(pkl_path):
            print(f"Ensemble model file not found: {pkl_path}")
            return None
        
        # Suppress version warnings temporarily
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            model = joblib.load(pkl_path)
        
        print(f"✅ Loaded ensemble model from {pkl_filename}")
        return model
    except Exception as e:
        print(f"Error loading ensemble model: {e}")
        print("💡 This might be due to sklearn version mismatch.")
        print("   Try: pip install scikit-learn==1.5.2")
        return None

# -------------------------------
# 5. Load Image Damage Model
# -------------------------------
def load_image_model(pth_filename, device, num_classes=4):
    """Load image model with error handling"""
    try:
        models_dir = get_models_dir()
        pth_path = os.path.join(models_dir, pth_filename)
        
        if not os.path.exists(pth_path):
            print(f"Image model file not found: {pth_path}")
            return None
        
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(256, 256, num_classes)
        model.load_state_dict(torch.load(pth_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ Loaded image model from {pth_filename}")
        return model
    except Exception as e:
        print(f"Error loading image model: {e}")
        return None

# -------------------------------
# 6. Load preprocessing pipeline (NEW FUNCTION)
# -------------------------------
def load_preprocessing_pipeline(pipeline_filename):
    """Load preprocessing pipeline with error handling"""
    if not SKLEARN_AVAILABLE:
        print(f"⚠️ Cannot load {pipeline_filename}: sklearn not available")
        return None
        
    try:
        models_dir = get_models_dir()
        pipeline_path = os.path.join(models_dir, pipeline_filename)
        
        if not os.path.exists(pipeline_path):
            print(f"Pipeline file not found: {pipeline_path}")
            return None
        
        # Suppress version warnings temporarily
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            pipeline = joblib.load(pipeline_path)
        
        print(f"✅ Loaded preprocessing pipeline from {pipeline_filename}")
        return pipeline
    except Exception as e:
        print(f"Error loading preprocessing pipeline: {e}")
        print("💡 This might be due to sklearn version mismatch.")
        print("   Try: pip install scikit-learn==1.5.2")
        return None


def safe_preprocess_inference_data(df_raw, pipeline):
    """
    Preprocess inference DataFrame safely.
    Ensures categorical columns are strings and numeric columns are numeric.
    """
    if pipeline is None:
        print("No preprocessing pipeline loaded")
        return None

    try:
        # Create a copy to avoid modifying the original
        df = df_raw.copy()
        
        # List of categorical columns (update as per your training pipeline)
        cat_cols = [
            'Month', 'DayOfWeek', 'MonthClaimed', 'DayOfWeekClaimed',
            'Make', 'VehicleCategory', 'VehiclePrice', 'PolicyType',
            'BasePolicy', 'Sex', 'MaritalStatus', 'AccidentArea',
            'Fault', 'PoliceReportFiled', 'WitnessPresent', 'AgentType',
            'Days_Policy_Accident', 'Days_Policy_Claim',
            'PastNumberOfClaims', 'NumberOfSuppliments', 'AddressChange_Claim',
            'AgeOfVehicle', 'AgeOfPolicyHolder'  # Added these as they seem categorical
        ]
        
        # Numeric columns
        num_cols = [
            'WeekOfMonth', 'WeekOfMonthClaimed', 'Year', 'Deductible',
            'DriverRating', 'Age', 'NumberOfCars'
        ]
        
        print(f"📊 Processing data with shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        
        # First, handle numeric columns and convert to proper numeric types
        for col in num_cols:
            if col in df.columns:
                # Convert to numeric, replacing non-numeric with NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"   ✅ Numeric column {col}: {df[col].dtype}")
        
        # Fill NaNs in numeric columns with appropriate defaults
        numeric_defaults = {
            'WeekOfMonth': 1,
            'WeekOfMonthClaimed': 1, 
            'Year': 2000,
            'Deductible': 0,
            'DriverRating': 1,
            'Age': 25,
            'NumberOfCars': 1
        }
        
        for col, default_val in numeric_defaults.items():
            if col in df.columns:
                df[col].fillna(default_val, inplace=True)
        
        # Handle categorical columns - convert everything to string and handle nulls
        for col in cat_cols:
            if col in df.columns:
                # First convert to string, handling any None/NaN values
                df[col] = df[col].astype(str)
                # Replace 'nan', 'None', empty strings with 'missing'
                df[col] = df[col].replace(['nan', 'None', '', 'null', 'NaN'], 'missing')
                print(f"   ✅ Categorical column {col}: unique values = {df[col].nunique()}")
        
        # Final safety check - ensure no remaining NaN values
        if df.isnull().any().any():
            print("⚠️ Warning: Found remaining NaN values, filling with defaults...")
            # Fill remaining NaNs based on column type
            for col in df.columns:
                if col in num_cols:
                    df[col].fillna(0, inplace=True)
                else:
                    df[col].fillna('missing', inplace=True)
        
        print(f"✅ Data preprocessing complete. Final shape: {df.shape}")
        print(f"✅ Data types: {df.dtypes.to_dict()}")
        
        # Transform using the pipeline
        X_processed = pipeline.transform(df)
        
        # Convert sparse matrix to dense if needed
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
            
        print(f"✅ Pipeline transformation complete. Output shape: {X_processed.shape}")
        return X_processed.astype(np.float32)

    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        print(f"❌ Error type: {type(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return None


def debug_pipeline_categories(pipeline):
    """
    Debug function to inspect the categories in the loaded pipeline
    """
    try:
        # Access the ColumnTransformer
        if hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps:
            preprocessor = pipeline.named_steps['preprocessor']
        elif hasattr(pipeline, 'transformers_'):
            preprocessor = pipeline
        else:
            print("Cannot find preprocessor in pipeline")
            return
            
        # Find the OneHotEncoder
        for name, transformer, columns in preprocessor.transformers_:
            if hasattr(transformer, 'categories_'):
                print(f"\n🔍 Transformer: {name}")
                print(f"   Columns: {columns}")
                for i, categories in enumerate(transformer.categories_):
                    print(f"   Column {i} categories: {categories[:5]}...")  # Show first 5
                    
    except Exception as e:
        print(f"Error debugging pipeline: {e}")


def create_emergency_preprocessing_pipeline():
    """
    Create a new preprocessing pipeline as emergency fallback
    This should match your original training pipeline structure
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    
    # Define column types (adjust based on your training data)
    categorical_columns = [
        'Month', 'DayOfWeek', 'MonthClaimed', 'DayOfWeekClaimed',
        'Make', 'VehicleCategory', 'VehiclePrice', 'PolicyType',
        'BasePolicy', 'Sex', 'MaritalStatus', 'AccidentArea',
        'Fault', 'PoliceReportFiled', 'WitnessPresent', 'AgentType',
        'Days_Policy_Accident', 'Days_Policy_Claim',
        'PastNumberOfClaims', 'NumberOfSuppliments', 'AddressChange_Claim',
        'AgeOfVehicle', 'AgeOfPolicyHolder'
    ]
    
    numerical_columns = [
        'WeekOfMonth', 'WeekOfMonthClaimed', 'Year', 'Deductible',
        'DriverRating', 'Age', 'NumberOfCars'
    ]
    
    # Create preprocessors
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        drop='first',
        sparse_output=False,
        handle_unknown='ignore'  # This is crucial for inference
    )
    
    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ]
    )
    
    return preprocessor


def emergency_preprocess_data(df_raw):
    """
    Emergency preprocessing without using the saved pipeline
    """
    try:
        df = df_raw.copy()
        
        # Create and fit emergency pipeline on the inference data itself
        emergency_pipeline = create_emergency_preprocessing_pipeline()
        
        # Clean the data first
        categorical_columns = [
            'Month', 'DayOfWeek', 'MonthClaimed', 'DayOfWeekClaimed',
            'Make', 'VehicleCategory', 'VehiclePrice', 'PolicyType',
            'BasePolicy', 'Sex', 'MaritalStatus', 'AccidentArea',
            'Fault', 'PoliceReportFiled', 'WitnessPresent', 'AgentType',
            'Days_Policy_Accident', 'Days_Policy_Claim',
            'PastNumberOfClaims', 'NumberOfSuppliments', 'AddressChange_Claim',
            'AgeOfVehicle', 'AgeOfPolicyHolder'
        ]
        
        numerical_columns = [
            'WeekOfMonth', 'WeekOfMonthClaimed', 'Year', 'Deductible',
            'DriverRating', 'Age', 'NumberOfCars'
        ]
        
        # Clean numerical columns
        for col in numerical_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Clean categorical columns
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).replace(['nan', 'None', ''], 'Unknown')
        
        # Fit and transform (only for emergency use)
        # Note: This won't give you the same feature space as your trained model
        X_processed = emergency_pipeline.fit_transform(df)
        
        print(f"⚠️ Emergency preprocessing completed. Shape: {X_processed.shape}")
        print("⚠️ WARNING: This may not match your trained model's feature space!")
        
        return X_processed.astype(np.float32)
        
    except Exception as e:
        print(f"❌ Emergency preprocessing also failed: {e}")
        return None


def safe_preprocess_with_fallback(df_raw, pipeline):
    """
    Try normal preprocessing first, fall back to emergency preprocessing
    """
    # Try the robust preprocessing first
    result = safe_preprocess_inference_data(df_raw, pipeline)
    
    if result is None:
        print("\n🚨 Normal preprocessing failed, trying emergency fallback...")
        result = emergency_preprocess_data(df_raw)
        
    return result
# -------------------------------
# 7. Tabular Data Preprocessing using saved sklearn pipeline
# -------------------------------
def preprocess_tabular_data_with_pipeline(df_raw, pipeline=None, pipeline_path=None):
    """Preprocess data with loaded pipeline or pipeline path"""
    if not SKLEARN_AVAILABLE:
        print("⚠️ Cannot preprocess tabular data: sklearn not available")
        return None
        
    try:
        # Load pipeline if path is provided
        if pipeline is None and pipeline_path is not None:
            pipeline = load_preprocessing_pipeline(pipeline_path)
        
        if pipeline is None:
            print("No valid preprocessing pipeline available")
            return None
        
        # --- FIX: Ensure correct types for pipeline ---
        cat_cols = [
            'Month', 'DayOfWeek', 'MonthClaimed', 'DayOfWeekClaimed',
            'Make', 'VehicleCategory', 'VehiclePrice', 'PolicyType',
            'BasePolicy', 'Sex', 'MaritalStatus', 'AccidentArea',
            'Fault', 'PoliceReportFiled', 'WitnessPresent', 'AgentType',
            'Days_Policy_Accident', 'Days_Policy_Claim',
            'PastNumberOfClaims', 'NumberOfSuppliments', 'AddressChange_Claim'
        ]
        for col in cat_cols:
            if col in df_raw.columns:
                df_raw[col] = df_raw[col].astype(str)
        
        # Convert numeric columns
        num_cols = ['WeekOfMonth', 'WeekOfMonthClaimed', 'Year', 'Deductible',
                    'DriverRating', 'Age', 'NumberOfCars']
        for col in num_cols:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        
        # Handle any remaining NaNs (optional, depending on your pipeline)
        df_raw.fillna('missing', inplace=True)  # categorical fallback
        df_raw.fillna(0, inplace=True)          # numeric fallback
        
        X_processed = pipeline.transform(df_raw)
        # Convert sparse matrix to dense if needed
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
        return X_processed.astype(np.float32)
    except Exception as e:
        print(f"Error preprocessing tabular data: {e}")
        return None


# -------------------------------
# 8. Feature Extraction and Prediction Functions
# -------------------------------
def get_tabular_features(feature_extractor, tabular_raw, device):
    """Extract features from tabular data"""
    if feature_extractor is None:
        return None
    
    try:
        with torch.no_grad():
            x = torch.tensor(tabular_raw).float().to(device)
            features = feature_extractor(x)
            return features.cpu().numpy()
    except Exception as e:
        print(f"Error extracting tabular features: {e}")
        return None

def get_tabular_pred_proba(feature_extractor, ensemble_model, tabular_raw, device):
    """Get prediction probability from tabular data"""
    if feature_extractor is None or ensemble_model is None:
        print("⚠️ Tabular prediction unavailable: missing models")
        return None
    
    try:
        feats = get_tabular_features(feature_extractor, tabular_raw, device)
        if feats is None:
            return None
        
        proba = ensemble_model.predict_proba(feats)[:, 1]
        return proba
    except Exception as e:
        print(f"Error getting tabular prediction: {e}")
        return None

def get_image_damage_score(image_model, image_path, device):
    """Get damage score from image"""
    if image_model is None:
        print("⚠️ Image prediction unavailable: missing image model")
        return 0.0
    
    try:
        # Handle both file path and PIL Image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        elif hasattr(image_path, 'convert'):  # PIL Image
            image = image_path.convert("RGB")
        else:
            print("Invalid image input type")
            return 0.0
            
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = image_model(image_tensor)[0]
        damage_scores = [output['scores'][i].item() for i, label in enumerate(output['labels']) if label.item() in [1, 2, 3]]
        return max(damage_scores) if damage_scores else 0.0
    except Exception as e:
        print(f"Error processing image: {e}")
        return 0.0

def fused_prediction(feature_extractor, ensemble_model, image_model, tabular_raw, image_path, device, alpha=0.5):
    """Perform fused prediction using both tabular and image data"""
    try:
        # Get tabular prediction
        proba_tab = get_tabular_pred_proba(feature_extractor, ensemble_model, tabular_raw, device)
        if proba_tab is None:
            proba_tab = 0.5  # Default fallback
            print("⚠️ Using default tabular prediction (0.5)")
        elif isinstance(proba_tab, np.ndarray):
            proba_tab = proba_tab[0]
        
        # Get image prediction
        proba_img = get_image_damage_score(image_model, image_path, device)
        if proba_img == 0.0:
            print("⚠️ Using default image prediction (0.0)")
        
        # Fuse predictions
        proba_fused = alpha * proba_tab + (1 - alpha) * proba_img
        pred = 1 if proba_fused > 0.5 else 0
        
        print(f"Fusion: tab={proba_tab:.3f}, img={proba_img:.3f}, fused={proba_fused:.3f}")
        return pred, proba_fused
    except Exception as e:
        print(f"Error in fused prediction: {e}")
        return 0, 0.5

# -------------------------------
# 9. Check system status
# -------------------------------
def get_system_status():
    """Get system status for debugging"""
    try:
        models_dir = get_models_dir()
        models_exist = os.path.exists(models_dir)
    except Exception as e:
        models_dir = f"Error: {e}"
        models_exist = False
    
    return {
        "sklearn_available": SKLEARN_AVAILABLE,
        "torch_available": torch.__version__ if 'torch' in globals() else False,
        "models_directory": models_dir,
        "models_directory_exists": models_exist
    }

# -------------------------------
# 10. Main Usage Example (for standalone testing)
# -------------------------------
if __name__ == "__main__":
    print("=== FUSION MODULE STATUS ===")
    
    # Configure Django settings for standalone execution
    try:
        import django
        from django.conf import settings
        if not settings.configured:
            settings.configure(
                DEBUG=True,
                BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            django.setup()
            print("✅ Django configured for standalone execution")
    except Exception as e:
        print(f"⚠️ Django configuration warning: {e}")
    
    status = get_system_status()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Sample CSV data (3 rows)
    csv_data = """
Month,WeekOfMonth,DayOfWeek,Make,AccidentArea,DayOfWeekClaimed,MonthClaimed,WeekOfMonthClaimed,Sex,MaritalStatus,Age,Fault,PolicyType,VehicleCategory,VehiclePrice,FraudFound_P,PolicyNumber,RepNumber,Deductible,DriverRating,Days_Policy_Accident,Days_Policy_Claim,PastNumberOfClaims,AgeOfVehicle,AgeOfPolicyHolder,PoliceReportFiled,WitnessPresent,AgentType,NumberOfSuppliments,AddressChange_Claim,NumberOfCars,Year,BasePolicy
Dec,5,Wednesday,Honda,Urban,Tuesday,Jan,1,Female,Single,21,Policy Holder,Sport - Liability,Sport,more than 69000,0,1,12,300,1,more than 30,more than 30,none,3 years,26 to 30,No,No,External,none,1 year,3 to 4,1994,Liability
Jan,3,Wednesday,Honda,Urban,Monday,Jan,4,Male,Single,34,Policy Holder,Sport - Collision,Sport,more than 69000,0,2,15,400,4,more than 30,more than 30,none,6 years,31 to 35,Yes,No,External,none,no change,1 vehicle,1994,Collision
Oct,5,Friday,Honda,Urban,Thursday,Nov,2,Male,Married,47,Policy Holder,Sport - Collision,Sport,more than 69000,0,3,7,400,3,more than 30,more than 30,1,7 years,41 to 50,No,No,External,none,no change,1 vehicle,1994,Collision
"""

    # Load raw CSV data into DataFrame
    df_raw = pd.read_csv(StringIO(csv_data))

    # Define file paths
    test_image_path = "test.webp"  # Replace with your actual image path

    print("\n=== LOADING MODELS ===")
    # Load preprocessing pipeline
    pipeline = load_preprocessing_pipeline("preprocessing_pipeline.pkl")
    
    if pipeline is not None:
        # Preprocess tabular data
        tabular_features = preprocess_tabular_data_with_pipeline(df_raw, pipeline)

        if tabular_features is not None:
            # Determine input feature size dynamically for model loading
            input_feature_size = tabular_features.shape[1]
            print(f"Input feature size: {input_feature_size}")

            # Load models
            feature_extractor = load_tabular_cnn_feature_extractor("cnn_model.pth", input_feature_size, device)
            ensemble_model = load_ensemble_model("ensemble_model.pkl")
            image_model = load_image_model("maskrcnn_damage_detection.pth", device)

            # Check if we have a valid image file for testing
            if os.path.exists(test_image_path):
                # Run fused prediction on first sample
                print("\n=== RUNNING PREDICTION ===")
                pred, proba = fused_prediction(feature_extractor, ensemble_model, image_model, tabular_features[0:1], test_image_path, device)
                print(f"\nFinal Result: Prediction={pred}, Confidence={proba:.4f}")
            else:
                print(f"⚠️ Test image not found: {test_image_path}")
                print("Skipping prediction test")
        else:
            print("Failed to preprocess tabular data")
    else:
        print("Failed to load preprocessing pipeline - will use fallback predictions")