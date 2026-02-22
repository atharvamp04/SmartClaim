# Annotated Images Fix - Diagnostic Report

## Issue Found ✅ FIXED
**Field Name Mismatch:**
- Backend returns: `annotated_image_base64`
- Frontend was looking for: `image_base64`

## Fix Applied
Updated [frontend/src/app/claim/page.tsx](frontend/src/app/claim/page.tsx) annotated images tab to:
- Use correct field name: `img.annotated_image_base64`
- Added helpful error message if images unavailable
- Added debug info showing when data exists but image doesn't

## Why Annotated Images Might Still Be Missing

### Root Cause Options:

1. **CNN Model Not Loaded**
   - Check backend logs: `"cnn_model_loaded": false` in debug_info
   - Model file: `maskrcnn_damage_detection.pth` might not exist
   - Solution: Ensure model file is in `backend/models/` directory

2. **Model Loading Failure**
   - `load_image_model()` function returns None
   - Check backend logs for error messages during model initialization
   - GPU/CUDA issues or torch/torchvision compatibility

3. **Empty CNN Results**
   - Model loads but returns no detections (no damage box confidence > 0.5)
   - This is normal for undamaged images
   - Annotated image still won't exist but damage_percentage will be low

### Backend Code Path:
```
predict_claim()
  ↓
process_multiple_images_with_yolo()
  ↓
  if cnn_image_model is not None:  ← Check this
    ↓
    process_damage_detection_image()  ← Returns annotated_image_base64
```

## What to Check

### 1. Backend Debug Info
The response now includes:
```json
"debug_info": {
  "cnn_model_loaded": true/false,  // ← Check this value
  "device": "cuda or cpu"
}
```

If `cnn_model_loaded: false`, the CNN model didn't load.

### 2. Model File Check
In terminal:
```powershell
ls backend/models/maskrcnn_damage_detection.pth
```

### 3. Backend Logs During Startup
Look for:
```
🖼️ Loading CNN image model...
✅ CNN image model loaded
OR
⚠️ CNN image model failed to load
```

## Frontend Fallback
Even if `annotated_image_base64` is missing, you can now see:

### ✅ Automatic Fallback Display
When CNN annotated images are not available, the frontend now shows:
- **Damage Percentage** per image
- **Severity Level** (LOW/MEDIUM/HIGH)
- **Severity Score** (0.0 - 1.0)
- **Weighted Damage Score** 
- **Detection Results** count
- **All Damage Regions** with confidence and relative size

This fallback data comes from `multi_image_analysis.individual_images` which is always available.

### ✅ Additional Data Always Available
Even without annotated images:
- ✅ Damage percentage (from CNN analysis)
- ✅ Severity level
- ✅ Per-image metrics (in Metrics tab)
- ✅ CNN-based fraud probability
- ✅ All YOLO detection data
- ✅ Complete claim breakdown
- ✅ Fraud analysis

## How to Enable CNN Models

### Option 1: Load Model Files
```
backend/
  models/
    - maskrcnn_damage_detection.pth  ← Needs to be here
    - yolov8_car_damage.pt
    - yolov8_car_parts.pt
```

### Option 2: Debug Backend Startup
Add to Django settings to see detailed logs:
```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG',
    },
}
```

## Summary

✅ **Frontend Fixed** - Now uses correct field names
⏳ **Backend Issue** - CNN model may not be loading
📊 **Data Still Available** - Damage metrics visible even without annotated images
🔧 **Workaround** - Check `result.debug_info.cnn_model_loaded` to diagnose

### Next Steps:
1. Check `response.debug_info.cnn_model_loaded` value
2. If `false`: Check if `maskrcnn_damage_detection.pth` exists in `backend/models/`
3. If file missing: Either provide model or disable CNN requirement
4. Frontend will show helpful message explaining why images unavailable
