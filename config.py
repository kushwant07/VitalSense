import os
import sys
import numpy as np

# --- SILENCE LOGS & RUNTIME CONFIG ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['GLOG_minloglevel'] = '2'
os.environ["PYTHONMALLOC"] = "malloc"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

# NumPy 2.0+ compatibility
if not hasattr(np, 'trapz'):
    np.trapz = getattr(np, 'trapezoid', None)

# --- PYINSTALLER RESOURCE PATH FIX ---
def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    primary_path = os.path.join(base_path, relative_path)
    if os.path.exists(primary_path):
        return primary_path
    
    # Fallback to root or models directory if relative path varies
    basename = os.path.basename(relative_path)
    root_path = os.path.join(base_path, basename)
    if os.path.exists(root_path):
        return root_path
        
    models_path = os.path.join(base_path, "models", basename)
    if os.path.exists(models_path):
        return models_path

    return primary_path

# --- CONSTANTS ---
CAPTURE_FRAMES = 900 
BATCH_SIZE = 300
FS_TARGET = 30.0

DTYPE_BUFFER = np.dtype([
    ('fh', np.uint8, (72, 72, 3)), 
    ('time', np.float64)
])

# --- 📁 MODEL PATHS (UPDATED) ---
TSCAN_PATH = get_resource_path(os.path.join("models", "tscan_Official_best_EPO60.pth"))
RESNET_PATH = get_resource_path(os.path.join("models", "hr_resnet_denoise.pth"))
STRESS_MODEL_PATH = get_resource_path(os.path.join("models", "stress_model-2.txt"))

# NEW BREATHING MODEL (PyTorch)
BREATHING_MODEL_PATH = get_resource_path(os.path.join("models", "resp_ppg_model2.pt"))