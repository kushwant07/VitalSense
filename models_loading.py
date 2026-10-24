import config
import os
import sys
import torch
import xgboost as xgb
import lightgbm as lgb
from collections import OrderedDict

# Import the architecture
try:
    from models_definitions import RespPPGNet, TSCAN, HeartRateResNet
except ImportError:
    RespPPGNet = None
    TSCAN = None
    HeartRateResNet = None
    print("⚠️ models_definitions.py not found. CNN loading will fail.")

class ModelLoader:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚙️  Compute Device: {self.device}")

    def load_tscan(self):
        if not os.path.exists(config.TSCAN_PATH): return None
        try:
            loaded = torch.load(config.TSCAN_PATH, map_location=self.device)
            if isinstance(loaded, OrderedDict):
                if TSCAN is None: return None
                model = TSCAN().to(self.device)
                model.load_state_dict(loaded, strict=False)
            else:
                model = loaded
            model.eval()
            print("✅ TSCAN Model Loaded Successfully.")
            return model
        except Exception as e:
            print(f"❌ TSCAN Load Error: {e}")
            return None

    def load_resnet(self):
        if not os.path.exists(config.RESNET_PATH): return None
        try:
            loaded = torch.load(config.RESNET_PATH, map_location=self.device)
            if isinstance(loaded, OrderedDict):
                if HeartRateResNet is None: return None
                model = HeartRateResNet().to(self.device)
                model.load_state_dict(loaded, strict=False)
            else:
                model = loaded
            model.eval()
            print("✅ ResNet Model Loaded Successfully.")
            return model
        except Exception as e:
            print(f"❌ ResNet Load Error: {e}")
            return None

    def load_breathing_model(self):
        """Loads the RespPPGNet CNN"""
        path = config.BREATHING_MODEL_PATH
        if not os.path.exists(path):
            print("❌ Breathing model file missing.")
            return None
        
        if RespPPGNet is None:
            print("❌ Architecture definition missing.")
            return None

        try:
            # 1. Initialize Architecture
            model = RespPPGNet().to(self.device)
            
            # 2. Load Weights
            state_dict = torch.load(path, map_location=self.device)
            
            # Handle potential key mismatch (e.g. 'module.' prefix)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("module.", "") 
                new_state_dict[name] = v
                
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            print("✅ Breathing CNN Loaded Successfully.")
            return model
            
        except Exception as e:
            print(f"❌ Breathing Model Load Error: {e}")
            return None

    def load_stress_model(self):
        if not os.path.exists(config.STRESS_MODEL_PATH):
            print("❌ Stress model file missing.")
            return None
        try:
            model = lgb.Booster(model_file=config.STRESS_MODEL_PATH)
            print("✅ Stress Model Loaded Successfully.")
            return model
        except Exception as e:
            print(f"❌ Stress Model Load Error: {e}")
            return None