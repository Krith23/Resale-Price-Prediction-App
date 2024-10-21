import joblib
from xgboost import XGBRegressor

def load_models():
    try:
        meta_model = joblib.load('models/meta_model.pkl')
        rf = joblib.load('models/random_forest_model.pkl')
        gb = joblib.load('models/gradient_boosting_model.pkl')
        
        # Load XGBoost model using its load_model method
        xgb = XGBRegressor()
        xgb.load_model('models/xgb_model.json')

        dt = joblib.load('models/decision_tree_model.pkl')
        scaler = joblib.load('models/scaler.pkl')

        return rf, gb, xgb, dt, meta_model, scaler
    except Exception as e:
        raise Exception(f"Error loading models: {e}")
