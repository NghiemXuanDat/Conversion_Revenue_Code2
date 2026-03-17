# slearner_hill.py
import lightgbm as lgb
from sklift.models import SoloModel

def get_optuna_param_space(trial, seed):
    """
    Định nghĩa không gian siêu tham số cho LightGBM thông qua Optuna.
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 10, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': seed,
        'verbose': -1
    }

def get_slearner_model(task_type, params):
    """
    Khởi tạo mô hình S-Learner dựa trên loại bài toán (Conversion vs Revenue).
    
    Parameters:
    - task_type (str): 'classification' cho Conversion, 'regression' cho Revenue.
    - params (dict): Dictionary chứa siêu tham số cho LightGBM.
    """
    if task_type == 'classification':
        # Bắt buộc cho biến mục tiêu nhị phân (Conversion)
        base_model = lgb.LGBMClassifier(**params)
    elif task_type == 'regression':
        # Bắt buộc cho biến mục tiêu liên tục (Spend/Revenue)
        base_model = lgb.LGBMRegressor(**params)
    else:
        raise ValueError("task_type phải là 'classification' hoặc 'regression'")
        
    # Bọc base_model bằng SoloModel (S-Learner) của sklift
    s_learner = SoloModel(estimator=base_model)
    return s_learner