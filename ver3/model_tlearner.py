# model_tlearner.py
from causalml.inference.meta import BaseTClassifier, BaseTRegressor
from xgboost import XGBClassifier, XGBRegressor

def get_t_learner(task_type='conversion'):
    if task_type == 'conversion':
        base_model = XGBClassifier(
            n_estimators=1000, learning_rate=0.0001, max_depth=10, 
            random_state=42, eval_metric='logloss'
        )
        return BaseTClassifier(learner=base_model)
    else:
        base_model = XGBRegressor(
            n_estimators=1000, learning_rate=0.0001, max_depth=10, 
            random_state=42, objective='reg:squarederror'
        )
        return BaseTRegressor(learner=base_model)