# model_xlearner.py
from causalml.inference.meta import BaseXClassifier, BaseXRegressor
from xgboost import XGBClassifier, XGBRegressor

def get_x_learner(task_type='conversion'):
    if task_type == 'conversion':
        outcome_learner = XGBClassifier(n_estimators=1000, max_depth=10, learning_rate=0.0001, random_state=42)
        effect_learner = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.0001, random_state=42)
        return BaseXClassifier(outcome_learner=outcome_learner, effect_learner=effect_learner)
    else:
        outcome_learner = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.0001, random_state=42)
        effect_learner = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.0001, random_state=42)
        return BaseXRegressor(outcome_learner=outcome_learner, effect_learner=effect_learner)