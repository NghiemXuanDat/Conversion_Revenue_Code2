# tlearner_hill.py
import optuna
import numpy as np
import pandas as pd
from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier, LGBMRegressor
from sklift.models import TwoModels

import sys

from metrics import auuc, auqc, lift

class TLearnerPipeline:
    def __init__(self, task_type='conversion'):
        """
        task_type: 'conversion' (dùng Classifier) hoặc 'revenue' (dùng Regressor)
        """
        self.task_type = task_type
        if self.task_type not in ['conversion', 'revenue']:
            raise ValueError("task_type phải là 'conversion' hoặc 'revenue'")

    def _get_base_estimator(self, params):
        """Khởi tạo Base Learner linh hoạt dựa trên bài toán"""
        if self.task_type == 'conversion':
            return LGBMClassifier(**params)
        else:
            return LGBMRegressor(**params)

    def optimize_and_train(self, X_train, y_train, t_train, 
                           X_val, y_val, t_val, 
                           X_test, y_test, t_test, 
                           seed):
        """Chạy Optuna tuning và trả về các metrics trên tập test"""
        
        def objective(trial):
            params = {
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
            
            # Với T-Learner, khởi tạo 2 model riêng biệt cho Treatment và Control
            est_trmnt = self._get_base_estimator(params)
            est_ctrl = self._get_base_estimator(params)
            
            t_learner = TwoModels(estimator_trmnt=est_trmnt, estimator_ctrl=est_ctrl, method='vanilla')
            t_learner.fit(X=X_train, y=y_train, treatment=t_train)
            
            # Predict trên tập Validation để tuning
            uplift_pred_val = t_learner.predict(X_val)
            score = auqc(y_val, t_val, uplift_pred_val, bins=100, plot=False)
            return score

        # 1. Tuning
        fixed_sampler = TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=fixed_sampler)
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        val_best_auqc = study.best_value
        best_params = study.best_params
        
        # 2. Huấn luyện Final Model với Best Params
        best_params['random_state'] = seed
        best_params['verbose'] = -1
        
        final_est_trmnt = self._get_base_estimator(best_params)
        final_est_ctrl = self._get_base_estimator(best_params)
        
        final_t_learner = TwoModels(estimator_trmnt=final_est_trmnt, estimator_ctrl=final_est_ctrl, method='vanilla')
        final_t_learner.fit(X=X_train, y=y_train, treatment=t_train)
        
        # 3. Đánh giá trên tập Test
        uplift_pred_test = final_t_learner.predict(X_test)
        
        test_auuc = auuc(y_test, t_test, uplift_pred_test, bins=100, plot=False)
        test_auqc = auqc(y_test, t_test, uplift_pred_test, bins=100, plot=False)
        test_lift = lift(y_test, t_test, uplift_pred_test, h=0.3)
        
        return {
            'val_best_auqc': val_best_auqc,
            'test_auuc': test_auuc,
            'test_auqc': test_auqc,
            'test_lift': test_lift,
            'best_params': best_params,
            'uplift_pred_test': uplift_pred_test # Lưu lại dự đoán để vẽ biểu đồ cho seed tốt nhất
        }