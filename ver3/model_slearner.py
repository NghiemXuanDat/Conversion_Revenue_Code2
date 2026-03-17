# model_slearner.py
import optuna
from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier, LGBMRegressor
from sklift.models import SoloModel
from sklift.metrics import qini_auc_score
import numpy as np
import random
import os

# Tắt bớt log của Optuna để console không bị rối
optuna.logging.set_verbosity(optuna.logging.WARNING)

def seed_everything(seed):
    """Cố định seed cho môi trường để đảm bảo khả năng tái tạo."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def tune_and_train_s_learner(X_train, y_train, t_train, X_val, y_val, t_val, task_type='conversion'):
    seeds = [412312, 42, 1874, 902745, 1]
    
    global_best_score = -float('inf')
    global_best_params = None
    global_best_seed = None

    for current_seed in seeds:
        print(f"\n{'='*50}")
        print(f"🚀 BẮT ĐẦU CHẠY TUNING S-LEARNER VỚI SEED: {current_seed}")
        print(f"{'='*50}")
        
        seed_everything(current_seed)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 10, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': current_seed,
                'verbose': -1
            }
            
            if task_type == 'conversion':
                base_model = LGBMClassifier(**params)
            else:
                base_model = LGBMRegressor(**params)
                
            s_learner = SoloModel(estimator=base_model)
            s_learner.fit(X=X_train, y=y_train, treatment=t_train)
            
            uplift_pred_val = s_learner.predict(X_val)
            # Dùng Qini AUC (AUQC) để làm metric đánh giá tuning
            score = qini_auc_score(y_val, uplift_pred_val, t_val)
            return score

        print("🔃 Đang chạy Optuna Tuning...")
        fixed_sampler = TPESampler(seed=current_seed)
        study_name = f"S_Learner_LGBM_Tuning_{current_seed}"
        study = optuna.create_study(direction="maximize", study_name=study_name, sampler=fixed_sampler)
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        val_best_auqc = study.best_value
        best_params = study.best_params
        print(f"✅ Tuning hoàn tất! Best Val AUQC: {val_best_auqc:.4f}")

        if val_best_auqc > global_best_score:
            global_best_score = val_best_auqc
            global_best_params = best_params.copy()
            global_best_seed = current_seed
            print(f"🏆 Cập nhật Global Best Model mới tại seed {current_seed} (AUQC: {global_best_score:.4f})")

    # Huấn luyện mô hình Final
    print("\n🔃 Huấn luyện mô hình Final S-Learner với Seed Tốt Nhất...")
    seed_everything(global_best_seed)
    
    global_best_params['random_state'] = global_best_seed
    global_best_params['verbose'] = -1
    
    if task_type == 'conversion':
        final_base_model = LGBMClassifier(**global_best_params)
    else:
        final_base_model = LGBMRegressor(**global_best_params)
        
    final_s_learner = SoloModel(estimator=final_base_model)
    final_s_learner.fit(X=X_train, y=y_train, treatment=t_train)
    
    return final_s_learner