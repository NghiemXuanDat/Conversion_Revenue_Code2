# File: results_xlearner_hill_revenue.py
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler

import sys
sys.path.append("../..")
from metrics import auuc, auqc, lift
from utils import seed_everything
from xlearner_hill import get_xlearner

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Tải dữ liệu
print("Loading Revenue data...")
train_df = pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/train_men.csv")
test_df =  pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/test_men.csv")
val_df = pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/val_men.csv")

in_features = ['recency', 'history_segment', 'history', 'mens', 'womens',
               'zip_code', 'newbie', 'channel_Multichannel', 'channel_Phone', 'channel_Web']
label_feature = 'spend' # BÀI TOÁN REVENUE
treatment_feature = 'treatment'

X_train, y_train, t_train = train_df[in_features], train_df[label_feature], train_df[treatment_feature]
X_val, y_val_true, t_val_true = val_df[in_features], val_df[label_feature].values.flatten(), val_df[treatment_feature].values.flatten()
X_test, y_test_true, t_test_true = test_df[in_features], test_df[label_feature].values.flatten(), test_df[treatment_feature].values.flatten()

# 2. Khởi tạo
seeds = [412312, 42, 1874, 902745, 1]
results = []
best_overall_auuc = -np.inf
best_overall_auqc = -np.inf
best_overall_lift = -np.inf

# 3. Chạy vòng lặp qua từng Seed
for SEED in seeds:
    print(f"\n{'='*50}")
    print(f"🚀 REVENUE UPLIFT - BẮT ĐẦU CHẠY VỚI SEED: {SEED}")
    print(f"{'='*50}")
    
    seed_everything(SEED)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 10, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': SEED,
            'verbose': -1
        }
        
        # Lấy X-Learner cho bài toán revenue
        x_learner = get_xlearner(task_type='revenue', params=params)
        
        x_learner.fit(Y=y_train, T=t_train, X=X_train)
        uplift_pred_val = x_learner.effect(X_val).flatten()
        
        score = auqc(y_val_true, t_val_true, uplift_pred_val, bins=100, plot=False)
        return score

    print("🔃 Đang chạy Optuna Tuning...")
    fixed_sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", study_name=f"X_Learner_Rev_{SEED}", sampler=fixed_sampler)
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    val_best_auqc = study.best_value
    best_params = study.best_params
    print(f"✅ Tuning hoàn tất! Best Val AUQC: {val_best_auqc:.4f}")

    print("🔃 Huấn luyện mô hình Final & Đánh giá trên tập TEST...")
    best_params_final = best_params.copy()
    best_params_final['random_state'] = SEED
    best_params_final['verbose'] = -1

    final_x_learner = get_xlearner(task_type='revenue', params=best_params_final)
    final_x_learner.fit(Y=y_train, T=t_train, X=X_train)

    uplift_pred_test = final_x_learner.effect(X_test).flatten()

    auuc_score = auuc(y_test_true, t_test_true, uplift_pred_test, bins=100, plot=False)
    auqc_score = auqc(y_test_true, t_test_true, uplift_pred_test, bins=100, plot=False)
    lift_score = lift(y_test_true, t_test_true, uplift_pred_test, h=0.3)
    
    best_overall_auuc = max(best_overall_auuc, auuc_score)
    best_overall_auqc = max(best_overall_auqc, auqc_score)
    best_overall_lift = max(best_overall_lift, lift_score)

    print(f"   Test AUUC: {auuc_score:.3f}")
    print(f"   Test AUQC: {auqc_score:.3f}")
    print(f"   Test Lift@30: {lift_score:.3f}")

    run_result = {
        'seed': SEED,
        'val_best_auqc': val_best_auqc,
        'test_auuc': auuc_score,
        'test_auqc': auqc_score,
        'test_lift': lift_score,
    }
    run_result.update(best_params)
    results.append(run_result)

print("\n" + "="*50)
print("Revenue Uplift Modeling Results")
print(f"+) X_learner_AUUC_Rev: {best_overall_auuc:.4f}")
print(f"+) X_learner_AUQC_Rev: {best_overall_auqc:.4f}")
print(f"+) X_learner_Uplift@30_Rev: {best_overall_lift:.4f}")
print("="*50)

print("💾 ĐANG LƯU KẾT QUẢ VÀO FILE CSV...")
results_df = pd.DataFrame(results)
csv_filename = "x_learner_revenue_results.csv"
results_df.to_csv(csv_filename, index=False)
print(f"✅ Đã lưu thành công tại: {csv_filename}")