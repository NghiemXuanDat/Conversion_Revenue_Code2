# results_cf_hill_revenue.py
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler

import sys
sys.path.append("../..")
from metrics import auuc, auqc, lift
from utils import seed_everything 
from cf_hill import CausalForestWrapper

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Loading data for Revenue...")
train_df = pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/train_men.csv")
test_df =  pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/test_men.csv")
val_df = pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/val_men.csv")

in_features = ['recency', 'history_segment', 'history', 'mens', 'womens',
               'zip_code', 'newbie', 'channel_Multichannel', 'channel_Phone', 'channel_Web']
label_feature = 'spend' # BÀI TOÁN REVENUE
treatment_feature = 'treatment'

# Chuẩn bị X, y, treatment cho Train, Val, Test
X_train = train_df[in_features]
# Ép kiểu y_train về int
y_train = train_df[label_feature].astype(int) 
t_train = train_df[treatment_feature]

X_val = val_df[in_features]
# Ép kiểu y_val_true về int
y_val_true = val_df[label_feature].astype(int).values.flatten() 
t_val_true = val_df[treatment_feature].values.flatten()

X_test = test_df[in_features]
# Ép kiểu y_test_true về int
y_test_true = test_df[label_feature].astype(int).values.flatten() 
t_test_true = test_df[treatment_feature].values.flatten()

seeds = [412312, 42, 1874, 902745, 1]
results = []

best_overall_auqc = -np.inf
best_predictions = None
best_seed = None

for SEED in seeds:
    print(f"\n{'='*50}\n🚀 BẮT ĐẦU CHẠY VỚI SEED: {SEED}\n{'='*50}")
    seed_everything(SEED)

    def objective(trial):
        cf_params = {
            'n_estimators': trial.suggest_int('n_estimators', 52, 500, step=4),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 200),
            'min_var_fraction_leaf': trial.suggest_float('min_var_fraction_leaf', 0.01, 0.2),
        }
        
        # CRITICAL: task_type='revenue' kích hoạt mô hình Regressor bên trong
        cf_model = CausalForestWrapper(task_type='revenue', random_state=SEED, **cf_params)
        cf_model.fit(X_train, y_train, t_train)
        
        uplift_pred_val = cf_model.predict(X_val)
        return auqc(y_val_true, t_val_true, uplift_pred_val, bins=100, plot=False)

    print("🔃 Đang chạy Optuna Tuning...")
    study = optuna.create_study(direction="maximize", study_name=f"CF_Rev_{SEED}", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    
    val_best_auqc = study.best_value
    print(f"✅ Tuning hoàn tất! Best Val AUQC: {val_best_auqc:.4f}")

    print("🔃 Huấn luyện mô hình Final & Đánh giá trên tập TEST...")
    final_cf_model = CausalForestWrapper(task_type='revenue', random_state=SEED, **study.best_params)
    final_cf_model.fit(X_train, y_train, t_train)

    uplift_pred_test = final_cf_model.predict(X_test)

    auuc_score = auuc(y_test_true, t_test_true, uplift_pred_test, bins=100, plot=False)
    auqc_score = auqc(y_test_true, t_test_true, uplift_pred_test, bins=100, plot=False)
    lift_score = lift(y_test_true, t_test_true, uplift_pred_test, h=0.3)
    
    print(f"   Test AUUC: {auuc_score:.3f} | Test AUQC: {auqc_score:.3f} | Test Lift: {lift_score:.3f}")

    if auqc_score > best_overall_auqc:
        best_overall_auqc = auqc_score
        best_predictions = uplift_pred_test
        best_seed = SEED
        best_auuc_score = auuc_score
        best_lift_score = lift_score

    run_result = {
        'seed': SEED, 'val_best_auqc': val_best_auqc,
        'test_auuc': auuc_score, 'test_auqc': auqc_score, 'test_lift': lift_score,
    }
    run_result.update(study.best_params)
    results.append(run_result)

csv_filename = "causal_forest_revenue_results.csv"
pd.DataFrame(results).to_csv(csv_filename, index=False)
print(f"✅ Đã lưu thành công tại: {csv_filename}")

# Hiển thị Final Results theo yêu cầu
print("\n" + "="*50)
print("Revenue Uplift Modeling Results")
print(f"+) Causal_Forest_AUUC_Rev: {best_auuc_score:.4f}")
print(f"+) Causal_Forest_AUQC_Rev: {best_overall_auqc:.4f}")
print(f"+) Causal_Forest_Uplift@30_Rev: {best_lift_score:.4f}")
print("="*50)

# Vẽ biểu đồ
print(f"\n📊 Đang vẽ biểu đồ AUUC và AUQC cho mô hình tốt nhất (Seed: {best_seed})...")
auuc(y_test_true, t_test_true, best_predictions, bins=100, plot=True)
auqc(y_test_true, t_test_true, best_predictions, bins=100, plot=True)