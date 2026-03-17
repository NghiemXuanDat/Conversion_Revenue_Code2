import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler

import sys
sys.path.append("../..")
from metrics import auuc, auqc, lift
from utils import seed_everything
from rlearner_hill import get_rlearner_conversion

# Tắt log Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Loading data for Conversion...")
train_df = pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/train_men.csv")
test_df =  pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/test_men.csv")
val_df = pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/val_men.csv")

in_features = ['recency', 'history_segment', 'history', 'mens', 'womens',
               'zip_code', 'newbie', 'channel_Multichannel', 'channel_Phone', 'channel_Web']
label_feature = 'conversion' 
treatment_feature = 'treatment'

# Chuẩn bị X, y, treatment cho Train, Val, Test
X_train = train_df[in_features]
# Ép kiểu dữ liệu về int để econml nhận diện đúng bài toán phân loại (binary)
y_train = train_df[label_feature].astype(int) 
t_train = train_df[treatment_feature].astype(int)

X_val = val_df[in_features]
y_val_true = val_df[label_feature].astype(int).values.flatten()
t_val_true = val_df[treatment_feature].astype(int).values.flatten()

X_test = test_df[in_features]
y_test_true = test_df[label_feature].astype(int).values.flatten()
t_test_true = test_df[treatment_feature].astype(int).values.flatten()

seeds = [412312, 42, 1874, 902745, 1]
results = []
best_global_auuc = -np.inf
best_global_auqc = -np.inf
best_global_lift = -np.inf

for SEED in seeds:
    print(f"\n{'='*50}\n🚀 BẮT ĐẦU CHẠY VỚI SEED: {SEED}\n{'='*50}")
    seed_everything(SEED)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 10, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }
        
        # Khởi tạo R-Learner với param tuning
        r_learner = get_rlearner_conversion(params, SEED)
        
        # econml dùng (Y, T, X) thay vì (X, y, treatment)
        r_learner.fit(Y=y_train, T=t_train, X=X_train)
        
        # Lấy uplift score bằng hàm effect()
        uplift_pred_val = r_learner.effect(X_val).flatten()
        score = auqc(y_val_true, t_val_true, uplift_pred_val, bins=100, plot=False)
        return score

    print("🔃 Đang chạy Optuna Tuning...")
    study = optuna.create_study(direction="maximize", study_name=f"R_Learner_Conv_{SEED}", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print("🔃 Huấn luyện mô hình Final & Đánh giá trên tập TEST...")
    final_r_learner = get_rlearner_conversion(study.best_params, SEED)
    final_r_learner.fit(Y=y_train, T=t_train, X=X_train)

    uplift_pred_test = final_r_learner.effect(X_test).flatten()

    # Bật plot=True ở vòng test cuối để trực quan hóa đồ thị qini/uplift curve
    auuc_score = auuc(y_test_true, t_test_true, uplift_pred_test, bins=100, plot=True)
    auqc_score = auqc(y_test_true, t_test_true, uplift_pred_test, bins=100, plot=True)
    lift_score = lift(y_test_true, t_test_true, uplift_pred_test, h=0.3)
    
    print(f"   Test AUUC: {auuc_score:.3f}")
    print(f"   Test AUQC: {auqc_score:.3f}")
    print(f"   Test Lift: {lift_score:.3f}")

    best_global_auuc = max(best_global_auuc, auuc_score)
    best_global_auqc = max(best_global_auqc, auqc_score)
    best_global_lift = max(best_global_lift, lift_score)

    run_result = {'seed': SEED, 'val_best_auqc': study.best_value, 'test_auuc': auuc_score, 'test_auqc': auqc_score, 'test_lift': lift_score}
    run_result.update(study.best_params)
    results.append(run_result)

print("\n" + "="*50)
print("Conversion Uplift Modeling Results")
print(f"+) R_learner_AUUC_Con: {best_global_auuc:.4f}")
print(f"+) R_learner_AUQC_Con: {best_global_auqc:.4f}")
print(f"+) R_learner_Uplift@30_Con: {best_global_lift:.4f}")

pd.DataFrame(results).to_csv("r_learner_conversion_results.csv", index=False)
print("✅ Đã lưu kết quả!")