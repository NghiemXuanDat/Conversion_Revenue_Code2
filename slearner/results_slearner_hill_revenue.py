# results_slearner_hill_revenue.py
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler

import sys
import matplotlib.pyplot as plt

# Import custom modules
sys.path.append("../..")
from ConRevCausalProject.slearner.metrics import auuc, auqc, lift
from ConRevCausalProject.slearner.utils import seed_everything
from slearner_hill import get_optuna_param_space, get_slearner_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

def main():
    print("Loading data for Revenue Uplift Modeling...")
    train_df = pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/train_men.csv")
    test_df =  pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/test_men.csv")
    val_df = pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/val_men.csv")

    in_features = ['recency', 'history_segment', 'history', 'mens', 'womens',
                   'zip_code', 'newbie', 'channel_Multichannel', 'channel_Phone', 'channel_Web']
    
    label_feature = 'spend' # BÀI TOÁN REVENUE (Regression)
    treatment_feature = 'treatment'

    X_train, y_train, t_train = train_df[in_features], train_df[label_feature], train_df[treatment_feature]
    X_val, y_val_true, t_val_true = val_df[in_features], val_df[label_feature].values.flatten(), val_df[treatment_feature].values.flatten()
    X_test, y_test_true, t_test_true = test_df[in_features], test_df[label_feature].values.flatten(), test_df[treatment_feature].values.flatten()

    seeds = [412312, 42, 1874, 902745, 1]
    results = []

    for SEED in seeds:
        print(f"\n{'='*50}\n🚀 BẮT ĐẦU CHẠY REVENUE VỚI SEED: {SEED}\n{'='*50}")
        seed_everything(SEED)

        def objective(trial):
            params = get_optuna_param_space(trial, SEED)
            # Quan trọng: Khởi tạo mô hình Hồi quy cho biến spend
            s_learner = get_slearner_model(task_type='regression', params=params)
            
            s_learner.fit(X=X_train, y=y_train, treatment=t_train)
            uplift_pred_val = s_learner.predict(X_val)
            
            score = auqc(y_val_true, t_val_true, uplift_pred_val, bins=100, plot=False)
            return score

        print("🔃 Đang chạy Optuna Tuning...")
        fixed_sampler = TPESampler(seed=SEED)
        study = optuna.create_study(direction="maximize", study_name=f"S_Learner_Rev_{SEED}", sampler=fixed_sampler)
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        print(f"✅ Tuning hoàn tất! Best Val AUQC: {study.best_value:.4f}")

        print("🔃 Huấn luyện Final & Đánh giá trên tập TEST...")
        best_params_final = study.best_params.copy()
        best_params_final.update({'random_state': SEED, 'verbose': -1})

        # Sử dụng Regression cho final model
        final_s_learner = get_slearner_model(task_type='regression', params=best_params_final)
        final_s_learner.fit(X=X_train, y=y_train, treatment=t_train)

        uplift_pred_test = final_s_learner.predict(X_test)

        # Bật plot để hiển thị đường cong cho cấu hình cuối cùng
        do_plot = True if SEED == seeds[-1] else False 
        
        if do_plot:
            print(f"\n📊 Đang vẽ biểu đồ AUUC và AUQC cho Revenue (Seed {SEED})...")

        auuc_score = auuc(y_test_true, t_test_true, uplift_pred_test, bins=100, plot=do_plot)
        auqc_score = auqc(y_test_true, t_test_true, uplift_pred_test, bins=100, plot=do_plot)
        lift_score = lift(y_test_true, t_test_true, uplift_pred_test, h=0.3)
        
        print(f"   Test AUUC: {auuc_score:.3f}")
        print(f"   Test AUQC: {auqc_score:.3f}")
        print(f"   Test Uplift@30: {lift_score:.3f}")

        run_result = {
            'seed': SEED,
            'val_best_auqc': study.best_value,
            'test_auuc': auuc_score,
            'test_auqc': auqc_score,
            'test_lift_30': lift_score
        }
        run_result.update(study.best_params)
        results.append(run_result)

    print("\n💾 ĐANG LƯU KẾT QUẢ VÀO FILE CSV...")
    results_df = pd.DataFrame(results)
    results_df.to_csv("s_learner_revenue_results.csv", index=False)
    print("✅ Đã lưu thành công tại: s_learner_revenue_results.csv")

if __name__ == "__main__":
    main()