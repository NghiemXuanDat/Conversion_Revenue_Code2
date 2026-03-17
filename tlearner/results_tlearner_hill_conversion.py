# results_tlearner_hill_conversion.py
import pandas as pd
import optuna
from tlearner_hill import TLearnerPipeline

import sys

from metrics import auuc, auqc
from utils import seed_everything

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Tải dữ liệu
print("Loading data for Conversion...")
train_df = pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/train_men.csv")
test_df =  pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/test_men.csv")
val_df = pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/val_men.csv")

in_features = ['recency', 'history_segment', 'history', 'mens', 'womens',
               'zip_code', 'newbie', 'channel_Multichannel', 'channel_Phone', 'channel_Web']
label_feature = 'conversion'
treatment_feature = 'treatment'

X_train, y_train, t_train = train_df[in_features], train_df[label_feature], train_df[treatment_feature]
X_val, y_val_true, t_val_true = val_df[in_features], val_df[label_feature].values.flatten(), val_df[treatment_feature].values.flatten()
X_test, y_test_true, t_test_true = test_df[in_features], test_df[label_feature].values.flatten(), test_df[treatment_feature].values.flatten()

# 2. Khởi tạo Pipeline cho Conversion
pipeline = TLearnerPipeline(task_type='conversion')
seeds = [412312, 42, 1874, 902745, 1]

results = []
best_overall_val_auqc = -float('inf')
best_seed_predictions = None
best_metrics = {}

# 3. Chạy vòng lặp qua từng Seed
for SEED in seeds:
    print(f"\n{'='*50}\n🚀 BẮT ĐẦU CHẠY VỚI SEED: {SEED}\n{'='*50}")
    seed_everything(SEED)
    
    run_output = pipeline.optimize_and_train(
        X_train, y_train, t_train,
        X_val, y_val_true, t_val_true,
        X_test, y_test_true, t_test_true,
        seed=SEED
    )
    
    print(f"✅ Tuning hoàn tất! Best Val AUQC: {run_output['val_best_auqc']:.4f}")
    print(f"   Test AUUC: {run_output['test_auuc']:.3f}")
    print(f"   Test AUQC: {run_output['test_auqc']:.3f}")
    print(f"   Test Lift@30: {run_output['test_lift']:.3f}")
    
    # Lưu lại kết quả tổng hợp
    run_result = {
        'seed': SEED,
        'val_best_auqc': run_output['val_best_auqc'],
        'test_auuc': run_output['test_auuc'],
        'test_auqc': run_output['test_auqc'],
        'test_lift': run_output['test_lift'],
    }
    run_result.update(run_output['best_params'])
    results.append(run_result)
    
    # Theo dõi seed có Val AUQC tốt nhất để in kết quả cuối cùng
    if run_output['val_best_auqc'] > best_overall_val_auqc:
        best_overall_val_auqc = run_output['val_best_auqc']
        best_metrics = {
            'auuc': run_output['test_auuc'],
            'auqc': run_output['test_auqc'],
            'lift': run_output['test_lift']
        }
        best_seed_predictions = run_output['uplift_pred_test']

# 4. Lưu CSV
results_df = pd.DataFrame(results)
csv_filename = "t_learner_conversion_results.csv"
results_df.to_csv(csv_filename, index=False)

# 5. Output hiển thị kết quả tốt nhất và vẽ biểu đồ theo format yêu cầu
print("\n" + "="*50)
print("Conversion Uplift Modeling Results")
print(f"+) T_learner_AUUC_Con: {best_metrics['auuc']:.4f}")
print(f"+) T_learner_AUQC_Con: {best_metrics['auqc']:.4f}")
print(f"+) T_learner_Uplift@30_Con: {best_metrics['lift']:.4f}")

print("\n📊 Đang vẽ biểu đồ AUUC và AUQC cho mô hình tốt nhất...")
# Hàm metrics của bạn đã tích hợp vẽ biểu đồ khi plot=True
auuc(y_test_true, t_test_true, best_seed_predictions, bins=100, plot=True)
auqc(y_test_true, t_test_true, best_seed_predictions, bins=100, plot=True)