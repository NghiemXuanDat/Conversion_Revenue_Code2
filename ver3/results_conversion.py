# results_conversion.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklift.metrics import uplift_auc_score, qini_auc_score, uplift_at_k, uplift_curve, qini_curve

from model_slearner import tune_and_train_s_learner
from model_tlearner import get_t_learner
from model_xlearner import get_x_learner
from model_rlearner import get_r_learner
from model_causalforest import get_causal_forest

def load_data():
    print("Loading data...")
    train_df = pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/train_men.csv")
    test_df =  pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/test_men.csv")
    val_df = pd.read_csv(r"/data/DatnxData/dataset/Hillstrom/Men/val_men.csv")
    
    in_features = ['recency', 'history_segment', 'history', 'mens', 'womens',
                   'zip_code', 'newbie', 'channel_Multichannel', 'channel_Phone', 'channel_Web']
    return train_df, val_df, test_df, in_features

def main():
    train_df, val_df, test_df, in_features = load_data()
    
    label_feature = 'conversion'
    treatment_feature = 'treatment'

    X_train, Y_train, T_train = train_df[in_features], train_df[label_feature], train_df[treatment_feature]
    X_val, Y_val, T_val = val_df[in_features], val_df[label_feature], val_df[treatment_feature]
    X_test, Y_test, T_test = test_df[in_features], test_df[label_feature], test_df[treatment_feature]

    uplift_preds = {}

    # 1. Tuning và huấn luyện S-Learner (Dynamic)
    s_model = tune_and_train_s_learner(X_train, Y_train, T_train, X_val, Y_val, T_val, task_type='conversion')
    uplift_preds['S_learner'] = s_model.predict(X_test)

    # 2. Huấn luyện các Meta-Learners còn lại (Static)
    static_models = {
        'T_learner': get_t_learner('conversion'),
        'X_learner': get_x_learner('conversion'),
        'R_learner': get_r_learner('conversion'),
        'Causal_forest': get_causal_forest('conversion')
    }

    print("\n🔃 Đang huấn luyện các Meta-Learner khác...")
    for name, model in static_models.items():
        if name == 'Causal_forest':
            model.fit(Y_train.values, T_train.values, X=X_train.values)
            uplift_preds[name] = model.effect(X_test.values).flatten()
        else:
            model.fit(X=X_train.values, treatment=T_train.values, y=Y_train.values)
            uplift_preds[name] = model.predict(X_test.values).flatten()

    # 3. In kết quả theo cấu trúc mẫu
    print("\nConversion Uplift Modeling Results")
    model_names_map = {
        'S_learner': 'S_learner',
        'T_learner': 'T_learner',
        'X_learner': 'X_learner',
        'R_learner': 'R_learner',
        'Causal_forest': 'Causal_forest'
    }

    for name, print_prefix in model_names_map.items():
        preds = uplift_preds[name]
        y_true = Y_test.values
        t_true = T_test.values
        
        auuc = uplift_auc_score(y_true, preds, t_true)
        auqc = qini_auc_score(y_true, preds, t_true)
        up30 = uplift_at_k(y_true, preds, t_true, strategy='overall', k=0.3)
        
        # Đặc biệt xử lý tên biến cho khớp chuẩn từng chữ theo yêu cầu
        if name == 'Causal_forest':
            print(f"+) {print_prefix}_AUUC_Con: {auuc:.4f}")
            print(f"+) {print_prefix}_AUQC_Con: {auqc:.4f}")
            print(f"+) Causal_forest_learner_Uplift@30_Con: {up30:.4f}")
        else:
            print(f"+) {print_prefix}_AUUC_Con: {auuc:.4f}")
            print(f"+) {print_prefix}_AUQC_Con: {auqc:.4f}")
            print(f"+) {print_prefix}_Uplift@30_Con: {up30:.4f}")

    # 4. Vẽ biểu đồ AUUC
    plt.figure(figsize=(10, 7))
    random_plotted = False
    for name, preds in uplift_preds.items():
        x, y = uplift_curve(Y_test.values, preds, T_test.values)
        plt.plot(x, y, label=name, linewidth=2)
        if not random_plotted:
            plt.plot([x[0], x[-1]], [y[0], y[-1]], color='black', linestyle='--', label='Random', zorder=1)
            random_plotted = True

    plt.title('Conversion Uplift: AUUC Curves', fontsize=14)
    plt.xlabel('Number of observations targeted')
    plt.ylabel('Cumulative Uplift')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('conversion_auuc_curve.png', dpi=300)
    plt.close()

    # 5. Vẽ biểu đồ AUQC
    plt.figure(figsize=(10, 7))
    random_plotted = False
    for name, preds in uplift_preds.items():
        x, y = qini_curve(Y_test.values, preds, T_test.values)
        plt.plot(x, y, label=name, linewidth=2)
        if not random_plotted:
            plt.plot([x[0], x[-1]], [y[0], y[-1]], color='black', linestyle='--', label='Random', zorder=1)
            random_plotted = True

    plt.title('Conversion Uplift: AUQC Curves', fontsize=14)
    plt.xlabel('Number of observations targeted')
    plt.ylabel('Cumulative Qini')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('conversion_auqc_curve.png', dpi=300)
    plt.close()

    print("\n+) Đã lưu thành công biểu đồ conversion_auuc_curve.png và conversion_auqc_curve.png")

if __name__ == "__main__":
    main()