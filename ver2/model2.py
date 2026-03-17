import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from causalml.inference.meta import (
    BaseSClassifier, BaseTClassifier, BaseXClassifier, BaseRClassifier,
    BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor
)
from causalml.inference.tree import UpliftRandomForestClassifier, CausalRandomForestRegressor
from causalml.propensity import ElasticNetPropensityModel
from causalml.metrics import auuc_score, qini_score, plot_gain, plot_qini

def get_conversion_models():
    """
    Khởi tạo 5 mô hình baseline cho bài toán Conversion Uplift.
    """
    # Bộ tham số tối ưu cho AUQC: Học chậm, cây nông, ép node lớn
    xgb_params = {
        'learning_rate': 0.01,       # Học chậm lại để ranking mượt hơn
        'n_estimators': 200,         # Tăng số cây để bù đắp learning rate
        'max_depth': 3,              # Cây rất nông để chống overfit nhiễu 1%
        'min_child_weight': 100,     # Ép mỗi lá phải chứa lượng lớn data
        'reg_lambda': 1.0,
        'use_label_encoder': False,
        'eval_metric': 'auc',        # Trực tiếp tối ưu hóa ranking
        'random_state': 42
    }
    
    base_clf = xgb.XGBClassifier(**xgb_params)
    effect_reg = xgb.XGBRegressor(learning_rate=0.01, n_estimators=200, max_depth=3, min_child_weight=100, random_state=42)

    models = {
        'S_Learner': BaseSClassifier(learner=base_clf, control_name=0),
        'T_Learner': BaseTClassifier(learner=base_clf, control_name=0),
        'X_Learner': BaseXClassifier(
            outcome_learner=base_clf, 
            effect_learner=effect_reg,
            control_name=0
        ),
        'R_Learner': BaseRClassifier(
            outcome_learner=base_clf, 
            effect_learner=effect_reg,
            control_name=0,
            n_fold=5 
        ),
        'Causal_Forest': UpliftRandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=500,     # Rất quan trọng: lá phải lớn để gom đủ lượng người chuyển đổi
            max_features='auto',
            evaluationFunction='KL', 
            control_name='control',   # Đổi thành chữ để triệt tiêu lỗi str/int của Cython
            random_state=42,
            n_jobs=-1
        )
    }
    return models

# (Phần code get_revenue_models, calculate_uplift_at_k, evaluate_and_plot giữ nguyên như bản chuẩn tôi đã cung cấp trước đó)
def calculate_uplift_at_k(df, uplift_col, treatment_col, outcome_col, k=0.3):
    df_sorted = df.sort_values(by=uplift_col, ascending=False).reset_index(drop=True)
    cutoff = int(len(df_sorted) * k)
    df_top_k = df_sorted.iloc[:cutoff]
    
    treat_mean = df_top_k[df_top_k[treatment_col] == 1][outcome_col].mean()
    ctrl_mean = df_top_k[df_top_k[treatment_col] == 0][outcome_col].mean()
    
    if pd.isna(treat_mean): treat_mean = 0
    if pd.isna(ctrl_mean): ctrl_mean = 0
        
    return treat_mean - ctrl_mean

def evaluate_and_plot(df_results, outcome_col, treatment_col, context_name):
    print(f"\n{'='*10} {context_name} Results {'='*10}")
    models = [col for col in df_results.columns if col not in [outcome_col, treatment_col]]
    
    for model in models:
        auuc = auuc_score(df_results, outcome_col=outcome_col, treatment_col=treatment_col)
        auqc = qini_score(df_results, outcome_col=outcome_col, treatment_col=treatment_col)
        uplift_30 = calculate_uplift_at_k(df_results, uplift_col=model, treatment_col=treatment_col, outcome_col=outcome_col, k=0.3)
        
        print(f"+) {model}_AUUC_{context_name[:3]}: {auuc[model]:.4f}")
        print(f"+) {model}_AUQC_{context_name[:3]}: {auqc[model]:.4f}")
        print(f"+) {model}_Uplift@30_{context_name[:3]}: {uplift_30:.4f}")

    plt.figure(figsize=(10, 6))
    plot_gain(df_results, outcome_col=outcome_col, treatment_col=treatment_col, normalize=True)
    plt.title(f'{context_name} - AUUC (Cumulative Gain)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{context_name}_AUUC.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plot_qini(df_results, outcome_col=outcome_col, treatment_col=treatment_col, normalize=True)
    plt.title(f'{context_name} - AUQC (Qini Curve)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{context_name}_AUQC.png')
    plt.show()