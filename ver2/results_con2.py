import pandas as pd
import numpy as np
from model2 import get_conversion_models, evaluate_and_plot

def main():
    # 1. Load data
    train = pd.read_csv('/data/DatnxData/dataset/Hillstrom/Men/train_men.csv')
    val = pd.read_csv('/data/DatnxData/dataset/Hillstrom/Men/val_men.csv')
    test = pd.read_csv('/data/DatnxData/dataset/Hillstrom/Men/test_men.csv')

    df_train = pd.concat([train, val], axis=0).reset_index(drop=True)
    df_test = test.copy()

    features = [c for c in df_train.columns if c not in ['treatment', 'spend', 'conversion', 'visit']]
    treatment_col = 'treatment'
    outcome_col = 'conversion'

    X_train = df_train[features].values
    y_train = df_train[outcome_col].values
    T_train = df_train[treatment_col].values

    X_test = df_test[features].values
    y_test = df_test[outcome_col].values
    T_test = df_test[treatment_col].values

    # Tính P(Treatment) thực tế
    p_train = np.full(X_train.shape[0], T_train.mean())

    # Ánh xạ T_train sang kiểu chuỗi tường minh để Causal Forest không bị crash
    T_train_cf = np.where(T_train == 0, 'control', 'treatment')

    # 2. Khởi tạo Models
    models = get_conversion_models()
    
    df_results = pd.DataFrame({
        outcome_col: y_test,
        treatment_col: T_test
    })

    # 3. Huấn luyện và Dự báo
    for name, model in models.items():
        print(f"Training {name} for Conversion Uplift...")
        try:
            if name == 'Causal_Forest':
                # Đưa mảng chuỗi T_train_cf vào thay vì T_train dạng số
                model.fit(X_train, treatment=T_train_cf, y=y_train)
                preds = model.predict(X_test)
                
                # Hàm predict trả về mảng 2 chiều hoặc 1 chiều tùy số lượng treatment. Xử lý an toàn:
                if len(preds.shape) > 1 and preds.shape[1] > 1:
                    df_results[name] = preds[:, 1] - preds[:, 0]
                else:
                    df_results[name] = preds.flatten()
                    
            elif name in ['X_Learner', 'R_Learner']:
                model.fit(X=X_train, treatment=T_train, y=y_train, p=p_train)
                preds = model.predict(X_test)
                df_results[name] = preds.flatten()
                
            else:
                model.fit(X=X_train, treatment=T_train, y=y_train)
                preds = model.predict(X_test)
                df_results[name] = preds.flatten()
                
        except Exception as e:
            print(f"Lỗi khi train {name}: {e}")

    # 4. Đánh giá và vẽ đồ thị
    evaluate_and_plot(df_results, outcome_col, treatment_col, context_name="Conversion_Uplift")

if __name__ == "__main__":
    main()