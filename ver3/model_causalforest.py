# model_causalforest.py
from econml.dml import CausalForestDML
from xgboost import XGBClassifier, XGBRegressor

def get_causal_forest(task_type='conversion'):
    """
    Khởi tạo mô hình CausalForestDML.
    LƯU Ý QUAN TRỌNG VỀ DML: 
    - model_y luôn phải là Regressor để nội bộ econml tính toán được residuals Y - E[Y|X].
    - Nếu là bài toán conversion (binary), dùng XGBRegressor với objective='binary:logistic' 
      để đầu ra khớp với kỳ vọng xác suất.
    """
    if task_type == 'conversion':
        # SỬA LỖI Ở ĐÂY: Thay XGBClassifier thành XGBRegressor + objective='binary:logistic'
        model_y = XGBRegressor(
            n_estimators=1000, 
            max_depth=10, 
            learning_rate=0.0001,
            random_state=42, 
            objective='binary:logistic'
        )
    else:
        # Đối với bài toán Revenue, outcome là số tiền liên tục
        model_y = XGBRegressor(
            n_estimators=1000, 
            max_depth=10, 
            learning_rate=0.0001,
            random_state=42, 
            objective='reg:squarederror'
        )
        
    # model_t (Treatment) vẫn là Classifier vì treatment của Hillstrom là nhị phân (0/1)
    # econml đã được thông báo qua tham số discrete_treatment=True ở dưới
    model_t = XGBClassifier(
        n_estimators=1000, 
        max_depth=10, 
        learning_rate=0.0001,
        random_state=42, 
        eval_metric='logloss'
    )
    
    return CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=2000,          # Số lượng cây trong Causal Forest
        min_samples_leaf=10,       # Giới hạn số mẫu ở lá để tránh Overfitting
        max_features='sqrt',
        random_state=42,
        discrete_treatment=True,   # Chỉ định T là biến rời rạc
        verbose=0
    )