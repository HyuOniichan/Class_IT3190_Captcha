import time
import os
import numpy as np
import joblib

from models.model_selection import DIM_REDUCTION_CONFIG
from models.utils import apply_pca
from models.base import ModelBaseClass
from models.knn import ModelKNN
from models.decision_tree import ModelDecisionTree
from models.random_forest import ModelRandomForest
from models.svm import ModelSVM
from models.cnn import ModelCNN


def benchmark_model(X_train, X_test, model: ModelBaseClass, **kwargs):
    """
    Hàm đo thời gian huấn luyện, thời gian dự đoán và dung lượng của một mô hình.
    
    Params:
        train_data: Import from "train.npz".
        test_data: Import from "test.npz".
        model: Model instance.
        **kwargs: Params to run (train) the model.
        
    Returns:
        results: Dict chứa thông tin thời gian (giây) và dung lượng (MB)
    """
    print(f"[Benchmark] Start evaluate model...")

    # 1. Đo thời gian Huấn luyện (Training Time)
    start_train = time.perf_counter()
    model.run(**kwargs)
    end_train = time.perf_counter()
    train_time = end_train - start_train

    # 2. Đo thời gian Dự đoán (Inference Time) trên toàn bộ tập Test
    start_pred = time.perf_counter()
    y_pred = model.predict(X_test)
    end_pred = time.perf_counter()
    inference_time = end_pred - start_pred

    # 3. Tính toán dung lượng bộ nhớ tạm thời của mô hình (Model Size)
    # Ghi tạm mô hình ra file để đo kích thước byte thực tế, sau đó xóa file tạm đi
    temp_filename = "temp_model_benchmark.pkl"
    try:
        joblib.dump(model, temp_filename)
        model_size_bytes = os.path.getsize(temp_filename)
        model_size_mb = model_size_bytes / (1024 * 1024) # Đổi sang Megabytes
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    except Exception as e:
        print(f"[Benchmark] (Warning) Không thể tính dung lượng mô hình: {e}")
        model_size_mb = -1.0  # Trả về -1 nếu mô hình không hỗ trợ serialize bằng joblib (ví dụ như mạng CNN phức tạp)

    # Đóng gói kết quả đo lường
    metrics = {
        "train_time_sec": round(train_time, 4),
        "inference_time_sec": round(inference_time, 4),
        "model_size_mb": round(model_size_mb, 4)
    }
    
    print(f"[Benchmark] (result): " 
          f"Thời gian train: {metrics['train_time_sec']}s | "
          f"Thời gian test: {metrics['inference_time_sec']}s | "
          f"Kích thước: {metrics['model_size_mb']} MB")
          
    return metrics, y_pred




# Test Benchmark
data_dir = "output/dataset/lv0_emnist/build"
dim_reduction_method = "pca"
# model = ModelKNN()
# hyperparams = {
#     'inplace': True,
#     'k': 6, 
#     'distance_fn': "cosine"
# }
model = ModelSVM()
hyperparams = {
    'inplace': True,
    'model_type': 'LinearSVC',
    'C': 10.0
}

train_data = np.load(os.path.join(data_dir, "train.npz"))
test_data = np.load(os.path.join(data_dir, "test.npz"))

dim_reduction_config = DIM_REDUCTION_CONFIG[dim_reduction_method]
dim_reduction_func = dim_reduction_config["func"]

train_data, test_data, final_dims = dim_reduction_func(
    train_data, test_data,
    configs=dim_reduction_config
)

X_train, y_train = train_data['X'], train_data['y']
X_test, y_test = test_data['X'], test_data['y']

model.prepare(X_train, X_test, y_train, y_test)


benchmark_model(
    X_train, X_test, model, 
    **hyperparams
)

