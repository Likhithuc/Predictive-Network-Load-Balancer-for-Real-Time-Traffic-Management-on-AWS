import json
import numpy as np
from tensorflow.keras.models import load_model

def model_fn(model_dir):
    model = load_model(f"{model_dir}/smart_scaling_model.h5")
    return model

def predict_fn(input_data, model):
    data = np.array(input_data['instances'])  # expects list of lists
    prediction = model.predict(data)
    return prediction.tolist()
