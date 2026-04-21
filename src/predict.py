import numpy as np

def predict_crop(model, scaler, le, input_data):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)
    crop = le.inverse_transform(prediction)

    return crop[0]