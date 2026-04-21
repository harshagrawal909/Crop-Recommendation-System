from src.preprocessing import load_data, preprocess_data
from src.train import train_models
from src.predict import predict_crop

# Load + preprocess
df = load_data("data/data.csv")
X, y, scaler, le = preprocess_data(df)

# Train model
model = train_models(X, y)

# Example prediction
sample = [90, 42, 43, 21, 82, 6.5, 200]

result = predict_crop(model, scaler, le, sample)
print("Recommended Crop:", result)