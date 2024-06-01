import pandas as pd
import joblib

def load_model(filepath):
    return joblib.load(filepath)

def make_prediction(model, input_df):
    prediction = model.predict(input_df)
    return prediction

# Main execution
model = load_model('trained_model.pkl')
# Ejemplo de datos de entrada
train_data = pd.read_csv("train_data.csv")
train_data = train_data.drop(columns=["Sales"])
input_df = train_data.head(10)

print(input_df)

prediction = make_prediction(model, input_df)
print("Predicted Sales:", prediction)
