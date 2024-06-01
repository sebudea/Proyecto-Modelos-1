from flask import Flask, jsonify, request, render_template
from loguru import logger
import joblib
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import warnings
import random

warnings.filterwarnings("ignore")
app = Flask(__name__)

train_status = "not training"
model = None

# Cargar el archivo CSV y obtener ejemplos
def load_examples():
    try:
        data = pd.read_csv('train_data.csv')
        examples = []

        for _ in range(5):
            example = data.sample(1).to_dict(orient='records')[0]

            # Normalizar los nombres de las columnas booleanas según el archivo CSV
            boolean_columns = ['Open', 'Promo', 'SchoolHoliday']

            # Inicializar todos los campos booleanos a False
            for key in boolean_columns:
                example[key] = 0
            
            # Establecer no más de 2 campos booleanos como True aleatoriamente
            true_keys = random.sample(boolean_columns, 2)
            for key in true_keys:
                example[key] = 1
            
            examples.append(example)
        
        return examples
    except Exception as e:
        logger.error(f"Error loading examples: {e}")
        return []

examples = load_examples()

def _train():
    global train_status, model
    
    train_status = "training"
    logger.info("Training started.")
    
    def train_model(X, y):
        model = GradientBoostingRegressor(alpha=0.3, n_estimators=320, learning_rate=0.9, max_depth=30)
        model.fit(X, y)
        return model

    def save_model(model, filename):
        joblib.dump(model, filename)
        logger.info(f"Model saved to {filename}")

    try:
        data = pd.read_csv('train_data.csv')
        logger.info("CSV file loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        train_status = "not training"
        return

    try:
        # Crear un Dataframe sin la columna Sales y una variable con la columna Sales
        X = data.drop('Sales', axis=1)
        y = data['Sales']
       
        # Aplicar el modelo Lasso para buscar la característica
        feature_sel_model = SelectFromModel(Lasso(alpha=0.2, random_state=0))
        feature_sel_model.fit(X, y)
        
        # Obtener las características importantes
        selected_feat = X.columns[(feature_sel_model.get_support())]
        logger.info(f"Selected features: {selected_feat}")
        
        # Crear un Dataframe con las características seleccionadas
        X_selected = X[selected_feat]
       
        # Eliminar columnas duplicadas
        X_selected = X_selected.loc[:, ~X_selected.columns.duplicated()]
     
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.33, random_state=44, shuffle=True)
        
        model = train_model(X_train, y_train)
        joblib.dump(model, 'model.joblib')
        
        train_status = "trained"
        logger.info("Training completed.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        train_status = "not training"

@app.route("/train", methods=["POST"])
def train():
    try:
        _train()
        return jsonify({"message": "Training started successfully."}), 200
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("predict.html", examples=examples)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        global model
        if model is None:
            model = joblib.load('model.joblib')
        
        # Obtener datos del formulario
        data = request.form.to_dict()
        
        # Convertir datos a un DataFrame
        df = pd.DataFrame([data])
        
        # Convertir tipos de datos
        boolean_columns = ['Open', 'Promo', 'SchoolHoliday']
        
        # Convertir los valores booleanos a 1/0
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].map({"True": 1, "False": 0})
                if df[col].isnull().any():
                    logger.error(f"Error converting column {col} with value {df[col].values}")
                    return jsonify({"error": f"Invalid value for column {col}"}), 400
        
        # Convertir el resto de las columnas a enteros
        integer_columns = ['Store', 'DayOfWeek', 'Customers']
        for col in integer_columns:
            if col in df.columns:
                try:
                    df[col] = df[col].astype(int)
                except ValueError as e:
                    logger.error(f"Error converting column {col} with value {df[col].values}: {e}")
                    return jsonify({"error": f"Invalid value for column {col}"}), 400
        
        # Comprobamos que todas las columnas necesarias estén presentes
        required_columns = model.feature_names_in_
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            return jsonify({"error": f"Missing columns: {missing_cols}"}), 400
        
        # Ordenamos las columnas del dataframe de entrada según el modelo
        df = df[required_columns]
        
        predictions = model.predict(df)
        return render_template("predict.html", prediction=predictions[0], examples=examples)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

# Script de entrenamiento inicial
data = pd.read_csv('train_data.csv')

X = data.drop('Sales', axis=1)
y = data['Sales']

feature_sel_model = SelectFromModel(Lasso(alpha=0.2, random_state=0))
feature_sel_model.fit(X, y)

selected_feat = X.columns[(feature_sel_model.get_support())]
X_selected = X[selected_feat]
X_selected = X_selected.loc[:, ~X_selected.columns.duplicated()]

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.33, random_state=44, shuffle=True)

model = GradientBoostingRegressor(alpha=0.3, n_estimators=320, learning_rate=0.9, max_depth=30)
model.fit(X_train, y_train)

joblib.dump(model, 'model.joblib')
print("Model saved successfully.")
