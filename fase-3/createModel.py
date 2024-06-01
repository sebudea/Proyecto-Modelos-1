from flask import Flask, jsonify, request, render_template
from loguru import logger
import joblib
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Cargar los datos desde el archivo CSV
data = pd.read_csv('train_data.csv')

# Separar las características (X) y la variable objetivo (y)
X = data.drop('Sales', axis=1)
y = data['Sales']

# Aplicar el modelo Lasso para la selección de características
feature_sel_model = SelectFromModel(Lasso(alpha=0.2, random_state=0))
feature_sel_model.fit(X, y)

# Obtener las características seleccionadas
selected_feat = X.columns[(feature_sel_model.get_support())]
X_selected = X[selected_feat]

# Eliminar columnas duplicadas
X_selected = X_selected.loc[:, ~X_selected.columns.duplicated()]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.33, random_state=44, shuffle=True)

# Entrenar el modelo de Gradient Boosting Regressor
model = GradientBoostingRegressor(alpha=0.3, n_estimators=320, learning_rate=0.9, max_depth=30)
model.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(model, 'model.joblib')
print("Model saved successfully.")
