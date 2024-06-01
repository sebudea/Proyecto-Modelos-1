# Proyecto-Modelos-1
## Isaac Jimenez y Sebastian Aristizabal
## Rossmann Store Sales
Competencia de Kaggle: https://www.kaggle.com/competitions/rossmann-store-sales/overview

Ejecute el notebook "01 - exploración" para generar datos y modelo de prueba inicial, para generar datos de test y train de muestra.

## ¿Como correr el contenedor?
Clona el repositorio https://github.com/sebudea/Proyecto-Modelos-1
Abre una terminal y situate en el proyecto
Ejecuta el comando docker build -t "nombre_del_ proyecto" para crear una imagen de Docker
Ejecuta el comando docker run -p 3000:3000 "nombre_del_ proyecto" para crear el contenedor

## Uso de la API

Para entrenar el modelo envíe una solicitud POST a /train para iniciar el entrenamiento del modelo.

Para predecir ventas envíe una solicitud POST a /predict con los datos necesarios para predecir las ventas.
Los datos deben incluir las características necesarias para la predicción, como el día de la semana, si la tienda está abierta, si hay promociones, etc.
