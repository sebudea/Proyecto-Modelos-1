#!/bin/sh

# Debug: Imprimir el contenido del directorio actual
echo "Contenido del directorio /app:"
ls -l /app


# Ejecutar el script de creación del modelo
python /app/createModel.py

# Iniciar el servidor Flask
gunicorn --bind 0.0.0.0:6000 myrestapi:app