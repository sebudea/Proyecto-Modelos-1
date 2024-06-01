# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY requirements.txt requirements.txt
COPY myrestapi.py myrestapi.py
COPY entrypoint.sh entrypoint.sh
COPY createModel.py createModel.py
COPY templates ./templates
COPY train_data.csv train_data.csv

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Dar permisos de ejecución al script de entrada
RUN chmod +x entrypoint.sh

# Exponer el puerto que usará la aplicación
EXPOSE 6000

# Especificar el comando para ejecutar la aplicación
ENTRYPOINT ["./entrypoint.sh"]