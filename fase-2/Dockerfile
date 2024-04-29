# Usar una imagen base oficial de Python
FROM python:3.8-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos de requisitos e instalar las dependencias
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar los scripts de Python y otros archivos necesarios
COPY train_data.csv predict.py train.py entrypoint.sh ./

# Dar permisos de ejecución al script
RUN chmod +x entrypoint.sh

RUN pip install openpyxl

# Establecer el script como punto de entrada
CMD ["./entrypoint.sh"]