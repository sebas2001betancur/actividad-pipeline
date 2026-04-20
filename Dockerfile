# imagen base de Python liviana
FROM python:3.11-slim

# directorio de trabajo dentro del contenedor
WORKDIR /app

# copiar dependencias primero (para aprovechar cache de Docker)
COPY requirements.txt .

# instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# copiar el resto del proyecto
COPY . .

# crear carpeta de outputs
RUN mkdir -p outputs

# comando que se ejecuta al correr el contenedor
CMD ["python", "main.py"]
