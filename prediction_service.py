import os
import io
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

# Cargar el modelo de IA
model_path = os.path.join(os.path.dirname(__file__), '..', 'pettrace-backend', 'models', 'model.h5')
model = load_model(model_path)

# Mapeo de las razas
breed_map = {
    0: 'Afgano', 1: 'Akita', 2: 'Alaskan Malamute', 3: 'Basenji', 4: 'Basset Hound',
    5: 'Beagle', 6: 'Bearded Collie', 7: 'Bichon Frise', 8: 'Border Collie', 9: 'Border Terrier',
    10: 'Borzoi', 11: 'Boston terrier', 12: 'Boxer', 13: 'Bulldog', 14: 'Bullmastiff',
    15: 'Cairn Terrier', 16: 'Cane Corso', 17: 'Caniche', 18: 'Cavalier King Charles Spaniel', 19: 'Chihuahua',
    20: 'Chow chow', 21: 'Cocker spaniel', 22: 'Collie', 23: 'Dalmata', 24: 'Doberman pinscher',
    25: 'Dogo Argentino', 26: 'Dogue de Bourdeaux', 27: 'Fox Terrier', 28: 'Galgo espanol', 29: 'GoldenRetriver',
    30: 'GranDanes', 31: 'Greyhound', 32: 'Grifon de Bruselas', 33: 'Havanese', 34: 'Husky',
    35: 'Irish Setter', 36: 'Jack russel terrier', 37: 'Keeshond', 38: 'Kerry Blue Terrier', 39: 'Komondor',
    40: 'Kuvasz', 41: 'Labrador retriever', 42: 'Lhasa Apso', 43: 'Maltes', 44: 'Mastin Napolitano',
    45: 'Mastin tibetano', 46: 'Norfolk terrier', 47: 'Norwich Terrier', 48: 'Papillon', 49: 'Pastor Aleman',
    50: 'Pequines', 51: 'Perro de agua portugues', 52: 'Perro de montana de Berna', 53: 'Perro lobo de saarloos', 54: 'Pinscher miniatura',
    55: 'PitBull', 56: 'Pomerania', 57: 'Presa canario', 58: 'Pug', 59: 'Rat terrier',
    60: 'Rottweiler', 61: 'Saluki', 62: 'Samoyedo', 63: 'San bernardo', 64: 'Schipperke',
    65: 'Schnauzer', 66: 'Setter Inglés', 67: 'Shar pei', 68: 'Shiba inu', 69: 'Shih Tzu',
    70: 'Staffordshire bull terrier', 71: 'Yorkshire terrier'
}

def preprocess_image(img):
    """ Preprocesa la imagen para que sea compatible con el modelo """
    img_bytes = img.read()  # Leer los bytes de la imagen del objeto FileStorage
    img = Image.open(io.BytesIO(img_bytes))  # Abrir imagen desde Bytes
    img = img.resize((224, 224))  # Redimensionar la imagen
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizar la imagen
    return img_array


def predict_breed(img):
    """ Realiza la predicción usando el modelo """
    img_array = preprocess_image(img)  # Preprocesar la imagen
    predictions = model.predict(img_array)[0]  # Obtener predicciones
    top_10_indices = predictions.argsort()[-10:][::-1]  # Obtener los 10 índices más altos
    top_10_breeds = [breed_map.get(i, 'Unknown') for i in top_10_indices]  # Obtener las razas
    return top_10_breeds

@app.route('/predict-breed/', methods=['POST'])
def predict_breed_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['file']  # Obtener el archivo de imagen

    try:
        image_bytes = image_file.read()  # Leer la imagen como bytes
        prediction = predict_breed(image_bytes)  # Realizar la predicción
        return jsonify({'top_10_breeds': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)