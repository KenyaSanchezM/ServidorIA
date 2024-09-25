from flask import Flask, request, jsonify
from prediction_service import predict_breed
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Ruta para hacer la predicción de raza de perro
@app.route('/predict-breed/', methods=['POST'])
def predict_breed_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['file']
    prediction = predict_breed(image)
    
    return jsonify({'breed': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
