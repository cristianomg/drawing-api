import json
from flask import Flask, jsonify, request

from flask_cors import CORS

from model.CNN import CNN

labels = ['avião', 'banana', 'abelha',
              'xícara de café' , 'caranguejo', 'guitarra',
              'hamburger', 'coelho', 'caminhão',
              'guarda chuva']

cnn = CNN('./model/model.h5', labels)

app = Flask(__name__)

CORS(app)

@app.route("/api/predict", methods=['POST'])
def predict():
    raw_data = request.files['file']

    return jsonify(cnn.predict(raw_data))

@app.route("/api/test")
def test():
    return jsonify('ok');

if __name__ == '__main__':
    app.run(host='127.0.0.1', port='80', debug=False)
