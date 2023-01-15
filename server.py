from flask import Flask, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)


@app.route('/')
def homepage():
    return "Homepages"


@app.route('/predict', methods=['POST'])
def predict():
    model = tf.keras.models.load_model('model')
    data = request.form.get('data')
    prediction = model.predict(data)
    score = tf.nn.softmax(prediction[0])
    return score


if __name__ == '__main__':
    app.run(port=3000, debug=True)
