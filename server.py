from flask import Flask, request
import numpy as np
import pickle

model = pickle.load(open('model-tensor.pickle', 'rb'))
local_scaler = pickle.load(open('scaler-tensor.pickle', 'rb'))


app = Flask(__name__)


@app.route('/', methods=['GET'])
def homepage():
    return 'Homepage'


@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.get_json(force=True)
    rssi = request_data['rssi']
    A1 = request_data['A1']
    P1 = request_data['P1']

    prediction = model.predict(
        local_scaler.transform(np.array([[rssi, A1, P1]])))

    return 'The prediction is {}'.format(prediction)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
