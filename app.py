import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from src.models.random_model import RandomModel
from src.models.svm import SVM
from src.models.neural_net import NeuralNetModel

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

rand_model = RandomModel()
svm_model = SVM()
NN_model = NeuralNetModel()
NN_model.load_model("model.json", "weights.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # NOTE: request.form.values() needs to be preprocessed audio sample
    #rand_prediction = RandomModel.predict()
    #svm_prediction = svm_model.predict(request.form.values())
    NN_prediction = NN_model.predict(request.form.values())

    # return prediction
    return render_template("index.html", prediction_text="Area of Origin: {}".format(NN_prediction))

    # NOTE: testing
    #return render_template("index.html", prediction_text="Area of Origin: {}".format("Kentucky"))

"""

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

"""

if __name__ == "__main__":
    app.run(debug=True)