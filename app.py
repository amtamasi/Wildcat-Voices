import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sklearn

from src.models.random_model import RandomModel
from src.models.svm import SVM
from src.models.neural_net import NeuralNetModel

app = Flask(__name__)

rand_model = RandomModel()

# BUG: sklearn module error
svm_model = SVM()
svm_model.load_model()

# BUG: Threading issues between flask and tensorflow
#nn_model = NeuralNetModel()
#nn_model.load_model("model.json", "weights.h5")
#graph = nn_model.graph

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # NOTE: request.form.values() needs to be preprocessed audio sample
    #rand_prediction = RandomModel.predict()
    svm_prediction = svm_model.predict(request.form["audio-file"])

    # BUG: Threading issues between flask and tensorflow
    """
    global graph
    with graph.as_default():
        nn_features = nn_model.process_data(request.form["audio-file"])
        nn_prediction = nn_model.predict(nn_features)
    """

    # return prediction
    return render_template("index.html", prediction_text="Area of Origin: {}".format(svm_prediction))

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