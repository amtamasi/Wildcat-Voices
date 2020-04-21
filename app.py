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
<<<<<<< HEAD
NN_model = NeuralNetModel()
rand_model.load_model()
NN_model.load_model()
svm_model.load_model()
=======
svm_model.load_model()

# BUG: Threading issues between flask and tensorflow
#nn_model = NeuralNetModel()
#nn_model.load_model("model.json", "weights.h5")
#graph = nn_model.graph
>>>>>>> 891ca220821319da731d07081ce3e9d9ad5ee5f9

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file_name = request.form['audio-file']
    # NOTE: request.form.values() needs to be preprocessed audio sample
<<<<<<< HEAD
    rand_prediction = rand_model.predict()
    svm_features = svm_model.process_data(file_name)
    svm_prediction = svm_model.predict(svm_features)
    #NN_features = NN_predictions = NN_model.process_data(file_name)
    #NN_prediction = NN_model.predict(NN_features)

    # return prediction
    return render_template("index.html", random_text="Area of Origin: {}".format(rand_prediction), svm_test="Area of Origin: {}".format(svm_prediction))
=======
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
>>>>>>> 891ca220821319da731d07081ce3e9d9ad5ee5f9

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
