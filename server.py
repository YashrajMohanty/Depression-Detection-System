from flask import Flask, after_this_request, jsonify, render_template
import flask.json as json
import numpy as np
import DepressoDetecto

app = Flask(__name__)
# flask --app server run

@app.route("/home", methods=['GET'])
def get_home():
    return render_template("DepressoWebHome.html")

@app.route("/depdet/<data>", methods=['GET'])
def get_js_data(data):
    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

    json_resp = json.loads(data)
    values = json_resp['values']
    model_num = json_resp['model']
    pred_json = request_handler(values, model_num)
    pred_json = jsonify(pred_json)
    return pred_json

@app.route("/depdet", methods=['GET'])
def get_depdet():
    return render_template("DepressoWebDepDet.html")


def request_handler(values, model_num):
    model_type = None
    values = np.array(values)
    values = values.reshape(1,-1)

    if model_num == 1:
        model_type = 'decision tree'
    elif model_num == 2:
        model_type = 'naive bayes'
    elif model_num == 3:
        model_type = 'svm'
    elif model_num == 4:
        model_type = 'knn'
    elif model_num == 5:
        model_type = 'random forest'
    elif model_num == 6:
        model_type = 'logistic regression'
    elif model_num == 7:
        model_type = 'dnn'

    prediction = DepressoDetecto.predict(values, model_type)
    prediction = float(prediction)
    return {'prediction' : prediction}