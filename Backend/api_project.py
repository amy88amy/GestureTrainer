from flask import Flask, Response, json, request
import logging

from prediction_project import prediction

app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def api_root():
    data = prediction(request.data)
    resp = Response(data, status=200, mimetype='application/text')
    logging.info(resp.status_code)
    return data
