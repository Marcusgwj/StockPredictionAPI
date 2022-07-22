from flask import Flask, request, jsonify
from lstm import lstm
from lr import lr
from svr import svr
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=["GET"])
def home():
    return "Stock Prediction API"

@app.route('/lr', methods=["POST"])
def predict_lr():
    json = request.json
    return jsonify({"Result": lr(json["ticker"]).tolist()})


@app.route('/lstm', methods=["POST"])
def predict_lstm():
    json = request.json
    result = (lstm(json["ticker"]))
    return jsonify({"Result": result.tolist()})


@app.route('/svr', methods=["POST"])
def predict_svr():
    json = request.json
    result = (svr(json["ticker"]))
    return jsonify({"Result": result.tolist()})


if __name__ == "__main__":
    app.run()
