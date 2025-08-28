from flask import Flask, request, jsonify, send_file
import pickle, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

app = Flask(__name__)
with open("model/model.pkl","rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict_post():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error":"Missing 'features' key"}), 400
    X = np.array(data["features"])
    preds = model.predict(X).tolist()
    return jsonify({"predictions":preds})

@app.route("/predict/<float:x>", methods=["GET"])
def predict_get1(x):
    pred = model.predict([[x]]).tolist()
    return jsonify({"prediction":pred})

@app.route("/predict/<float:x>/<float:y>", methods=["GET"])
def predict_get2(x,y):
    pred = model.predict([[x+y]]).tolist()
    return jsonify({"prediction":pred})

@app.route("/plot", methods=["GET"])
def plot():
    xs = np.linspace(0,10,50).reshape(-1,1)
    ys = model.predict(xs)
    plt.figure()
    plt.plot(xs,ys,label="Prediction")
    plt.legend(); plt.xlabel("x"); plt.ylabel("y")
    img_path = Path("reports/plot.png")
    plt.savefig(img_path)
    return send_file(img_path, mimetype="image/png")

if __name__=="__main__":
    app.run(debug=True)
