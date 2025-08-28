# make_stage13_project.py
"""
Stage 13 — Productization Demo
This script generates the full project structure:
- src/model_utils.py
- model/model.pkl
- notebooks/stage13_productization_demo.ipynb
- app.py (Flask API)
- requirements.txt
- README.md
- reports/stakeholder_summary.md
"""

import pickle
from pathlib import Path
import nbformat as nbf
import numpy as np
from sklearn.linear_model import LinearRegression
import os
BASE = Path(os.path.abspath(os.path.dirname(__file__)))
SRC = BASE / "src"
NB_DIR = BASE / "notebooks"
MODEL_DIR = BASE / "model"
REPORTS = BASE / "reports"
for d in [SRC, NB_DIR, MODEL_DIR, REPORTS]:
    d.mkdir(parents=True, exist_ok=True)

# --- src/model_utils.py ---
SRC.joinpath("model_utils.py").write_text('''\
import numpy as np
from sklearn.linear_model import LinearRegression

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_model(model, X):
    return model.predict(X)
''', encoding="utf-8")

# --- Train and pickle a simple model ---
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,6,8,10])
model = LinearRegression().fit(X,y)
pickle.dump(model, open(MODEL_DIR/"model.pkl", "wb"))

# --- notebooks/stage13_productization_demo.ipynb ---
nb = nbf.v4.new_notebook()
def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

nb.cells = [
    md("# Stage 13 — Productization Homework Demo"),
    md("Train a model, save it, reload it, and interact with the Flask API."),
    code("""\
import pickle
with open('../model/model.pkl','rb') as f:
    model = pickle.load(f)
print("Prediction for [[6]]:", model.predict([[6]]))
"""),
    code("""\
# Call the Flask API (make sure app.py is running)
import requests
url = "http://127.0.0.1:5000/predict"
payload = {"features":[[6]]}
r = requests.post(url, json=payload)
print("POST /predict response:", r.json())
"""),
    md("## Next Steps\n- Run `python app.py`\n- Use curl or requests to test endpoints"),
]
nbf.write(nb, NB_DIR/"stage13_productization_demo.ipynb")

# --- app.py ---
BASE.joinpath("app.py").write_text('''\
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
''', encoding="utf-8")

# --- requirements.txt ---
BASE.joinpath("requirements.txt").write_text('''\
flask
scikit-learn
matplotlib
numpy
pandas
requests
''')

# --- README.md ---
BASE.joinpath("README.md").write_text('''\
# Stage 13 — Productization Demo

## Structure
- src/: modular code
- model/: trained pickle model
- notebooks/: demo notebook
- reports/: stakeholder summary
- app.py: Flask API
- requirements.txt

## Running
1. pip install -r requirements.txt
2. python app.py
3. Call endpoints:
   - POST /predict with JSON {"features":[[val]]}
   - GET /predict/<x> or /predict/<x>/<y>
   - GET /plot

## Risks
- Simple toy model
- No authentication
''', encoding="utf-8")

# --- Stakeholder summary ---
REPORTS.joinpath("stakeholder_summary.md").write_text('''\
# Stakeholder Summary

**Objective:** Show how a model is productized.  
**Result:** Model trained, saved, served via Flask API.  
**Risks:** Toy model only, no security.  
**Next Steps:** Replace with real model, add tests and auth.
''', encoding="utf-8")

print("Stage 13 project created at:", BASE.resolve())
