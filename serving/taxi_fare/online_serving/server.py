#from statistics import mode
import numpy as np
import hyper as hp
from flask import Flask, request
import flask
import json
import io
import utils


global model 
model = None
# Khởi tạo flask app
app = Flask(__name__)

@app.route("/")
def _hello_world():
	return "Hello world!	"


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    probs=''
	
    if request.json:
		
		# Lấy json request
        sample = json.dumps(flask.request.json)
		# Pre-processing
		
		# Dự báo phân phối xác suất
        probs = utils._predict(sample,model)
		# probability of classes
        data["success"] = True
    return probs

if __name__ == "__main__":
	print("App run!")
	# Load model
	model = utils._load_model()
	app.run(debug=False, host=hp.IP, threaded=False)