import numpy as np
import pickle
import json
import pandas as pd
import hyper as hp

##### With spark
'''
from pyspark import SparkContext
sc = SparkContext(appName="...")
model = LinearRegressionModel.load(sc, "/home/hadoop/...")
'''

def _load_model():
	# Khởi tạo model
    with open('model\sme_model.sav', 'rb') as handle:
        model = pickle.load(handle)
    
    print("Load model complete!")
    return model
		
def _predict(sample,model):
   
    sample = pd.read_json(sample).fillna(0)
    result = pd.DataFrame(model.predict_proba(sample))
    # Thay với hyper
    result.columns = hp.CLASSES
    print(result)
    return result.to_json()

def _preprocess_data():
	pass


