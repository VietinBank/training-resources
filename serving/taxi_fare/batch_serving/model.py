# Pandas tương tác dữ liệu
import pandas as pd
# numpy
import numpy as np
# Sklearn components

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pickle,os

# Read data
data_url = 'https://raw.githubusercontent.com/VietinBank/training-resources/main/labs/data/taxi-trips.csv'

source_data = pd.read_csv(data_url)


def pre_process(data):
    # Đưa trường datetime về đúng định dạng
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])
    # Extract dayofweek, hourofday
    data['day'] = data['pickup_datetime'].dt.dayofweek
    data['hour'] = data['pickup_datetime'].dt.hour
    data = data.drop(['pickup_datetime','dropoff_datetime'],axis=1)
    data['fare_amount'] = data['fare_amount'] + data['tolls_amount']
    data = data.drop('tolls_amount',axis=1)
    data = data[data['passenger_count'] !=0].reset_index(drop=True)
    # impute null data
    data['rate_code'] = data['rate_code'].fillna(1)
    # calculate distance
    data['distance'] = minkowski_distance(data.pickup_longitude, data.dropoff_longitude, 
                                           data.pickup_latitude, data.dropoff_latitude, 1)
    return data

def feature_engineering(source_data):
    data = pre_process(source_data)
    # onehot
    onehot = OneHotEncoder(sparse=False)
    onehot_encoded_vendor= onehot.fit_transform(data[["vendor_id"]])
    data = data.drop('vendor_id',axis=1)
    df_vendor = pd.DataFrame(onehot_encoded_vendor,columns = onehot.get_feature_names_out(['vendor_id']))
    data = pd.concat([data,df_vendor],axis = 1)

    # bucketized
    step = 0.01
    to_bin = lambda x: np.floor(x / step) * step
    data["pickup_latitude"] = to_bin(data.pickup_latitude)
    data["pickup_longitude"] = to_bin(data.pickup_longitude)
    data["dropoff_latitude"] = to_bin(data.dropoff_latitude)
    data["dropoff_longitude"] = to_bin(data.dropoff_longitude)
    
    # feature cross
    data['day_hour'] = data.apply(lambda x: x['day']*x['hour'],axis =1)

    return data


def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)

def get_dataset(data):
    data = feature_engineering(source_data)
    num_columns = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','rate_code','passenger_count','fare_amount','day','hour','vendor_id_CMT','vendor_id_VTS','distance','day_hour']
    X = data[num_columns]
    y = data['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=69)

    return X_train, X_test, y_train, y_test

def get_loss(model,X_test,y_test):
    # print loss
    y_predict = model.predict(X_test)
    return mean_squared_error(y_test, y_predict)
def save_model(model,model_name):
    filename = "{}.sav".format(model_name)
    pickle.dump(model, open(filename, 'wb'))
    # Get the current working directory
    cwd = os.getcwd()

    print("Saved model {} at {}".format(filename,cwd))


if __name__ == "__main__":
    print("Running model.py")

    X_train, X_test, y_train, y_test = get_dataset(source_data)
    model = LinearRegression()
    # training model
    print("------start training-------")
    model.fit(X_train, y_train)
    print("------finish training-------")
    
    loss = get_loss(model,X_test, y_test)
    print("Loss value : {}".format(loss))
    # save model
    save_model(model,'taxi_fare')