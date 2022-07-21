from flask import Flask,jsonify,request
import traceback

import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


data1=pd.read_csv('train_advanced.csv')
data3= pd.read_csv('final_updated - final_features_glucose -updated.csv')
#------------------------------adding glucose predict column
data=data1
data['glucose_predict'] = np.nan
data['glucose_predict'][0:len(data)-1] = data['Historic Glucose mg/dL'][1:len(data)]
data.fillna(method='ffill', inplace=True)
data['Time']=pd.to_datetime(data['Date'] + ' ' + data['Start time'])
cols = data.columns.tolist()

cols = cols[-1:] + cols[:-1]
data= data[cols]
del data['Date']
del data['Start time']

data2=pd.read_csv('test_advanced.csv')
data_=data2
data_['glucose_predict'] = np.nan
data_['glucose_predict'][0:len(data_)-1] = data_['Historic Glucose mg/dL'][1:len(data_)]
data_.fillna(method='ffill', inplace=True)
data_['Time']=pd.to_datetime(data_['Date'] + ' ' + data_['Start time'])
cols = data_.columns.tolist()

cols = cols[-1:] + cols[:-1]
data_= data_[cols]
del data_['Date']
del data_['Start time']
#del data_['Carbohydrates (grams)']

#split 
train_data = data
test_data = data_
#70:30

test_time = test_data['Time']
test_gl_value = test_data['glucose_predict']  
test_data.drop(columns = ['Time'], inplace = True)
train_data.drop(columns = ['Time'], inplace = True)
    
empty_train_col = [0]*len(train_data)
for i, item in enumerate(test_data.columns):
   if item not in train_data.columns:
       train_data.insert(i, item, empty_train_col)

empty_test_col = [0]*len(test_data)
for i, item in enumerate(train_data.columns):
    if item not in test_data.columns:
        test_data.insert(i, item, empty_test_col)
#test_data
X_data = train_data.drop(columns = ['glucose_predict'])
y_data = train_data[['glucose_predict']]
input_dim = X_data.shape[1]                              #-----no_of_features
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_x.fit(X_data)
scaler_y.fit(y_data)

###########################################################################################
app = Flask(__name__)
@app.route("/")
def hello():
    return "hey" 

@app.route('/Mypredict', methods=['POST'])
def Mypredict():
    print("a")
    ir=pickle.load(open('prediction_of_glucose_level.pkl', 'rb'))
    if ir :
        try:
            json=request.get_json()
            input=list(json[0].values())
            input=np.array([input,])
            scaled_input_data = scaler_x.transform(input)
            scaled_input_data = scaled_input_data.reshape(scaled_input_data.shape[0], 1, scaled_input_data.shape[1])
            prediction=ir.predict(scaled_input_data, batch_size = 32)
            scaled_prediction = scaler_y.inverse_transform(prediction)
            scaled_prediction = scaled_prediction[0]
            return jsonify({'prediction': str(scaled_prediction[0])})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model here to use')    


# if __name__ == '__main__':
#     app.run(threaded=False)