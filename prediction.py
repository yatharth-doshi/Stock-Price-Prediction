import os
from flask import Flask, request, render_template
import requests
import joblib

def prediction(open_price, high_price, low_price, volume, model_name):

    file_name = 'models/' + model_name + '.joblib'

    loaded_model = joblib.load(file_name)
    y_pred = loaded_model.predict([[open_price, high_price, low_price, volume]])
    
    return round(y_pred[0], 6)

def file_name():

    model_dir = 'models/'
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    model_names = []

    for file in model_files:
        name = os.path.splitext(file)[0]
        model_names.append(name)

    #print(model_names)
    return model_names

app = Flask(__name__)

# Define the home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stockprice')
def stockprice():
    return render_template('stockprice.html', models = file_name())

@app.route('/predict', methods=['POST'])
def predict():
    open_price = request.form['open']
    high_price = request.form['high']
    low_price = request.form['low']
    volume = request.form['volume']
    model_name = request.form['stock_name']
    result = prediction(open_price, high_price, low_price, volume, model_name)
    return render_template('predict.html', results = result, stock_name = model_name)

@app.route('/contactus')
def contactus():
    return render_template('contactus.html')

if __name__ == '__main__':
    app.run()
