from fastapi import FastAPI
import requests
import pandas as pd
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras import models
from keras import layers
from keras.layers import LSTM, Dense

def get_crypto_data(crypto_name):

    #sources API coingecko
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_name}/market_chart?vs_currency=usd&days=max&interval=daily"
    response = requests.get(url)
    data = response.json()['prices']

    #creates DataFrame with columns date and price
    df = pd.DataFrame(data, columns=['date', 'price'])
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)
    df.index = df.index.date

    df = df.groupby(df.index).max()
    return df

app = FastAPI() # we create a fastapi instance

# Define a root `/` endpoint with a '@' decorator
@app.get('/')
def index():
    # whenever I "get" the "/" root, I have connection: True as a result
    return {'connection': True}

# Creating a new endpoint called predict with params
@app.get('/product') # "get" endpoint again
def product(input_1, input_2):
    # returns the product of two inputs
    product = int(input_1) * int(input_2) # changing str to int
    return {'result': product} # passing a value without parameters for the time being

@app.get('/predict') # model prediction
def predict():

    df = get_crypto_data("bitcoin")

    print("coin gecko working")

    # Prepare data
    price_data = df['price'].values

    #How many previous prices the model will use to predict the target
    past_days = 7

    X = []
    y = []

    for i in range(len(price_data) - past_days):
        X.append(price_data[i:i+past_days])
        y.append(price_data[i+past_days])

    X = np.array(X)
    y = np.array(y)

    latest_prices = df['price'].values[-past_days:]
    X = np.append(X, latest_prices)[-past_days:]

    # Reshape X to match the input shape of the LSTM model
    X = X.reshape((1, past_days, 1))

    print("starting model")

    model = models.Sequential()
    model.add(layers.LSTM(64, activation='relu', input_shape = (past_days, 1)))
    model.add(layers.Dense(32, activation = "relu"))
    model.add(layers.Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics = ["mae"])

    # Use the trained model to predict the next price
    next_price = model.predict(X, verbose = 0)[0][0]

    return {'result': round(next_price)}

if __name__ == '__main__':
    print(predict())
