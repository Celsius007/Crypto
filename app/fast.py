from fastapi import FastAPI

app = FastAPI() # we create a fastapi instance

# Define a root `/` endpoint with a '@' decorator
@app.get('/')
def index():
    # whenever I "get" the "/" root, I have ok: True as a result
    return {'ok': True}

# Creating a new endpoint called predict with params
@app.get('/predict') # "get" endpoint again
def predict(day_of_week, time_of_day):
    # returns wait time as the product of week day and day time
    wait_time = int(day_of_week) * int(time_of_day) # changing str to int
    return {'wait_time': wait_time} # passing a value without parameters for the time being
