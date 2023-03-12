from fastapi import FastAPI

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

if __name__ == '__main__':
    print(product(1, 2))
