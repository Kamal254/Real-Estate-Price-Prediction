import json
import pickle
import numpy as np
__location = None
__data_columns = None
__model = None


def load_saved_artifacts():
    print("loding saved artifacts")
    global __data_columns
    global __location

    with open("./artifects/columns1.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __location = __data_columns[3:]
    global __model
    with open("./artifects/model1.pickle", 'rb') as f:
        __model = pickle.load(f)
    print("loading saved artifacts---> Done")


def get_location_names():
    return __location

def get_estimated_price(location, sqft, bhk, bath):
    try:
       loc_index = __data_columns.index(location.lower())

    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))  # this is an array of zeros of size no of columns
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    # print(x)
    return round(__model.predict([x])[0], 2)


if __name__=="util":
    load_saved_artifacts()
    print(get_estimated_price('1st phase jp nagar', 1000, 3, 3))
    print(get_location_names())