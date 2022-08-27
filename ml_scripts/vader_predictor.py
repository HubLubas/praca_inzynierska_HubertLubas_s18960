import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import pickle
from sklearn.preprocessing import scale


def predict_vader(file_name, model_name):
    news_df = pd.read_pickle(file_name)
    loaded_model = pickle.load(open(model_name, "rb"))
    prediction = loaded_model.predict(news_df)
    response = pd.DataFrame(prediction, columns=["Label"])
    return response.to_json(orient="table")


# TO DO ENDPOINT
def predict_vader_v2(data, model_filename="models/model.pkl"):
    # data = data.select_dtypes(include=["float64"])
    loaded_model = pickle.load(open(model_filename, "rb"))
    data = scale(data)
    prediction = loaded_model.predict(data)
    response = pd.DataFrame(prediction, columns=["Label"])
    return response
