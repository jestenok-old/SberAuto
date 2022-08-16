from typing import Union

import dill
import pickle
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

with open('data/sberauto_pipe.pkl', 'rb') as file:
    pipe = dill.load(file)


class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: int
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: Union[str, None]
    utm_adcontent: Union[str, None]
    utm_keyword: Union[str, None]
    device_category: str
    device_os: Union[str, None]
    device_brand: str
    device_model: Union[str, None]
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    session_id: str
    result: float


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return pipe['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])

    y = pipe['model'].predict(df)

    return {
        'session_id': form.session_id,
        'result': 1 if y[0][0] >= 0.929 else 0
    }
