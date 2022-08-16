import dill
import pickle

import pandas as pd
from datetime import datetime

from keras.models import load_model

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def main():
    preprocessor = Pipeline(steps=[
        ('add_features', FunctionTransformer(add_features)),
        ('filter', FunctionTransformer(filter_data)),
        ('column_transformer', FunctionTransformer(_column_transformer)),
        ('delete_unimportant_columns', FunctionTransformer(delete_unimportant_columns)),
    ])

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', load_model('data/keras_seq.h5'))
    ])

    with open('data/sberauto_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'Car price prediction model',
                'author': 'Kamil Taigunov',
                'version': 1,
                'date': datetime.now(),
                'type': type(pipe.named_steps["classifier"]).__name__,
                # 'accuracy': score.mean()
            }
        }, file)


def filter_data(df):
    columns_to_drop = [
        'session_id',
        'client_id',

        'device_model',
        'utm_keyword',
        'device_screen_resolution',

    ]

    return df.drop(columns_to_drop, axis=1)


def add_features(df):
    import pandas as pd

    df['device_os'] = df['device_os'].fillna('other')
    df['device_brand'] = df['device_brand'].fillna('other')

    df['utm_campaign'] = df['utm_campaign'].fillna('LTuZkdKfxRGVceoWkVyg')
    df['utm_source'] = df['utm_source'].fillna('ZpYIoDJMcFzVoPFsHGJL')
    df['utm_adcontent'] = df['utm_adcontent'].fillna('JNHcPlZPxEMWDnRiyoBf')

    df['visit_date'] = pd.to_datetime(df['visit_date'])

    # обработка выбросов
    df.loc[df['visit_number'] > 4, 'visit_number'] = 4

    # непопулярные города
    unpopular_city = pd.read_csv('data/unpopular_city.csv')['0']
    df.loc[df['geo_city'].isin(unpopular_city), 'geo_city'] = 'other'

    # органический трафик
    df['is_organic'] = df.utm_medium.isin(['organic', 'referral', '(none)']).apply(lambda x: 1 if x else 0)

    # реклама в социальных сетях
    advertising_tags = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
                        'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw',
                        'gVRrcxiDQubJiljoTbGm']
    df['is_advertising'] = df['utm_source'].isin(advertising_tags).apply(lambda x: 1 if x else 0)

    # день недели
    df['dayofweek'] = df['visit_date'].dt.dayofweek.astype('str')

    # час посещения
    df['visit_time_hour'] = df['visit_time'].apply(lambda x: str(x.hour))

    # разрешение экрана
    df.loc[df['device_screen_resolution'] == '(not set)', 'device_screen_resolution'] = '414x896'
    df['device_screen_resolution_x'] = df['device_screen_resolution'].apply(lambda x: int(x.split('x')[0]))
    df['device_screen_resolution_y'] = df['device_screen_resolution'].apply(lambda x: int(x.split('x')[1]))

    return df


def _column_transformer(df):
    categorical = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
                   'device_os', 'device_brand', 'device_browser',
                   'geo_country', 'geo_city',
                   'dayofweek', 'visit_time_hour']
    numerical = ['visit_number', 'device_screen_resolution_x', 'device_screen_resolution_y']

    ohe = pickle.load(open("data/ohe.pickle", "rb"))
    data = ohe.transform(df[categorical])

    mm_scaler = pickle.load(open("data/mm_scaler.pickle", "rb"))
    data[numerical] = mm_scaler.transform(df[numerical])

    return data


def delete_unimportant_columns(df):
    important_columns = pd.read_csv('data/important_columns.csv')

    df = df[important_columns['0']]
    print(df.shape)
    print(df.columns)
    return df


if __name__ == '__main__':
    main()
