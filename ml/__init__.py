import os

import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

from s3 import download_file

class Model(object):
    def __init__(self, raw_path, model_path, train_count = 200, train_steps = 60):
        self._raw_path = raw_path
        self._model_path = model_path
        self._df = None
        self._train_count = train_count
        self._train_steps = train_steps
        self._data = None
        self._X_train = None
        self._Y_train = None
        self._scaler = None
        self._model = None

        self.row_count = 0
        self.tip_avg = 0
        self.exists = os.path.isfile(self._model_path)
    
    def get_stats(self):
        if self.row_count == 0:
            self._load_data()

        return self.row_count, self.tip_avg

    def predict(self, data):
        if not self.exists: return None

        predicted_total = self._model.predict(data)
        return self._scaler.inverse_transform(predicted_total)

    def train(self):
        print('Loading RAW DATA ...')
        self._load_data()
        print('Preprocessing data ...')
        self._preprocess()
        print('Normalizing data ...')
        self._normalize()
        print('Preparing training datasets ...')
        self._prepare_datasets()
        print('Building model ...')
        self._build_model()
        print('Testing model ...')
        # self._test_model()
        print('Saving model ...')
        self._save()
        self.exists = True
        print('Training done!')

    def _load_data(self):
        if not os.path.isfile(self._raw_path):
            download_file(self._raw_path)

        self._df = pd.read_csv(self._raw_path)

        print(f'Tipos:\n{self._df.dtypes}')
        print(f'Cabecera:\n{self._df.head()}')
        print(f'Pie:\n{self._df.tail(7)}')
        self.row_count = self._df.shape[0]
        self.tip_avg = self._df["tip_amount"].mean()

    def _preprocess(self):
        self._df = self._df[['tpep_pickup_datetime', 'trip_distance', 'tip_amount', 'total_amount']]
        self._df['tpep_pickup_datetime'] = pd.to_datetime(self._df.tpep_pickup_datetime, format = '%Y-%m-%d %H:%M:%S')
        self._df.index = self._df['tpep_pickup_datetime']
        self._df = self._df.sort_index(ascending = True, axis = 0)
        print(f'Cabecera preparada:\n{self._df.head()}')
        print(f'Pie preparada:\n{self._df.tail(7)}')
    
    def _normalize(self):
        data = self._df.iloc[:, 1:2].values
        self._scaler = MinMaxScaler(feature_range = (0, 1))
        self._data = self._scaler.fit_transform(data)

    def _prepare_datasets(self):
        X_train, Y_train = [], []
        
        for i in range(60, 2035):
            X_train.append(self._data[i - 60:i, 0])
            Y_train.append(self._data[i, 0])
        
        self._X_train, self._Y_train = np.array(X_train), np.array(Y_train)
        self._X_train = np.reshape(self._X_train, (self._X_train.shape[0], self._X_train.shape[1], 1))
    
    def _build_model(self):
        self._model = Sequential()
        self._model.add( LSTM( 
            units = 50
            , return_sequences = True
            , input_shape = (self._X_train.shape[1], 1) 
        ) )
        self._model.add( Dropout(0.2) )

        self._model.add( LSTM(units = 50, return_sequences = True) )
        self._model.add( Dropout(0.2) )

        self._model.add( LSTM(units = 50, return_sequences = True) )
        self._model.add( Dropout(0.2) )

        self._model.add( LSTM(units = 50) )
        self._model.add( Dropout(0.2) )

        self._model.add( Dense( units = 1) )

        self._model.compile(
            loss = 'mean_squared_error'
            , optimizer = 'adam'
        )

        self._model.fit(self._X_train, self._Y_train
            , epochs = 100
            , batch_size = 32
            , verbose = 2
        )

    def _test_model(self):
        test_data = self._df[len(self._df) - 60:].values
        print[f'Test data:\n{test_data}']
        test_data = test_data.reshape[-1, 1]
        test_data = self._scaler.transform(test_data)
        X_test = []

        for i in range(60, 76):
            X_test.append(test_data[i - 60:i, 0])
        
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        prediction = self.predict(X_test)
        print(f'Predictions:\n{prediction}')

    def _save(self):
        self._model.save(self._model_path)

    def _load_model(self):
        self._model = load_model(self._model_path)
