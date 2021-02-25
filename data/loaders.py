import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .datasets import TimeSeriesRegressionDataset

def flatten_(x):
        
    s = np.zeros((x.shape[0]*x.shape[1], x.shape[2]))
    
    for i, u in enumerate(x):
        s[i:i+x.shape[1]] = u

    return s

class Loader:

    def __init__(self, n_timestamps, test_size, val_size, seed, x_transformation, y_transformation):

        self.n_timestamps = n_timestamps
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.x_transformation = x_transformation
        self.y_transformation = y_transformation


    def generic_processing(self, RX, RY):

        '''
        After reading, all datasets are processed the same ...
        '''

        # Build tensors
        X = np.zeros((int(RX.shape[0]/self.n_timestamps), self.n_timestamps, RX.shape[1]))
        Y = np.zeros((int(RY.shape[0]/self.n_timestamps), self.n_timestamps, RY.shape[1]))
        i = 0
        while i < X.shape[0]:
            X[i] = RX[i:i+self.n_timestamps]
            Y[i] = RY[i:i+self.n_timestamps]
            i+=1

        # Rearrange axis order
        X = np.transpose(X, axes=(0, 2, 1))
        Y = np.transpose(Y, axes=(0, 2, 1))

        # Cross validation split
        X_tmp, X_test, Y_tmp, Y_test = train_test_split(
            X, Y, shuffle=True, test_size=self.test_size, random_state=self.seed
        )
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_tmp, Y_tmp, shuffle=True, test_size=self.val_size, random_state=self.seed
        )

        # Scalers
        x_scaler = MinMaxScaler().fit(flatten_(X_train))
        y_scaler = MinMaxScaler().fit(flatten_(Y_train))

        # Pytorch datasets
        train_set = TimeSeriesRegressionDataset(X=X_train, Y=Y_train,
                                                x_transformation=self.x_transformation,
                                                y_transformation=self.y_transformation,
                                                x_scaler=x_scaler,
                                                y_scaler=y_scaler)
        val_set = TimeSeriesRegressionDataset(X=X_train, Y=Y_train,
                                            x_transformation=self.x_transformation,
                                            y_transformation=self.y_transformation,
                                            x_scaler=x_scaler,
                                            y_scaler=y_scaler)   
        test_set = TimeSeriesRegressionDataset(X=X_train, Y=Y_train,
                                            x_transformation=self.x_transformation,
                                            y_transformation=self.y_transformation,
                                            x_scaler=x_scaler,
                                            y_scaler=y_scaler)   
            
        return train_set, val_set, test_set

    def air_quality(self):

        # Read
        r = os.path.join(os.path.dirname(__file__), 'air_quality_uci')
        path = os.path.join(r, 'AirQualityUCI.csv')
        df = pd.read_csv(path)
        df.drop(['Date', 'Time', 'AH'], axis=1, inplace=True)

        # Na
        df.replace(to_replace=-200, value=np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)

        # Features /  Targets
        features = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)',
                    'NOx(GT)', 'NO2(GT)', 'T', 'RH']
        targets = ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
                'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)',
                'PT08.S5(O3)']
        
        df_X = df.drop(targets, axis=1)
        df_Y = df.drop(features, axis=1)

        RX = df_X.to_numpy()
        RY = df_Y.to_numpy()

        return self.generic_processing(RX, RY)
