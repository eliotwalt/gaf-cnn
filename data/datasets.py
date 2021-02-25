import torch
from torch.utils.data import Dataset
import numpy as np
# from pyts.image import GramianAngularField, MarkovTransitionField
from .transforms import Gasf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

class TimeSeriesRegressionDataset(Dataset):

    '''
    TimeSeriesRegressionDataset:

    Pytorch dataset to for timeseries to timeseries regression.
    '''

    def __init__(self, X, Y, 
                 x_transformation, y_transformation,
                 x_scaler, y_scaler):

        self.X = X
        self.Y = Y 
        self.x_transformation = x_transformation
        self.y_transformation = y_transformation
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler 
        self.tx = None
        self.ty = None
        
        self.get_transform()

    def get_transform(self):

        if self.x_transformation == 'gasf':
            self.tx = Gasf(scaler=self.x_scaler ,feature_range=(0,1))
        # add other reversible transformations ...
        else:
            raise ValueError(f'Only gramian angular summation field `gasf` is supported for this dataset. Received: {self.x_transformation}.')
        
        if self.y_transformation == 'gasf':
            self.ty = Gasf(scaler=self.y_scaler ,feature_range=(0,1))
        # add other reversible transformations ...
        else:
            raise ValueError(f'Only gramian angular summation field `gasf` is supported for this dataset. Received: {self.y_transformation}.')

    def show(self, y, y_pred, g, g_pred):
        
        '''
        y: true, untransformed multivariate timeseries
        y_pred: predicted, untransformed multivariate timeseries
        g: true, transformed multivariate timeseries
        g_pred: predicted, transformed multivariate timeseries
        '''

        for name, arr in [('ground truth', g) , ('prediction', g_pred)]:            
            fig = plt.figure(figsize=(20, 4))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(1, arr.shape[0]),
                             axes_pad=0.15,
                             share_all=True,
                             cbar_location="right",
                             cbar_mode="single",
                             cbar_size="7%",
                             cbar_pad=0.3,
                            )
            for subarr, ax in zip(arr, grid):
                im = ax.imshow(subarr, cmap='rainbow', origin='lower')
            ax.cax.colorbar(im)
            ax.cax.toggle_label(True)
            plt.suptitle(name, y=0.98, fontsize=16)
            plt.show()

        n = np.arange(y.shape[1])    
        for i, (yy, yy_pred) in enumerate(zip(y, y_pred)):
            plt.title(f'feature {i}')
            plt.plot(n, yy, 'r', label='target')
            plt.plot(n, yy_pred, 'bo--', label='pred')
            plt.legend()
            plt.show()

    def __len__(self):

        return self.X.shape[0]

    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.Y[idx]

        return (x, torch.Tensor(self.tx.transform(x)).float(),
                y, torch.Tensor(self.ty.transform(y)).float())

