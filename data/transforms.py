import numpy as np

class Gasf:

    def __init__(self, scaler, feature_range):

        self.scaler = scaler
        self.feature_range = feature_range

    def gasf(self, s_cos, s_sin): return np.outer(s_cos, s_cos) - np.outer(s_sin, s_sin)

    def gasf_invert(self, z): return np.sqrt((z+1)/2)

    def transform(self, S):

        n_attributes, n_timestamps = S.shape
        ss = np.zeros((n_attributes, n_timestamps, n_timestamps))
        S_scaled = self.scaler.transform(S)
        # S_scaled = np.where(S_scaled >= self.feature_range[1], self.feature_range[1], S_scaled)
        # S_scaled = np.where(S_scaled <= self.feature_range[0], self.feature_range[0], S_scaled)

        for i, s in enumerate(S_scaled):
            s_cos = s 
            s_sin = np.sqrt(np.clip(1-s_cos**2, 0, 1))
            s = self.gasf(s_cos, s_sin)
            ss[i] = s
        
        return ss
    
    def invert(self, Z):

        Z = np.diagonal(Z, axis1=-2, axis2=-1)
        # Z = np.where(Z >= self.feature_range[1], self.feature_range[1], Z)
        # Z = np.where(Z <= self.feature_range[0], self.feature_range[0], Z)
        S = self.gasf_invert(Z)
        S = self.scaler.inverse_transform(S)

        return S
