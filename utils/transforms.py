import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import PowerTransformer
import logging
logger = logging.getLogger(__name__)


def replace_outliers(arr, range, upper=True, lower=False):
    """_summary_
    For each element greater than the upper bound, replace with
    a value sampled from a uniform distribution within the range.
    """
    if not upper and not lower:
        return arr
    if upper:
        mask = arr > range[1]
        arr[mask] = np.random.uniform(range[0], range[1], size=mask.sum())
    if lower:
        mask = arr < range[0]
        arr[mask] = np.random.uniform(range[0], range[1], size=mask.sum())
    return arr


def remove_outliers(df, calo, sample):
    start_len = len(df)
    if calo == "eb":
        df = df[
            (df["probe_pt"] < 400)
            & (df["probe_r9"] < 1.)
            & (df["probe_s4"] > 0.4)
            & (df["probe_sieie"] < 0.016) & (df["probe_sieie"] > 0.005)
            & (df["probe_sieip"] < 0.0002) & (df["probe_sieip"] > -0.0002)
            & (df["probe_etaWidth"] < 0.03)
            & (df["probe_phiWidth"] < 0.21)
            & (df["probe_pfPhoIso03"] < 10.)
            & (df["probe_pfChargedIsoPFPV"] < 10.)
            & (df["probe_pfChargedIsoWorstVtx"] < 100.)
        ]
    elif calo == "ee":
        pass
    end_len = len(df)
    diff = start_len - end_len
    logger.info(f"Removed {diff/start_len*100:.3f}% of events from {sample} {calo} calo training set")
    logger.info(f"New length: {end_len}")

    return df


class IsoTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, std):
        self.std = std
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        zero_indices = np.where(X <= 0)[0]
        # replace 0s with values sampled from left half of gaussian
        X[zero_indices] = np.random.normal(
            0., self.std, len(zero_indices)
        ).reshape(-1, 1)
        while any(X[zero_indices] > 0.):
            positive_indices = np.where(X[zero_indices] > 0.)[0]
            X[zero_indices[positive_indices]] = np.random.normal(
                    0., self.std, len(positive_indices)
                ).reshape(-1, 1)
        return X 

    def inverse_transform(self, X, y=None):
        # set all values <= 0 to 0
        X[X <= 0.] = 0.
        return X


class CustomLog(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        greater_than_zero = np.where(X > 0)[0]
        X[greater_than_zero] = np.log1p(X[greater_than_zero])
        self.min_value = np.min(X[greater_than_zero])
        X[greater_than_zero] -= self.min_value
        return X

    def inverse_transform(self, X, y=None):
        X = X.copy()
        greater_than_zero = np.where(X > 0)[0]
        X[greater_than_zero] += self.min_value
        X[greater_than_zero] = np.expm1(X[greater_than_zero])
        return X


class CustomPT(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.potr = PowerTransformer(method="box-cox")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        greater_than_zero = np.where(X > 0)[0]
        X[greater_than_zero] = self.potr.fit_transform(X[greater_than_zero])
        self.min_value = np.min(X[greater_than_zero])
        X[greater_than_zero] -= self.min_value
        return X

    def inverse_transform(self, X, y=None):
        X = X.copy()
        greater_than_zero = np.where(X > 0)[0]
        X[greater_than_zero] += self.min_value
        X[greater_than_zero] = self.potr.inverse_transform(X[greater_than_zero])
        return X
