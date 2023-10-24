import json
import argparse
import numpy as np
import pickle5 as pickle
import logging

logger = logging.getLogger(__name__)

import dask.dataframe as dd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
    )

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    Sampler,
    BatchSampler,
    Dataset,
    DataLoader,
    Subset,
    SubsetRandomSampler,
    random_split
    )

from chainedMorpher import *
from utils.plots import dump_main_plot
from utils.phoid import calculate_photonid_mva

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor') if torch.cuda.is_available() else print ('cpu')

class DataSet(Dataset):
    def __init__(self, path, columns):
        super(DataSet, self).__init__()
        self.path = path
        self.columns = columns
        self.dask_dataset = dd.read_parquet(path, columns=columns, engine='fastparquet')
        self.np_dataset = self.dask_dataset.values.compute()
        self.np_train = self.np_dataset[:50000]
        self.np_test = self.np_dataset[50000:100000]
        self.transformed_train = self.transform(self.dask_dataset, self.columns)[:50000]
        self.transformed_test = self.transform(self.dask_dataset, self.columns)[50000:100000]
        self.inverse_transformed_test = self.inverse_transform(self.transformed_test, self.columns)
        self.corrected_test = None

    def __getitem__(self, index):
        y = self.labels[index]
        x = self.samples[index]
        return x, y

    def transform(self, data, columns):
        with open(f'preprocess/pipelines_eb.pkl', 'rb') as f:
            pipelines = pickle.load(f)['pipe1']

            data_np = dict()
            for var, pipeline in pipelines.items():
                if var in columns:
                    data_np[var] = pipeline.transform(data[var].values.compute().reshape(-1, 1)).reshape(-1,1)

        return np.concatenate([data_np[v] for v in columns],axis=1)

    def inverse_transform(self, data, columns):
        with open(f'preprocess/pipelines_eb.pkl', 'rb') as f:
            pipelines = pickle.load(f)['pipe1']
            data = dd.from_array(data, columns=columns)
            data_np = dict()
            for var, pipeline in pipelines.items():
                if var in columns:
                    data_np[var] = pipeline.inverse_transform(data[var].values.compute().reshape(-1, 1)).reshape(-1,1)

        return np.concatenate([data_np[v] for v in columns],axis=1)

    def correct(self, trainer):

        correct = trainer.correctFull(self.transformed_test.copy())
        self.corrected_test = self.inverse_transform(correct, self.columns)


# variables that have to be morphed
shower_shapes = [
    'probe_r9',
    'probe_s4',
    'probe_sieie',
    'probe_sieip',
    'probe_etaWidth',
    'probe_phiWidth',
]
isolation = [
    'probe_pfPhoIso03',
    'probe_pfChargedIsoPFPV',
    'probe_pfChargedIsoWorstVtx',
]
extra = [
    'probe_energyRaw',
    'probe_eta',
    'probe_fixedGridRhoAll',
]
columns = shower_shapes + isolation

data = DataSet('preprocess/data_eb_train.parquet', columns)
mc = DataSet('preprocess/mc_eb_train.parquet', columns)

trainer = trainflows(
    mc.transformed_train.copy(),
    mc.transformed_test.copy(),
    data.transformed_train.copy(),
    data.transformed_test.copy(),
    iNLayers=3,
    iSeparateScale=False)

mc.correct(trainer)

with open('./preprocess/var_specs.json', 'r') as f:
    vars_config = json.load(f)
    # turn into a dict with name as key
    vars_config = {d['name']: d for d in vars_config}

fig_output_dir = './preprocess/qmnf_figures'
fig_output_dirs = []
# create directory if it doesn't exist
if not os.path.exists(fig_output_dir):
    os.makedirs(fig_output_dir)
fig_output_dirs.append(fig_output_dir)

for col in columns:
    print(f'plotting {col}')
    dump_main_plot(
        data.np_test[:, columns.index(col)],
        mc.np_test[:, columns.index(col)],
        vars_config[col],
        fig_output_dirs,
        'eb',
        mc_corr=mc.corrected_test[:, columns.index(col)],
        weights=None,
        extra_name='_test_pre')

# photon ID MVA
mc_extra = DataSet('preprocess/mc_eb_train.parquet', extra)
mccorr_test = np.hstack((mc.corrected_test, mc_extra.np_test))
mccorr_df_test = dd.from_array(mccorr_test, columns=columns+extra)

data_extra = DataSet('preprocess/data_eb_train.parquet', columns+extra)
mc_extra = DataSet('preprocess/mc_eb_train.parquet', columns+extra)

logger.info('Calculate photon ID MVA')
pho_id_data_test = calculate_photonid_mva(data_extra.dask_dataset, 'eb', var_range=(50000,100000))
pho_id_mc_test = calculate_photonid_mva(mc_extra.dask_dataset, 'eb', var_range=(50000,100000))
pho_id_mc_corr_test = calculate_photonid_mva(mccorr_df_test, 'eb')

dump_main_plot(
    pho_id_data_test,
    pho_id_mc_test,
    vars_config['probe_mvaID'],
    fig_output_dirs,
    'eb',
    mc_corr=pho_id_mc_corr_test,
    weights=None,
    extra_name='_test_reweighted')
