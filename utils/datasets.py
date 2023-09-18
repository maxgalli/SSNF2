import torch
from torch.utils.data import Dataset
from torch import nn
from torch.distributed import init_process_group
import os
import numpy as np
import pandas as pd
from copy import deepcopy


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class ParquetDataset(Dataset):
    def __init__(
        self,
        parquet_file,
        context_variables,
        target_variables,
        device=None,
        pipelines=None,
        retrain_pipelines=False,
        rows=None,
    ):
        self.parquet_file = parquet_file
        self.context_variables = context_variables
        self.target_variables = target_variables
        self.all_variables = context_variables + target_variables
        data = pd.read_parquet(
            parquet_file, columns=self.all_variables + ["weight", "probe_energyRaw"], engine="fastparquet"
        )
        self.pipelines = pipelines
        if self.pipelines is not None:
            for var, pipeline in self.pipelines.items():
                if var in self.all_variables:
                    trans = (
                        pipeline.fit_transform
                        if retrain_pipelines
                        else pipeline.transform
                    )
                    data[var] = trans(data[var].values.reshape(-1, 1)).reshape(-1)
        if rows is not None:
            data = data.iloc[:rows]
        self.target = data[target_variables].values
        self.context = data[context_variables].values
        self.weight = data["weight"].values
        self.extra = data["probe_energyRaw"].values
        if device is not None:
            self.target = torch.tensor(self.target, dtype=torch.float32).to(device)
            self.context = torch.tensor(self.context, dtype=torch.float32).to(device)
            self.weight = torch.tensor(self.weight, dtype=torch.float32).to(device)
            self.extra = torch.tensor(self.extra, dtype=torch.float32).to(device)

    def __len__(self):
        assert len(self.context) == len(self.target)
        return len(self.target)

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx], self.weight[idx], self.extra[idx]


class ParquetDatasetOne(Dataset):
    def __init__(
        self,
        mc_parquet_file,
        context_variables,
        target_variables,
        device=None,
        pipelines=None,
        retrain_pipelines=False,
        rows=None,
        data_parquet_file=None,
    ):
        self.context_variables = deepcopy(context_variables)
        self.target_variables = deepcopy(target_variables)
        self.all_variables = self.context_variables + self.target_variables
        mc = pd.read_parquet(
            mc_parquet_file, columns=self.all_variables + ["weight", "probe_energyRaw"], engine="fastparquet"
        )
        mc["lab"] = 1
        if data_parquet_file is not None:
            data = pd.read_parquet(
                data_parquet_file, columns=self.all_variables + ["weight", "probe_energyRaw"], engine="fastparquet"
            )
            data["lab"] = 0
            # concatenate
            mc = pd.concat([data, mc])
        self.context_variables.append("lab")
        # shuffle
        mc = mc.sample(frac=1).reset_index(drop=True)
        self.pipelines = pipelines
        if self.pipelines is not None:
            for var, pipeline in self.pipelines.items():
                if var in self.all_variables:
                    trans = pipeline.fit_transform if retrain_pipelines else pipeline.transform
                    mc[var] = trans(
                        mc[var].values.reshape(-1, 1)
                    ).reshape(-1)
        if rows is not None:
            mc = mc.iloc[:rows]
        self.target = mc[self.target_variables].values
        self.context = mc[self.context_variables].values
        self.weight = mc["weight"].values
        self.extra = mc["probe_energyRaw"].values
        if device is not None:
            self.target = torch.tensor(self.target, dtype=torch.float32).to(device)
            self.context = torch.tensor(self.context, dtype=torch.float32).to(device)
            self.weight = torch.tensor(self.weight, dtype=torch.float32).to(device)
            self.extra = torch.tensor(self.extra, dtype=torch.float32).to(device)

    def __len__(self):
        assert len(self.context) == len(self.target)
        return len(self.target)

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx], self.weight[idx], self.extra[idx]