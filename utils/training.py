import numpy as np
import pickle as pkl
import time
import os
import logging

logger = logging.getLogger(__name__)
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import ddp_setup, ParquetDataset, ParquetDatasetOne
from utils.custom_models import (
    create_mixture_flow_model,
    save_model,
    load_model,
    load_fff_mixture_model,
)
from utils.models import get_conditional_base_flow, get_zuko_nsf
from utils.plots import sample_and_plot_base, transform_and_plot_top, plot_one
from utils.log import setup_comet_logger


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = 10e10
        self.early_stop = False

    def __call__(self, val_loss):
        relative_loss = (self.best_loss - val_loss) / self.best_loss * 100
        if relative_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif relative_loss < self.min_delta:
            self.counter += 1
            logger.info(
                f"Early stopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                logger.info("Early stopping")
                self.early_stop = True


def train_base(device, cfg, world_size=None, device_ids=None):
    # device is device when not distributed and rank when distributed
    if world_size is not None:
        ddp_setup(device, world_size)

    device_id = device_ids[device] if device_ids is not None else device

    # create (and load) the model
    input_dim = len(cfg.target_variables)
    context_dim = len(cfg.context_variables)
    if cfg.model.name == "mixture":
        flow_params_dct = {
            "input_dim": input_dim,
            "context_dim": context_dim,
            "base_kwargs": {
                "num_steps_maf": cfg.model.maf.num_steps,
                "num_steps_arqs": cfg.model.arqs.num_steps,
                "num_transform_blocks_maf": cfg.model.maf.num_transform_blocks,
                "num_transform_blocks_arqs": cfg.model.arqs.num_transform_blocks,
                "activation": cfg.model.activation,
                "dropout_probability_maf": cfg.model.maf.dropout_probability,
                "dropout_probability_arqs": cfg.model.arqs.dropout_probability,
                "use_residual_blocks_maf": cfg.model.maf.use_residual_blocks,
                "use_residual_blocks_arqs": cfg.model.arqs.use_residual_blocks,
                "batch_norm_maf": cfg.model.maf.batch_norm,
                "batch_norm_arqs": cfg.model.arqs.batch_norm,
                "num_bins_arqs": cfg.model.arqs.num_bins,
                "tail_bound_arqs": cfg.model.arqs.tail_bound,
                "hidden_dim_maf": cfg.model.maf.hidden_dim,
                "hidden_dim_arqs": cfg.model.arqs.hidden_dim,
                "init_identity": cfg.model.init_identity,
            },
            "transform_type": cfg.model.transform_type,
        }
        model = create_mixture_flow_model(**flow_params_dct)

    elif cfg.model.name == "splines":
        model = get_conditional_base_flow(
            input_dim=input_dim,
            context_dim=context_dim,
            nstack=cfg.model.nstack,
            nnodes=cfg.model.nnodes,
            nblocks=cfg.model.nblocks,
            tail_bound=cfg.model.tail_bound,
            nbins=cfg.model.nbins,
            activation=cfg.model.activation,
            dropout_probability=cfg.model.dropout_probability,
        )
    
    elif cfg.model.name == "zuko_nsf":
        model = get_zuko_nsf(
            input_dim=input_dim,
            context_dim=context_dim,
            ntransforms=cfg.model.ntransforms,
            nbins=cfg.model.nbins,
            nnodes=cfg.model.nnodes,
            nlayers=cfg.model.nlayers,
        )

    if cfg.checkpoint is not None:
        # assume that the checkpoint is path to a directory
        model, _, _, start_epoch, th, _ = load_model(
            model, model_dir=cfg.checkpoint, filename="checkpoint-latest.pt"
        )
        model = model.to(device)
        best_train_loss = np.min(th)
        logger.info("Loaded model from checkpoint: {}".format(cfg.checkpoint))
        logger.info("Resuming from epoch {}".format(start_epoch))
        logger.info("Best train loss found to be: {}".format(best_train_loss))
    else:
        start_epoch = 1
        best_train_loss = 10000000

    model = model.to(device)

    early_stopping = EarlyStopping(
        patience=cfg.stopper.patience, min_delta=cfg.stopper.min_delta
    )

    if world_size is not None:
        ddp_model = DDP(
            model,
            device_ids=[device],
            output_device=device,
            # find_unused_parameters=True,
        )
        model = ddp_model.module
    else:
        ddp_model = model
    logger.info(
        "Number of parameters: {}".format(sum(p.numel() for p in model.parameters()))
    )

    # make datasets
    sample = cfg.sample
    calo = cfg.calo

    if sample == "data":
        if calo == "eb":
            train_file = f"{script_dir}/../preprocess/data_eb_train.parquet"
            test_file = f"{script_dir}/../preprocess/data_eb_test.parquet"
        elif calo == "ee":
            train_file = f"{script_dir}/../preprocess/data_ee_train.parquet"
            test_file = f"{script_dir}/../preprocess/data_ee_test.parquet"
    elif sample == "mc":
        if calo == "eb":
            train_file = f"{script_dir}/../preprocess/mc_eb_train.parquet"
            test_file = f"{script_dir}/../preprocess/mc_eb_test.parquet"
        elif calo == "ee":
            train_file = f"{script_dir}/../preprocess/mc_ee_train.parquet"
            test_file = f"{script_dir}/../preprocess/mc_ee_test.parquet"

    with open(f"{script_dir}/../preprocess/pipelines_{calo}.pkl", "rb") as file:
        pipelines = pkl.load(file)
        pipelines = pipelines[cfg.pipelines]

    train_dataset = ParquetDataset(
        train_file,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=pipelines,
        rows=cfg.train.size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False if world_size is not None else True,
        sampler=DistributedSampler(train_dataset) if world_size is not None else None,
        # num_workers=2,
        # pin_memory=True,
    )
    test_dataset = ParquetDataset(
        test_file,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=train_dataset.pipelines,
        rows=cfg.test.size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        # num_workers=2,
        # pin_memory=True,
    )

    # train the model
    writer = SummaryWriter(log_dir="runs")
    comet_name = os.getcwd().split("/")[-1]
    comet_logger = setup_comet_logger(comet_name, cfg.model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

    train_history, test_history = [], []
    for epoch in range(start_epoch, cfg.epochs + 1):
        if world_size is not None:
            b_sz = len(next(iter(train_loader))[0])
            logger.info(
                f"[GPU{device_id}] | Rank {device} | Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_loader)}"
            )
            train_loader.sampler.set_epoch(epoch)
        logger.info(f"Epoch {epoch}/{cfg.epochs}:")

        train_losses = []
        test_losses = []
        # train
        start = time.time()
        logger.info("Training...")
        for context, target, weights, _ in train_loader:
            # context, target = context.to(device), target.to(device)
            model.train()
            optimizer.zero_grad()

            if cfg.model.name == "mixture":
                log_prog, logabsdet = ddp_model(target, context=context)
                loss = weights * (-log_prog - logabsdet)
            elif cfg.model.name == "splines":
                loss = ddp_model(target, context=context)
                loss = weights * loss
            elif "zuko" in cfg.model.name:
                loss = -ddp_model(context).log_prob(target)
                loss = weights * loss
            loss = loss.mean()
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_train_loss = np.mean(train_losses)
        train_history.append(epoch_train_loss)

        # test
        logger.info("Testing...")
        for context, target, weights, _ in test_loader:
            # context, target = context.to(device), target.to(device)
            with torch.no_grad():
                model.eval()
                if cfg.model.name == "mixture":
                    log_prog, logabsdet = ddp_model(target, context=context)
                    loss = weights * (-log_prog - logabsdet)
                elif cfg.model.name == "splines":
                    loss = ddp_model(target, context=context)
                    loss = weights * loss
                elif "zuko" in cfg.model.name:
                    loss = -ddp_model(context).log_prob(target)
                    loss = weights * loss
                loss = loss.mean()
                test_losses.append(loss.item())

        epoch_test_loss = np.mean(test_losses)
        test_history.append(epoch_test_loss)
        if device == 0 or world_size is None:
            writer.add_scalars(
                "Losses", {"train": epoch_train_loss, "val": epoch_test_loss}, epoch
            )
            comet_logger.log_metrics(
                {"train": epoch_train_loss, "val": epoch_test_loss}, step=epoch
            )

        # sample and validation
        if epoch % cfg.sample_every == 0 or epoch == 1:
            logger.info("Sampling and plotting...")
            sample_and_plot_base(
                test_loader=test_loader,
                model=model,
                model_name=cfg.model.name,
                epoch=epoch,
                writer=writer,
                comet_logger=comet_logger,
                context_variables=cfg.context_variables,
                target_variables=cfg.target_variables,
                device=device,
                pipeline=cfg.pipelines,
                calo=calo,
            )

        duration = time.time() - start
        logger.info(
            f"Epoch {epoch} | GPU{device_id} | Rank {device} - train loss: {epoch_train_loss:.4f} - val loss: {epoch_test_loss:.4f} - time: {duration:.2f}s"
        )

        if device == 0 or world_size is None:
            save_model(
                epoch,
                ddp_model,
                scheduler,
                train_history,
                test_history,
                name="checkpoint-latest.pt",
                model_dir=".",
                optimizer=optimizer,
                is_ddp=world_size is not None,
            )

        if epoch_train_loss < best_train_loss:
            logger.info("New best train loss, saving model...")
            best_train_loss = epoch_train_loss
            if device == 0 or world_size is None:
                save_model(
                    epoch,
                    ddp_model,
                    scheduler,
                    train_history,
                    test_history,
                    name="best_train_loss.pt",
                    model_dir=".",
                    optimizer=optimizer,
                    is_ddp=world_size is not None,
                )

        early_stopping(epoch_train_loss)
        if early_stopping.early_stop:
            break

    writer.close()


def train_top(device, cfg, world_size=None, device_ids=None):
    # device is device when not distributed and rank when distributed
    if world_size is not None:
        ddp_setup(device, world_size)

    device_id = device_ids[device] if device_ids is not None else device

    # models
    input_dim = len(cfg.target_variables)
    context_dim = len(cfg.context_variables)
    if cfg.model.name == "mixture":
        flow_params_dct = {
            "input_dim": input_dim,
            "context_dim": context_dim,
            "base_kwargs": {
                "num_steps_maf": cfg.model.maf.num_steps,
                "num_steps_arqs": cfg.model.arqs.num_steps,
                "num_transform_blocks_maf": cfg.model.maf.num_transform_blocks,
                "num_transform_blocks_arqs": cfg.model.arqs.num_transform_blocks,
                "activation": cfg.model.activation,
                "dropout_probability_maf": cfg.model.maf.dropout_probability,
                "dropout_probability_arqs": cfg.model.arqs.dropout_probability,
                "use_residual_blocks_maf": cfg.model.maf.use_residual_blocks,
                "use_residual_blocks_arqs": cfg.model.arqs.use_residual_blocks,
                "batch_norm_maf": cfg.model.maf.batch_norm,
                "batch_norm_arqs": cfg.model.arqs.batch_norm,
                "num_bins_arqs": cfg.model.arqs.num_bins,
                "tail_bound_arqs": cfg.model.arqs.tail_bound,
                "hidden_dim_maf": cfg.model.maf.hidden_dim,
                "hidden_dim_arqs": cfg.model.arqs.hidden_dim,
                "init_identity": cfg.model.init_identity,
            },
            "transform_type": cfg.model.transform_type,
        }
        create_function = create_mixture_flow_model
        which = "mixture"
    elif cfg.model.name == "splines":
        pass
    elif cfg.model.name == "zuko_nsf":
        flow_params_dct = {
            "input_dim": input_dim,
            "context_dim": context_dim,
            "ntransforms": cfg.model.ntransforms,
            "nbins": cfg.model.nbins,
            "nnodes": cfg.model.nnodes,
            "nlayers": cfg.model.nlayers,
        }
        create_function = get_zuko_nsf
        which = "zuko_nsf"

    model_data = create_function(**flow_params_dct)
    model_data, _, _, _, _, _ = load_model(
        model_data, model_dir=cfg.data.checkpoint, filename="best_train_loss.pt", which=which
    )
    model_data = model_data.to(device)
    model_mc = create_function(**flow_params_dct).to(device)
    model_mc, _, _, _, _, _ = load_model(
        model_mc, model_dir=cfg.mc.checkpoint, filename="best_train_loss.pt", which=which
    )
    model_mc = model_mc.to(device)

    flow_params_dct["mc_flow"] = model_mc
    flow_params_dct["data_flow"] = model_data
    penalty = {
        "penalty_type": cfg.model.penalty,
        "penalty_weight": cfg.model.penalty_weight,
        "anneal": cfg.model.anneal,
    }
    flow_params_dct["penalty"] = penalty
    model = create_function(**flow_params_dct)
    start_epoch = 1
    best_train_loss = np.inf
    if cfg.checkpoint is not None:
        if cfg.model.name == "mixture":
            model, _, _, start_epoch, th, _ = load_fff_mixture_model(
                top_file=f"{cfg.checkpoint}/checkpoint-latest.pt",
                mc_file=f"{cfg.mc.checkpoint}/best_train_loss.pt",
                data_file=f"{cfg.data.checkpoint}/best_train_loss.pt",
                top_penalty=penalty,
            )
            model_data = model.flow_data
            model_mc = model.flow_mc
            best_train_loss = np.min(th)

    model = model.to(device)

    early_stopping = EarlyStopping(
        patience=cfg.stopper.patience, min_delta=cfg.stopper.min_delta
    )

    if world_size is not None:
        ddp_model = DDP(
            model,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=True,
        )
        model = ddp_model.module
    else:
        ddp_model = model
    logger.info("Number of parameters: {}".format(sum(p.numel() for p in model.parameters())))

    # make datasets
    calo = cfg.calo
    train_file_data = f"{script_dir}/../preprocess/data_{calo}_train.parquet"
    train_file_mc = f"{script_dir}/../preprocess/mc_{calo}_train.parquet"
    test_file_data = f"{script_dir}/../preprocess/data_{calo}_test.parquet"
    test_file_mc = f"{script_dir}/../preprocess/mc_{calo}_test.parquet"

    with open(f"{script_dir}/../preprocess/pipelines_{calo}.pkl", "rb") as file:
        pipelines_data = pkl.load(file)
        pipelines_data = pipelines_data[cfg.pipelines]

    train_dataset_data = ParquetDataset(
        train_file_data,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=pipelines_data,
        rows=cfg.train.size,
    )
    train_loader_data = DataLoader(
        train_dataset_data,
        batch_size=cfg.train.batch_size,
        shuffle=False if world_size is not None else True,
        sampler=DistributedSampler(train_dataset_data)
        if world_size is not None
        else None,
    )
    test_dataset_data = ParquetDataset(
        test_file_data,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=pipelines_data,
        rows=cfg.test.size,
    )
    test_loader_data = DataLoader(
        test_dataset_data,
        batch_size=cfg.test.batch_size,
        shuffle=False if world_size is not None else True,
        sampler=DistributedSampler(test_dataset_data)
        if world_size is not None
        else None,
    )
    train_dataset_mc = ParquetDataset(
        train_file_mc,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=pipelines_data,
        rows=cfg.train.size,
    )
    train_loader_mc = DataLoader(
        train_dataset_mc,
        batch_size=cfg.train.batch_size,
        shuffle=False if world_size is not None else True,
        sampler=DistributedSampler(train_dataset_mc)
        if world_size is not None
        else None,
    )
    test_dataset_mc = ParquetDataset(
        test_file_mc,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=pipelines_data,
        rows=cfg.test.size,
    )
    test_loader_mc = DataLoader(
        test_dataset_mc,
        batch_size=cfg.test.batch_size,
        shuffle=False if world_size is not None else True,
        sampler=DistributedSampler(test_dataset_mc) if world_size is not None else None,
    )
    test_dataset_mc_full = ParquetDataset(
        test_file_mc,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=pipelines_data,
        rows=None,
    )
    test_loader_mc_full = DataLoader(
        test_dataset_mc_full,
        batch_size=cfg.test.batch_size,
        shuffle=False if world_size is not None else True,
        sampler=DistributedSampler(test_dataset_mc_full)
        if world_size is not None
        else None,
    )
    test_dataset_data_full = ParquetDataset(
        test_file_data,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=pipelines_data,
        rows=None,
    )
    test_loader_data_full = DataLoader(
        test_dataset_data_full,
        batch_size=cfg.test.batch_size,
        shuffle=False if world_size is not None else True,
        sampler=DistributedSampler(test_dataset_data_full)
        if world_size is not None
        else None,
    )

    # freeze base flows
    for param in model_data.parameters():
        param.requires_grad = False
    for param in model_mc.parameters():
        param.requires_grad = False
    # check that are freezed also in the model
    for param in model.flow_mc.parameters():
        assert param.requires_grad == False
    for param in model.flow_data.parameters():
        assert param.requires_grad == False

    # train the model
    writer = SummaryWriter(log_dir="runs")
    comet_name = os.getcwd().split("/")[-1]
    comet_logger = setup_comet_logger(comet_name, cfg.model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

    train_history, test_history = [], []

    # 1 - log prob base flow
    # 2 - logabsdet top transform
    # 3 - distance
    for epoch in range(start_epoch, cfg.epochs + 1):
        if world_size is not None:
            b_sz = len(next(iter(train_loader_mc))[0])
            logger.info(
                f"[GPU{device_id}] | Rank {device} | Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_loader_mc)}"
            )
            train_loader_mc.sampler.set_epoch(epoch)
            train_loader_data.sampler.set_epoch(epoch)

        logger.info(f"Epoch {epoch}/{cfg.epochs}:")
        epoch_is_even = epoch % 2 == 0
        start = time.time()
        train_losses, test_losses = [], []
        train_losses_1, test_losses_1 = [], []
        train_losses_2, test_losses_2 = [], []
        train_losses_3, test_losses_3 = [], []
        # train
        logger.info("Training...")
        for i, (data, mc) in enumerate(zip(train_loader_data, train_loader_mc)):
            if i % 2 == 0 + int(epoch_is_even):
                # print(f"Epoch {epoch} - Batch {i} - inverse = False")
                context, target, weights, _ = mc
                inverse = False
            else:
                context, target, weights, _ = data
                inverse = True

            optimizer.zero_grad()

            if "zuko" in cfg.model.name:
                loss1, loss2, loss3 = ddp_model.log_prob(target, context, inverse=inverse)
            else:
                loss1, loss2, loss3 = ddp_model(target, context, inverse=inverse)
            loss1 = loss1 * weights
            loss2 = loss2 * weights
            loss3 = loss3 * weights
            loss = - loss1 - loss2 - loss3
            loss = loss.mean()
            loss1 = loss1.mean()
            loss2 = loss2.mean()
            loss3 = loss3.mean()
            train_losses.append(loss.item())
            train_losses_1.append(loss1.item())
            train_losses_2.append(loss2.item())
            train_losses_3.append(loss3.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_train_loss = np.mean(train_losses)
        train_history.append(epoch_train_loss)
        epoch_train_loss_1 = np.mean(train_losses_1)
        epoch_train_loss_2 = np.mean(train_losses_2)
        epoch_train_loss_3 = np.mean(train_losses_3)

        # test
        print("Testing...")
        for i, (data, mc) in enumerate(zip(test_loader_data, test_loader_mc)):
            if i % 2 == 0 + int(epoch_is_even):
                context, target, weights, _ = mc
                inverse = False
            else:
                context, target, weights, _ = data
                inverse = True
            with torch.no_grad():
                if "zuko" in cfg.model.name:
                    loss1, loss2, loss3 = ddp_model.log_prob(target, context, inverse=inverse)
                else:
                    loss1, loss2, loss3 = ddp_model(target, context, inverse=inverse)
                loss1 = loss1 * weights
                loss2 = loss2 * weights
                loss3 = loss3 * weights
                loss = - loss1 - loss2 - loss3
                loss = loss.mean()
                loss1 = loss1.mean()
                loss2 = loss2.mean()
                loss3 = loss3.mean()
                test_losses.append(loss.item())
                test_losses_1.append(loss1.item())
                test_losses_2.append(loss2.item())
                test_losses_3.append(loss3.item())

        epoch_test_loss = np.mean(test_losses)
        test_history.append(epoch_test_loss)
        epoch_test_loss_1 = np.mean(test_losses_1)
        epoch_test_loss_2 = np.mean(test_losses_2)
        epoch_test_loss_3 = np.mean(test_losses_3)

        if device == 0 or world_size is None:
            writer.add_scalars(
                "Total Losses",
                {"train": epoch_train_loss, "val": epoch_test_loss},
                epoch,
            )
            writer.add_scalars(
                "Log prob base flow",
                {"train": epoch_train_loss_1, "val": epoch_test_loss_1},
                epoch,
            )
            writer.add_scalars(
                "Logabsdet top transform",
                {"train": epoch_train_loss_2, "val": epoch_test_loss_2},
                epoch,
            )
            writer.add_scalars(
                "Distance",
                {"train": epoch_train_loss_3, "val": epoch_test_loss_3},
                epoch,
            )
            comet_logger.log_metrics(
                {
                    "train_total": epoch_train_loss,
                    "val_total": epoch_test_loss,
                    "train_log_prob": epoch_train_loss_1,
                    "val_log_prob": epoch_test_loss_1,
                    "train_logabsdet": epoch_train_loss_2,
                    "val_logabsdet": epoch_test_loss_2,
                    "train_distance": epoch_train_loss_3,
                    "val_distance": epoch_test_loss_3,
                },
                step=epoch,
            )

        # sample and validation
        if epoch % cfg.sample_every == 0 or epoch == 1:
            print("Sampling and plotting...")
            transform_and_plot_top(
                mc_loader=test_loader_mc_full,
                data_loader=test_loader_data_full,
                model=model,
                epoch=epoch,
                writer=writer,
                comet_logger=comet_logger,
                context_variables=cfg.context_variables,
                target_variables=cfg.target_variables,
                device=device,
                pipeline=cfg.pipelines,
                calo=cfg.calo,
            )

        duration = time.time() - start
        logger.info(
            f"Epoch {epoch} | GPU{device_id} | Rank {device} - train loss: {epoch_train_loss:.4f} - val loss: {epoch_test_loss:.4f} - time: {duration:.2f}s"
        )

        # print([k for k in model.state_dict().keys() if "distance_object" in k])
        # print(model.state_dict()["flow_data._transform._transforms.22.autoregressive_net.blocks.2.linear_layers.0.weight"])
        # print(ddp_model.module.state_dict()["flow_data._transform._transforms.22.autoregressive_net.blocks.2.linear_layers.0.weight"])

        if device == 0 or world_size is None:
            save_model(
                epoch,
                ddp_model,
                scheduler,
                train_history,
                test_history,
                name="checkpoint-latest.pt",
                model_dir=".",
                optimizer=optimizer,
                is_ddp=world_size is not None,
                # save_both=epoch % cfg.sample_every == 0,
            )

        if epoch_train_loss < best_train_loss:
            logger.info("New best train loss!")
            best_train_loss = epoch_train_loss
            if device == 0 or world_size is None:
                save_model(
                    epoch,
                    ddp_model,
                    scheduler,
                    train_history,
                    test_history,
                    name="best_train_loss.pt",
                    model_dir=".",
                    optimizer=optimizer,
                    is_ddp=world_size is not None,
                    # save_both=epoch % cfg.sample_every == 0,
                )

        early_stopping(epoch_train_loss)
        if early_stopping.early_stop:
            break

    writer.close()


def train_one(device, cfg, world_size=None, device_ids=None):
    # device is device when not distributed and rank when distributed
    if world_size is not None:
        ddp_setup(device, world_size)

    device_id = device_ids[device] if device_ids is not None else device

    # create (and load) the model
    input_dim = len(cfg.target_variables)
    context_dim = len(cfg.context_variables)
    if cfg.model.name == "mixture":
        flow_params_dct = {
            "input_dim": input_dim,
            "context_dim": context_dim + 1,
            "base_kwargs": {
                "num_steps_maf": cfg.model.maf.num_steps,
                "num_steps_arqs": cfg.model.arqs.num_steps,
                "num_transform_blocks_maf": cfg.model.maf.num_transform_blocks,
                "num_transform_blocks_arqs": cfg.model.arqs.num_transform_blocks,
                "activation": cfg.model.activation,
                "dropout_probability_maf": cfg.model.maf.dropout_probability,
                "dropout_probability_arqs": cfg.model.arqs.dropout_probability,
                "use_residual_blocks_maf": cfg.model.maf.use_residual_blocks,
                "use_residual_blocks_arqs": cfg.model.arqs.use_residual_blocks,
                "batch_norm_maf": cfg.model.maf.batch_norm,
                "batch_norm_arqs": cfg.model.arqs.batch_norm,
                "num_bins_arqs": cfg.model.arqs.num_bins,
                "tail_bound_arqs": cfg.model.arqs.tail_bound,
                "hidden_dim_maf": cfg.model.maf.hidden_dim,
                "hidden_dim_arqs": cfg.model.arqs.hidden_dim,
                "init_identity": cfg.model.init_identity,
            },
            "transform_type": cfg.model.transform_type,
        }
        model = create_mixture_flow_model(**flow_params_dct)

    elif cfg.model.name == "splines":
        model = get_conditional_base_flow(
            input_dim=input_dim,
            context_dim=context_dim + 1,
            nstack=cfg.model.nstack,
            nnodes=cfg.model.nnodes,
            nblocks=cfg.model.nblocks,
            tail_bound=cfg.model.tail_bound,
            nbins=cfg.model.nbins,
            activation=cfg.model.activation,
            dropout_probability=cfg.model.dropout_probability,
        )
    
    elif cfg.model.name == "zuko_nsf":
        model = get_zuko_nsf(
            input_dim=input_dim,
            context_dim=context_dim + 1,
            ntransforms=cfg.model.ntransforms,
            nbins=cfg.model.nbins,
            nnodes=cfg.model.nnodes,
            nlayers=cfg.model.nlayers,
        )

    if cfg.checkpoint is not None:
        if cfg.model.name == "mixture":
            # assume that the checkpoint is path to a directory
            model, _, _, start_epoch, th, _ = load_model(
                model, model_dir=cfg.checkpoint, filename="checkpoint-latest.pt"
            )
            best_train_loss = np.min(th)
            logger.info("Loaded model from checkpoint: {}".format(cfg.checkpoint))
            logger.info("Resuming from epoch {}".format(start_epoch))
            logger.info("Best train loss found to be: {}".format(best_train_loss))
    else:
        start_epoch = 1
        best_train_loss = 10000000
    
    model = model.to(device)

    early_stopping = EarlyStopping(
        patience=cfg.stopper.patience, min_delta=cfg.stopper.min_delta
    )

    if world_size is not None:
        ddp_model = DDP(
            model,
            device_ids=[device],
            output_device=device,
            #find_unused_parameters=True,
        )
        model = ddp_model.module
    else:
        ddp_model = model
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    # make datasets
    calo = cfg.calo

    data_train_file = f"{script_dir}/../preprocess/data_{calo}_train.parquet"
    data_test_file = f"{script_dir}/../preprocess/data_{calo}_test.parquet"
    mc_train_file = f"{script_dir}/../preprocess/mc_{calo}_train.parquet"
    mc_test_file = f"{script_dir}/../preprocess/mc_{calo}_test.parquet"
    
    with open(f"{script_dir}/../preprocess/pipelines_{calo}.pkl", "rb") as file:
        pipelines = pkl.load(file)
        pipelines = pipelines[cfg.pipelines]

    train_dataset = ParquetDatasetOne(
        mc_train_file,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=pipelines,
        rows=cfg.train.size,
        data_parquet_file=data_train_file,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False if world_size is not None else True,
        sampler=DistributedSampler(train_dataset) if world_size is not None else None,
    )
    test_dataset = ParquetDatasetOne(
        mc_test_file,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=train_dataset.pipelines,
        rows=cfg.test.size,
        data_parquet_file=data_train_file,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
    )
    data_test_dataset = ParquetDataset(
        data_test_file,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=train_dataset.pipelines,
        #rows=cfg.test.size,
    )
    data_test_loader = DataLoader(
        data_test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
    )
    mc_test_dataset = ParquetDatasetOne(
        mc_test_file,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=train_dataset.pipelines,
        rows=len(data_test_dataset)
    )
    mc_test_loader = DataLoader(
        mc_test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
    )

    # train the model
    writer = SummaryWriter(log_dir="runs")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

    train_history, test_history = [], []
    for epoch in range(start_epoch, cfg.epochs + 1):
        if world_size is not None:
            b_sz = len(next(iter(train_loader))[0])
            logger.info(
                f"[GPU{device_id}] | Rank {device} | Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_loader)}"
            )
            train_loader.sampler.set_epoch(epoch)
        logger.info(f"Epoch {epoch}/{cfg.epochs}:")
        
        train_losses = []
        test_losses = []
        # train
        start = time.time()
        logger.info("Training...")
        for i, (context, target, weights, _) in enumerate(train_loader):
            #context, target = context.to(device), target.to(device)
            model.train()
            optimizer.zero_grad()
            
            if cfg.model.name == "mixture":
                log_prog, logabsdet = ddp_model(target, context=context)
                loss = - log_prog * weights - logabsdet * weights
            elif cfg.model.name in ["splines"]:
                loss = ddp_model.log_prob(target, context=context)
                loss = loss * weights
            elif "zuko" in cfg.model.name:
                loss = -ddp_model(context).log_prob(target)
            
            loss = loss.mean()
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_train_loss = np.mean(train_losses)
        train_history.append(epoch_train_loss)

        # test
        logger.info("Testing...")
        for i, (context, target, weights, _) in enumerate(test_loader):
            #context, target = context.to(device), target.to(device)
            with torch.no_grad():
                model.eval()
                if cfg.model.name == "mixture":
                    log_prog, logabsdet = ddp_model(target, context=context)
                    loss = - log_prog * weights - logabsdet * weights
                elif cfg.model.name in ["splines"]:
                    loss = ddp_model.log_prob(target, context=context)
                    loss = loss * weights
                elif "zuko" in cfg.model.name:
                    loss = -ddp_model(context).log_prob(target)
                loss = loss.mean()
                test_losses.append(loss.item())

        epoch_test_loss = np.mean(test_losses)
        test_history.append(epoch_test_loss)
        if device == 0 or world_size is None:
            writer.add_scalars(
                "Losses", {"train": epoch_train_loss, "val": epoch_test_loss}, epoch
            )
        
        # sample and validation
        if epoch % cfg.sample_every == 0 or epoch == 1:
            print("Sampling and plotting...")
            plot_one(
                mc_test_loader=mc_test_loader,
                data_test_loader=data_test_loader,
                model=model,
                model_name=cfg.model.name,
                epoch=epoch,
                writer=writer,
                context_variables=cfg.context_variables,
                target_variables=cfg.target_variables,
                device=device,
                pipeline=cfg.pipelines,
                calo=calo,
            )

        duration = time.time() - start
        print(
            f"Epoch {epoch} | GPU{device_id} | Rank {device} - train loss: {epoch_train_loss:.4f} - val loss: {epoch_test_loss:.4f} - time: {duration:.2f}s"
        )

        if device == 0 or world_size is None:
            save_model(
                epoch,
                ddp_model,
                scheduler,
                train_history,
                test_history,
                name="checkpoint-latest.pt",
                model_dir=".",
                optimizer=optimizer,
                is_ddp=world_size is not None,
            )
        
        if epoch_train_loss < best_train_loss:
            print("New best train loss, saving model...")
            best_train_loss = epoch_train_loss
            if device == 0 or world_size is None:
                save_model(
                    epoch,
                    ddp_model,
                    scheduler,
                    train_history,
                    test_history,
                    name="best_train_loss.pt",
                    model_dir=".",
                    optimizer=optimizer,
                    is_ddp=world_size is not None,
                )

        early_stopping(epoch_train_loss)
        if early_stopping.early_stop:
            break

    writer.close()