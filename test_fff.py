import argparse
from omegaconf import OmegaConf
from hydra import compose, initialize
import torch
from torch.utils.data import DataLoader
import pickle as pkl
import os
import logging

from utils.log import setup_logger
from utils.models import get_zuko_nsf, load_model, load_fff_model
from utils.datasets import ParquetDataset
from utils.plots import transform_and_plot_top

def parse_arguments():
    parser = argparse.ArgumentParser(description="Test a fff model")
    
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Model to test",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory",
    )

    return parser.parse_args()


def main(args):
    # setup logger
    logger = setup_logger(level="INFO")
    logger.info("diomerda")
    # read the config called train_top.yaml inside the model directory
    with initialize(config_path=args.model_dir):
        cfg = compose(config_name="train_top")
    #print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the model
    logger.info("Creating model")
    input_dim = len(cfg.target_variables)
    context_dim = len(cfg.context_variables)
    if cfg.model.name == "zuko_nsf":
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

    # properly load the model
    model, _, _, epoch, th, _ = load_fff_model(
        top_file=f"{args.model_dir}/best_train_loss.pt",
        mc_file=f"{cfg.mc.checkpoint}/best_train_loss.pt",
        data_file=f"{cfg.data.checkpoint}/best_train_loss.pt",
        top_penalty=penalty,
        which=which,
    )
    model = model.to(device)

    # make datasets
    logger.info("Creating datasets")
    calo = cfg.calo
    test_file_data = f"preprocess/data_{calo}_test.parquet"
    test_file_mc = f"preprocess/mc_{calo}_test.parquet"

    with open(f"preprocess/pipelines_{calo}.pkl", "rb") as file:
        pipelines_data = pkl.load(file)
        pipelines_data = pipelines_data[cfg.pipelines]
   
    test_dataset_data = ParquetDataset(
        test_file_data,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=pipelines_data,
        #rows=10000
    )
    test_loader_data = DataLoader(
        test_dataset_data,
        batch_size=cfg.test.batch_size,
        shuffle=False,
    )
    test_dataset_mc = ParquetDataset(
        test_file_mc,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        pipelines=pipelines_data,
        #rows=10000
    )
    test_loader_mc = DataLoader(
        test_dataset_mc,
        batch_size=cfg.test.batch_size,
        shuffle=False,
    )

    # make plots
    logger.info("Making plots")
    # make output dir with name of the directory of the model
    new_output_dir = f"{args.output_dir}/{args.model_dir.split('/')[-1]}"
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)
    transform_and_plot_top(
        mc_loader=test_loader_mc,
        data_loader=test_loader_data,
        model=model,
        epoch=epoch,
        context_variables=cfg.context_variables,
        target_variables=cfg.target_variables,
        device=device,
        pipeline=cfg.pipelines,
        calo=calo,
        output_dir=new_output_dir,
    )
    new_output_dir = f"{args.output_dir}/{args.model_dir.split('/')[-1]}_twoflows"
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)
    transform_and_plot_top(
        mc_loader=test_loader_mc,
        data_loader=test_loader_data,
        model=(model_mc, model_data),
        epoch=epoch,
        context_variables=cfg.context_variables,
        target_variables=cfg.target_variables,
        device=device,
        pipeline=cfg.pipelines,
        calo=calo,
        output_dir=new_output_dir,
    )
       

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
