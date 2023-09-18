import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import dask.dataframe as dd
from dask.distributed import LocalCluster, Client
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
import cloudpickle
from copy import deepcopy
import os
from hep_ml import reweight

from utils.transforms import CustomLog, IsoTransformer, remove_outliers
from utils.plots import dump_main_plot, transformed_ranges
from utils.log import setup_logger
from utils.phoid import calculate_photonid_mva


# More than one pipeline defined since we don't know yet if the first one will be the best
pipelines = {
    "pipe0": {
        # Context
        "probe_pt": Pipeline(
            [
                ("box_cox", PowerTransformer(method="box-cox")),
            ]
        ),
        "probe_eta": Pipeline(
            [
                ("standard", StandardScaler()),
            ]
        ),
        "probe_phi": Pipeline(
            [
                ("standard", StandardScaler()),
            ]
        ),
        "probe_fixedGridRhoAll": Pipeline(
            [
                ("standard", StandardScaler()),
            ]
        ),
        # Shower shapes
        "probe_r9": Pipeline(
            [
                ("johnson", PowerTransformer(method="yeo-johnson")),
                ("standard", StandardScaler()),
            ]
        ),
        "probe_s4": Pipeline(
            [
                ("johnson", PowerTransformer(method="yeo-johnson")),
                ("standard", StandardScaler()),
            ]
        ),
        "probe_sieie": Pipeline(
            [
                ("box_cox", PowerTransformer(method="box-cox")),
                ("standard", StandardScaler()),
            ]
        ),
        "probe_sieip": Pipeline(
            [
                ("log", FunctionTransformer(np.log1p, np.expm1)),
                ("standard", StandardScaler()),
            ]
        ),
        "probe_etaWidth": Pipeline(
            [
                (
                    "arctan_trans",
                    FunctionTransformer(
                        lambda x: np.arctan(x * 100 - 0.15),
                        inverse_func=lambda x: (np.tan(x) + 0.15) / 100,
                    ),
                ),
                ("standard", StandardScaler()),
            ]
        ),
        "probe_phiWidth": Pipeline(
            [
                ("log", FunctionTransformer(np.log, np.exp)),
                ("standard", StandardScaler()),
            ]
        ),
        "probe_pfPhoIso03": Pipeline(
            [
                ("sampler", IsoTransformer(0.5)),
                ("custom_log", CustomLog()),
                ("standard", StandardScaler()),
            ]
        ),
        "probe_pfChargedIsoPFPV": Pipeline(
            [
                ("sampler", IsoTransformer(1.5)),
                ("custom_log", CustomLog()),
                ("standard", StandardScaler()),
            ]
        ),
        "probe_pfChargedIsoWorstVtx": Pipeline(
            [
                ("sampler", IsoTransformer(0.1)),
                ("custom_log", CustomLog()),
                ("standard", StandardScaler()),
            ]
        ),
        # Others
        "probe_energyRaw": Pipeline(
            [
                ("none", None)
            ]
        ),
    },
}


def main():
    logger = setup_logger(level="INFO")

    output_dir = "./preprocess"
    fig_output_dir = "./preprocess/figures"
    # create directory if it doesn't exist
    if not os.path.exists(fig_output_dir):
        os.makedirs(fig_output_dir)

    with open("./preprocess/var_specs.json", "r") as f:
        vars_config = json.load(f)
        # turn into a dict with name as key
        vars_config = {d["name"]: d for d in vars_config}

    context = ["probe_pt", "probe_eta", "probe_phi", "probe_fixedGridRhoAll"]
    shower_shapes = [
        "probe_r9",
        "probe_s4",
        "probe_sieie",
        "probe_sieip",
        "probe_etaWidth",
        "probe_phiWidth",
    ]
    isolation = [
        "probe_pfPhoIso03",
        "probe_pfChargedIsoPFPV",
        "probe_pfChargedIsoWorstVtx",
    ]
    others = ["probe_energyRaw"]
    columns = context + shower_shapes + isolation + others

    # start a local cluster for parallel processing
    cluster = LocalCluster()
    client = Client(cluster)

    data_file_pattern = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/samples/data/DoubleEG/nominal/*.parquet"
    mc_uncorr_file_pattern = "/work/gallim/devel/CQRRelatedStudies/NormalizingFlow/samples/mc_uncorr/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/nominal/*.parquet"

    data_df = dd.read_parquet(data_file_pattern, columns=columns, engine="fastparquet")
    mc_uncorr_df = dd.read_parquet(
        mc_uncorr_file_pattern, columns=columns, engine="fastparquet"
    )

    logger.info("Reading data...")
    data_df = data_df.compute()
    logger.info("Reading MC...")
    mc_uncorr_df = mc_uncorr_df.compute()

    data_df_eb = data_df[np.abs(data_df.probe_eta) < 1.4442]
    data_df_ee = data_df[np.abs(data_df.probe_eta) > 1.56]
    mc_uncorr_df_eb = mc_uncorr_df[np.abs(mc_uncorr_df.probe_eta) < 1.4442]
    mc_uncorr_df_ee = mc_uncorr_df[np.abs(mc_uncorr_df.probe_eta) > 1.56]

    dataframes = {
        "eb": [data_df_eb, mc_uncorr_df_eb],
        "ee": [data_df_ee, mc_uncorr_df_ee],
    }
    #for calo in ["eb", "ee"]:
    for calo in ["eb"]:
        data_df, mc_df = dataframes[calo]

        # common cuts
        # data_df = data_df[data_df.probe_r9 < 1.2]
        # mc_df = mc_df[mc_df.probe_r9 < 1.2]

        data_df_train, data_df_test = train_test_split(
            data_df, test_size=0.2, random_state=42
        )
        mc_df_train, mc_df_test = train_test_split(
            mc_df, test_size=0.2, random_state=42
        )

        for col in columns:
            dump_main_plot(
                data_df_test[col].values,
                mc_df_test[col].values,
                vars_config[col],
                fig_output_dir,
                calo,
                mc_corr=None,
                weights=None,
                extra_name="_test_pre",
            )

        # train 4D reweighter
        reweighter_name = "{}/reweighter_{}.pkl".format(output_dir, calo)
        if os.path.exists(reweighter_name):
            logger.info("Load reweighter from {}".format(reweighter_name))
            reweighter = pickle.load(open(reweighter_name, "rb"))
        else:
            logger.warning("Train reweighter as {} does not exist".format(reweighter_name))
            reweighter = reweight.GBReweighter(
                n_estimators=250,
                learning_rate=0.1,
                max_depth=3,
                min_samples_leaf=1000,
                gb_args={"subsample": 0.4},
            )
            reweighter.fit(
                mc_df_train[context].values[:100000],
                data_df_train[context].values[:100000],
            )
            pickle.dump(reweighter, open(reweighter_name, "wb"))

        train_weights = reweighter.predict_weights(
            mc_df_train[context].values,
        )
        
        test_weights = reweighter.predict_weights(
            mc_df_test[context].values,
        )

        # add weights to mc dataframe and ones to data dataframe
        mc_df_train["weight"] = train_weights
        data_df_train["weight"] = np.ones(len(data_df_train))
        mc_df_test["weight"] = test_weights
        data_df_test["weight"] = np.ones(len(data_df_test))

        logger.info("Plot reweighted context variables")
        for col in context:
            dump_main_plot(
                data_df_test[col].values,
                mc_df_test[col].values,
                vars_config[col],
                fig_output_dir,
                calo,
                mc_corr=None,
                weights=test_weights,
                extra_name="_test_reweighted",
            )

        # remove outliers
        logger.info("Remove outliers")
        if calo == "eb":
            data_df_train = remove_outliers(data_df_train, calo, "data")
            mc_df_train = remove_outliers(mc_df_train, calo, "mc")

        # save
        logger.info("Save dataframes")
        for ext, df_ in zip(["train", "test"], [data_df_train, data_df_test]):
            df_.to_parquet(f"{output_dir}/data_{calo}_{ext}.parquet", engine="fastparquet")
        for ext, df_ in zip(["train", "test"], [mc_df_train, mc_df_test]):
            df_.to_parquet(f"{output_dir}/mc_{calo}_{ext}.parquet", engine="fastparquet")
   
        for version in pipelines.keys(): 
            logger.info(f"Transform data with pipeline {version}")
            dct = pipelines[version]

            # train transforms with full data
            for var, pipe in dct.items():
                data_arr = data_df[var].values
                mc_arr = mc_df[var].values
                transformed_data_arr = pipe.fit_transform(data_arr.reshape(-1, 1))
                transformed_mc_arr = pipe.transform(mc_arr.reshape(-1, 1))

                # now plot for test dataframes
                data_test_arr = data_df_test[var].values
                mc_test_arr = mc_df_test[var].values
                transformed_data_test_arr = pipe.transform(data_test_arr.reshape(-1, 1))
                transformed_mc_test_arr = pipe.transform(mc_test_arr.reshape(-1, 1))
                local_var_config = deepcopy(vars_config[var])
                local_var_config["range"] = transformed_ranges[version][var]
                dump_main_plot(
                    transformed_data_test_arr,
                    transformed_mc_test_arr,
                    local_var_config,
                    fig_output_dir,
                    calo,
                    mc_corr=None,
                    weights=test_weights,
                    extra_name=f"_{version}_test_transformed",
                )

            # save pipelines
            # we save one for each combination sample/calo, containing all the versions
            # this way when we load them durtig the training the transformations are already fitted
            with open(os.path.join(output_dir, f"pipelines_{calo}.pkl"), "wb") as f:
                cloudpickle.dump(pipelines, f)

        # photon ID MVA
        logger.info("Calculate photon ID MVA")
        pho_id_data_test = calculate_photonid_mva(data_df_test, calo)
        pho_id_mc_test = calculate_photonid_mva(mc_df_test, calo)
        
        dump_main_plot(
            pho_id_data_test,
            pho_id_mc_test,
            vars_config["probe_mvaID"],
            fig_output_dir,
            calo,
            mc_corr=None,
            weights=test_weights,
            extra_name="_test_reweighted",
        )


if __name__ == "__main__":
    main()
