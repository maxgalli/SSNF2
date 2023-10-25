import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
import torch
import pandas as pd
import json

hep.style.use("CMS")
import logging

logger = logging.getLogger(__name__)
from pathlib import Path

from utils.phoid import calculate_photonid_mva

script_dir = Path(__file__).parent.absolute()

transformed_ranges = {
    "pipe0": {
        "probe_pt": [-4, 4],
        "probe_eta": [-2, 2],
        "probe_phi": [-2, 2],
        "probe_fixedGridRhoAll": [-3, 5],
        "probe_r9": [-2, 2],
        "probe_s4": [-2, 3],
        "probe_sieie": [-6, 6],
        "probe_sieip": [-6, 6],
        "probe_etaWidth": [-3, 5],
        "probe_phiWidth": [-3, 3],
        "probe_pfPhoIso03": [-4, 3],
        "probe_pfChargedIsoPFPV": [-4, 3],
        "probe_pfChargedIsoWorstVtx": [-3, 6],
        "probe_energyRaw": [0, 300],
    },
    "pipe1": {
        "probe_pt": [-4, 4],
        "probe_eta": [-2, 2],
        "probe_phi": [-2, 2],
        "probe_fixedGridRhoAll": [-3, 5],
        "probe_r9": [-2, 2],
        "probe_s4": [-2, 3],
        "probe_sieie": [-6, 6],
        "probe_sieip": [-6, 6],
        "probe_etaWidth": [-3, 5],
        "probe_phiWidth": [-3, 3],
        "probe_pfPhoIso03": [-3, 3],
        "probe_pfChargedIsoPFPV": [-2, 3.5],
        "probe_pfChargedIsoWorstVtx": [-5, 6],
        "probe_energyRaw": [0, 300],
    }
}


def divide_dist(distribution, bins):
    sorted_dist = np.sort(distribution)
    subgroup_size = len(distribution) // bins
    edges = [sorted_dist[0]]
    for i in range(subgroup_size, len(sorted_dist), subgroup_size):
        edges.append(sorted_dist[i])
    edges[-1] = sorted_dist[-1]
    return edges


def dump_profile_plot(
    ax, ss_name, cond_name, sample_name, ss_arr, cond_arr, color, cond_edges
):
    df = pd.DataFrame({ss_name: ss_arr, cond_name: cond_arr})
    quantiles = [0.25, 0.5, 0.75]
    qlists = [[], [], []]
    centers = []
    for left_edge, right_edge in zip(cond_edges[:-1], cond_edges[1:]):
        dff = df[(df[cond_name] > left_edge) & (df[cond_name] < right_edge)]
        qlist = np.quantile(dff[ss_name], quantiles)
        for i, q in enumerate(qlist):
            qlists[i].append(q)
        centers.append((left_edge + right_edge) / 2)
    mid_index = len(quantiles) // 2
    for qlist in qlists[:mid_index]:
        ax.plot(centers, qlist, color=color, linestyle="dashed")
    for qlist in qlists[mid_index:]:
        ax.plot(centers, qlist, color=color, linestyle="dashed")
    ax.plot(centers, qlists[mid_index], color=color, label=sample_name)

    return ax


def dump_main_plot(
    data,
    mc_uncorr,
    variable_conf,
    output_dir,
    subdetector,
    mc_corr=None,
    weights=None,
    extra_name="",
    labels=None,
    writer_epoch=None,
    cometlogger_epoch=None,
):
    name = variable_conf["name"]
    title = variable_conf["title"] + "_" + subdetector
    x_label = variable_conf["x_label"]
    bins = variable_conf["bins"]
    range = variable_conf["range"]

    if type(output_dir) == str:
        output_dir = [output_dir]

    # specific ranges for EB and EE
    if name == "probe_sieie" and subdetector == "EE":
        range = [0.005, 0.04]

    if labels is None:
        labels = [
            f"Data - {subdetector}",
            f"MC - {subdetector} (uncorr.)",
            f"MC - {subdetector}",
        ]

    logger.info("Plotting variable: {}".format(name))

    fig, (up, down) = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={"height_ratios": (2, 1)},
        sharex=True,
    )
    mc_hist, mc_bins, _ = up.hist(
        mc_uncorr,
        bins=bins,
        range=range,
        histtype="step",
        label=labels[1],
        density=True,
        weights=weights,
        linewidth=2,
        color="r",
    )
    if mc_corr is not None:
        mc_corr_hist, mc_corr_bins, _ = up.hist(
            mc_corr,
            bins=bins,
            range=range,
            histtype="step",
            label=labels[2],
            density=True,
            weights=weights,
            linewidth=2,
            color="b",
        )
    data_hist, data_bins = np.histogram(data, bins=bins, range=range, density=True)
    data_centers = (data_bins[1:] + data_bins[:-1]) / 2
    up.plot(
        data_centers,
        data_hist,
        label=labels[0],
        color="k",
        marker="o",
        linestyle="",
        markersize=3,
    )
    down.plot(
        data_centers,
        data_hist / mc_hist,
        color="r",
        marker="o",
        linestyle="",
        markersize=3,
    )
    if mc_corr is not None:
        down.plot(
            data_centers,
            data_hist / mc_corr_hist,
            color="b",
            marker="o",
            linestyle="",
            markersize=3,
        )

    if name in ["probe_pfChargedIsoPFPV", "probe_pfPhoIso03"]:
        up.set_yscale("log")
    if name == "probe_sieip" and "transformed" not in extra_name:
        ticks = [-0.0002, -0.0001, 0, 0.0001, 0.0002]
        down.set_xticks(ticks)
        down.set_xticklabels(ticks)
    down.set_xlabel(x_label)
    up.set_ylabel("Events / BinWidth")
    down.set_ylabel("Ratio")
    down.set_xlim(range[0], range[1])
    down.set_ylim(0.8, 1.2)
    down.axhline(
        1,
        color="grey",
        linestyle="--",
    )
    y_minor_ticks = np.arange(0.8, 1.2, 0.1)
    down.set_yticks(y_minor_ticks, minor=True)
    down.grid(True, alpha=0.4, which="minor")
    up.legend()
    # if probe_pt log scale
    hep.cms.label(
        loc=0, data=True, llabel="Work in Progress", rlabel="", ax=up, pad=0.05
    )
    fig_name = name + "_" + subdetector + extra_name
    if writer_epoch is not None:
        writer, epoch = writer_epoch
        writer.add_figure(fig_name, fig, epoch)
    if cometlogger_epoch is not None:
        comet_logger, epoch = cometlogger_epoch
        comet_logger.log_figure(fig_name, fig, step=epoch)
    if writer_epoch is None and cometlogger_epoch is None:
        for dr in output_dir:
            for ext in ["pdf", "png"]:
                fig.savefig(dr + "/" + fig_name + "." + ext, bbox_inches="tight")
    plt.close(fig)


def sample_and_plot_base(
    test_loader,
    model,
    model_name,
    epoch,
    writer,
    comet_logger,
    context_variables,
    target_variables,
    device,
    pipeline,
    calo,
):
    target_size = len(target_variables)
    with torch.no_grad():
        gen, reco, samples = [], [], []
        for context, target, weights, extra in test_loader:
            context = context.to(device)
            target = target.to(device)
            if "zuko" in model_name:
                sample = model(context).sample()
            else:
                sample = model.sample(num_samples=1, context=context)
            context = context.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            sample = sample.detach().cpu().numpy()
            sample = sample.reshape(-1, target_size)
            gen.append(context)
            reco.append(target)
            samples.append(sample)
    gen = np.concatenate(gen, axis=0)
    reco = np.concatenate(reco, axis=0)
    samples = np.concatenate(samples, axis=0)
    gen = pd.DataFrame(gen, columns=context_variables)
    reco = pd.DataFrame(reco, columns=target_variables)
    samples = pd.DataFrame(samples, columns=target_variables)

    # plot the reco and sampled distributions
    for var in target_variables:
        if device == 0 or type(device) != int:
            dump_main_plot(
                reco[var],
                samples[var],
                variable_conf={
                    "name": var,
                    "title": var,
                    "x_label": var,
                    "bins": 100,
                    "range": transformed_ranges[pipeline][var],
                },
                output_dir="",
                subdetector=calo,
                extra_name=f"_reco_sampled_transformed",
                writer_epoch=(writer, epoch),
                cometlogger_epoch=(comet_logger, epoch),
                labels=["Original", "Sampled"],
            )

    # plot after preprocessing back
    preprocess_dct = test_loader.dataset.pipelines
    reco_back = {}
    samples_back = {}
    with open(f"{script_dir}/../preprocess/var_specs.json", "r") as f:
        vars_config = json.load(f)
        vars_config = {d["name"]: d for d in vars_config}
    for var in target_variables:
        reco_back[var] = (
            preprocess_dct[var]
            .inverse_transform(reco[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        samples_back[var] = (
            preprocess_dct[var]
            .inverse_transform(samples[var].values.reshape(-1, 1))
            .reshape(-1)
        )
    reco_back = pd.DataFrame(reco_back)
    samples_back = pd.DataFrame(samples_back)
    for var in target_variables:
        dump_main_plot(
            reco_back[var],
            samples_back[var],
            variable_conf=vars_config[var],
            output_dir="",
            subdetector=calo,
            extra_name=f"_reco_sampled",
            writer_epoch=(writer, epoch),
            cometlogger_epoch=(comet_logger, epoch),
            labels=["Original", "Sampled"],
        )


def transform_and_plot_top(
    mc_loader,
    data_loader,
    model,
    epoch,
    writer,
    comet_logger,
    context_variables,
    target_variables,
    device,
    pipeline,
    calo,
):
    with torch.no_grad():
        data_lst, mc_lst, mc_corr_lst = [], [], []
        data_context_lst, mc_context_lst, mc_corr_context_lst = [], [], []
        mc_weights_lst = []
        data_extra_lst, mc_extr_lst = [], []
        for data, mc in zip(data_loader, mc_loader):
            context_data, target_data, weights_data, extra_data = data
            context_mc, target_mc, weights_mc, extra_mc = mc
            target_mc_corr, _ = model.transform(target_mc, context_mc, inverse=False)
            target_data = target_data.detach().cpu().numpy()
            target_mc = target_mc.detach().cpu().numpy()
            target_mc_corr = target_mc_corr.detach().cpu().numpy()
            context_data = context_data.detach().cpu().numpy()
            context_mc = context_mc.detach().cpu().numpy()
            weights_mc = weights_mc.detach().cpu().numpy()
            extra_data = extra_data.detach().cpu().numpy()
            extra_mc = extra_mc.detach().cpu().numpy()
            data_lst.append(target_data)
            mc_lst.append(target_mc)
            mc_corr_lst.append(target_mc_corr)
            data_context_lst.append(context_data)
            mc_context_lst.append(context_mc)
            mc_corr_context_lst.append(context_mc)
            mc_weights_lst.append(weights_mc)
            data_extra_lst.append(extra_data)
            mc_extr_lst.append(extra_mc)
    data = np.concatenate(data_lst, axis=0)
    mc = np.concatenate(mc_lst, axis=0)
    mc_corr = np.concatenate(mc_corr_lst, axis=0)
    data = pd.DataFrame(data, columns=target_variables)
    mc = pd.DataFrame(mc, columns=target_variables)
    mc_corr = pd.DataFrame(mc_corr, columns=target_variables)
    data_context = np.concatenate(data_context_lst, axis=0)
    mc_context = np.concatenate(mc_context_lst, axis=0)
    mc_corr_context = np.concatenate(mc_corr_context_lst, axis=0)
    data_context = pd.DataFrame(data_context, columns=context_variables)
    mc_context = pd.DataFrame(mc_context, columns=context_variables)
    mc_corr_context = pd.DataFrame(mc_corr_context, columns=context_variables)
    weights_mc = np.concatenate(mc_weights_lst, axis=0)
    data_extra = np.concatenate(data_extra_lst, axis=0)
    data_extra = pd.DataFrame(data_extra, columns=["probe_energyRaw"])
    mc_extra = np.concatenate(mc_extr_lst, axis=0)
    mc_extra = pd.DataFrame(mc_extra, columns=["probe_energyRaw"])

    # plot the reco and sampled distributions
    for var in target_variables:
        if device == 0 or type(device) != int:
            dump_main_plot(
                data[var],
                mc[var],
                variable_conf={
                    "name": var,
                    "title": var,
                    "x_label": var,
                    "bins": 100,
                    "range": transformed_ranges[pipeline][var],
                }, 
                output_dir="",
                subdetector=calo,
                mc_corr=mc_corr[var],
                weights=weights_mc,
                extra_name=f"_top_transformed",
                writer_epoch=(writer, epoch),
                cometlogger_epoch=(comet_logger, epoch),
                labels=None,
            )

    # sample back
    # note that pipelines are actually the same, trained on data
    data_pipeline = data_loader.dataset.pipelines
    mc_pipeline = mc_loader.dataset.pipelines

    with open(f"{script_dir}/../preprocess/var_specs.json", "r") as f:
        vars_config = json.load(f)
        vars_config = {d["name"]: d for d in vars_config}

    for var in target_variables:
        data[var] = (
            data_pipeline[var]
            .inverse_transform(data[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc[var] = (
            mc_pipeline[var]
            .inverse_transform(mc[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc_corr[var] = (
            mc_pipeline[var]
            .inverse_transform(mc_corr[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        if device == 0 or type(device) != int:
            dump_main_plot(
                data[var],
                mc[var],
                variable_conf=vars_config[var],
                output_dir="",
                subdetector=calo,
                mc_corr=mc_corr[var],
                weights=weights_mc,
                extra_name="_top",
                writer_epoch=(writer, epoch),
                cometlogger_epoch=(comet_logger, epoch),
                labels=None,
            )

    for var in context_variables:
        data_context[var] = (
            data_pipeline[var]
            .inverse_transform(data_context[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc_context[var] = (
            mc_pipeline[var]
            .inverse_transform(mc_context[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc_corr_context[var] = (
            mc_pipeline[var]
            .inverse_transform(mc_corr_context[var].values.reshape(-1, 1))
            .reshape(-1)
        )

    # photon ID
    # make dataframes by merging context, target and extra
    data_df = pd.concat([data, data_context, data_extra], axis=1)
    mc_df = pd.concat([mc, mc_context, mc_extra], axis=1)
    mc_corr_df = pd.concat([mc_corr, mc_corr_context, mc_extra], axis=1)

    pho_id_data = calculate_photonid_mva(data_df, calo=calo)
    pho_id_mc = calculate_photonid_mva(mc_df, calo=calo)
    pho_id_mc_corr = calculate_photonid_mva(mc_corr_df, calo=calo)
    
    dump_main_plot(
        pho_id_data,
        pho_id_mc,
        vars_config["probe_mvaID"],
        output_dir="",
        subdetector=calo,
        mc_corr=pho_id_mc_corr,
        weights=weights_mc,
        extra_name="_top",
        labels=None,
        writer_epoch=(writer, epoch), 
        cometlogger_epoch=(comet_logger, epoch),
    )

    # now plot profiles
    nbins = 8
    for column in target_variables:
        for cond_column in context_variables:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            data_ss_arr = data[column].values
            data_cond_arr = data_context[cond_column].values
            mc_uncorr_ss_arr = mc[column].values
            mc_uncorr_cond_arr = mc_context[cond_column].values
            mc_corr_ss_arr = mc_corr[column].values
            mc_corr_cond_arr = mc_corr_context[cond_column].values
            cond_edges = divide_dist(data_cond_arr, nbins)

            for name, ss_arr, cond_arr, color in [
                ("data", data_ss_arr, data_cond_arr, "blue"),
                ("mc", mc_uncorr_ss_arr, mc_uncorr_cond_arr, "red"),
                ("mc corr", mc_corr_ss_arr, mc_corr_cond_arr, "green"),
            ]:
                ax = dump_profile_plot(
                    ax=ax,
                    ss_name=column,
                    cond_name=cond_column,
                    sample_name=name,
                    ss_arr=ss_arr,
                    cond_arr=cond_arr,
                    color=color,
                    cond_edges=cond_edges,
                )
            ax.legend()
            ax.set_xlabel(cond_column)
            ax.set_ylabel(column)
            # reduce dimension of labels and axes names
            plt.rcParams["axes.labelsize"] = 12
            plt.rcParams["xtick.labelsize"] = 12
            plt.rcParams["ytick.labelsize"] = 12
            plt.rcParams["legend.fontsize"] = 12
            fig.tight_layout()
            if writer is not None:
                writer.add_figure(
                    f"profiles_{column}_{cond_column}", fig, epoch
                )
                comet_logger.log_figure(
                    f"profiles_{column}_{cond_column}", fig, step=epoch
                )

    # close figures
    plt.close("all")

    
def plot_one(
    mc_test_loader,
    data_test_loader,
    model,
    model_name,
    epoch,
    writer,
    context_variables,
    target_variables,
    device,
    pipeline,
    calo,
):
    with torch.no_grad():
        data_lst, mc_lst, mc_corr_lst = [], [], []
        data_context_lst, mc_context_lst, mc_corr_context_lst = [], [], []
        mc_weights_lst = []
        data_extra_lst, mc_extr_lst = [], []
        for (data_context, data_target, data_weights, data_extra), (mc_context, mc_target, mc_weights, mc_extra) in zip(
            data_test_loader, mc_test_loader
        ):
            data_context = data_context.to(device)
            data_target = data_target.to(device)
            mc_context = mc_context.to(device)
            mc_target = mc_target.to(device)
            if "zuko" in model_name:
                latent_mc = model(mc_context).transform(mc_target)
            else:
                latent_mc = model._transform(mc_target, context=mc_context)[0]
            # replace the last column in mc_context with 0 instead of 1
            mc_context[:, -1] = 0
            if "zuko" in model_name:
                mc_target_corr = model(mc_context).transform.inv(latent_mc)
            else:
                mc_target_corr = model._transform.inverse(latent_mc, context=mc_context)[0]
            data_target = data_target.detach().cpu().numpy()
            data_context = data_context.detach().cpu().numpy()
            data_extra = data_extra.detach().cpu().numpy()
            mc_target = mc_target.detach().cpu().numpy()
            mc_target_corr = mc_target_corr.detach().cpu().numpy()
            mc_context = mc_context.detach().cpu().numpy()
            mc_extra = mc_extra.detach().cpu().numpy()
            mc_weights = mc_weights.detach().cpu().numpy()
            data_lst.append(data_target)
            data_context_lst.append(data_context)
            data_extra_lst.append(data_extra)
            mc_lst.append(mc_target)
            mc_corr_lst.append(mc_target_corr)
            mc_context_lst.append(mc_context)
            mc_weights_lst.append(mc_weights)
            mc_extr_lst.append(mc_extra)
    data = np.concatenate(data_lst, axis=0)
    mc = np.concatenate(mc_lst, axis=0)
    mc_corr = np.concatenate(mc_corr_lst, axis=0)
    data = pd.DataFrame(data, columns=target_variables)
    mc = pd.DataFrame(mc, columns=target_variables)
    mc_corr = pd.DataFrame(mc_corr, columns=target_variables)
    data_context = np.concatenate(data_context_lst, axis=0)
    mc_context = np.concatenate(mc_context_lst, axis=0)
    # remove the last column from mc_context
    mc_context = mc_context[:, :-1]
    data_context = pd.DataFrame(data_context, columns=context_variables)
    mc_context = pd.DataFrame(mc_context, columns=context_variables)
    mc_weights = np.concatenate(mc_weights_lst, axis=0)
    data_extra = np.concatenate(data_extra_lst, axis=0)
    data_extra = pd.DataFrame(data_extra, columns=["probe_energyRaw"])
    mc_extra = np.concatenate(mc_extr_lst, axis=0)
    mc_extra = pd.DataFrame(mc_extra, columns=["probe_energyRaw"])

    for var in target_variables:
        if device == 0 or type(device) != int:
            dump_main_plot(
                data[var],
                mc[var],
                variable_conf={
                    "name": var,
                    "title": var,
                    "x_label": var,
                    "bins": 100,
                    "range": transformed_ranges[pipeline][var],
                },
                output_dir="",
                subdetector=calo,
                mc_corr=mc_corr[var],
                weights=mc_weights,
                extra_name="_one_transformed",
                writer_epoch=(writer, epoch),
                labels=None,
            )

    # sample back
    pipeline = mc_test_loader.dataset.pipelines

    with open(f"{script_dir}/../preprocess/var_specs.json", "r") as f:
        vars_config = json.load(f)
        vars_config = {d["name"]: d for d in vars_config}

    for var in target_variables:
        data[var] = (
            pipeline[var]
            .inverse_transform(data[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc[var] = (
            pipeline[var]
            .inverse_transform(mc[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc_corr[var] = (
            pipeline[var]
            .inverse_transform(mc_corr[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        if device == 0 or type(device) != int:
            dump_main_plot(
                data[var],
                mc[var],
                variable_conf=vars_config[var],
                output_dir="",
                subdetector=calo,
                mc_corr=mc_corr[var],
                weights=mc_weights,
                extra_name="_one",
                writer_epoch=(writer, epoch),
                labels=None,
            )

    for var in context_variables:
        data_context[var] = (
            pipeline[var]
            .inverse_transform(data_context[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        mc_context[var] = (
            pipeline[var]
            .inverse_transform(mc_context[var].values.reshape(-1, 1))
            .reshape(-1)
        )

    # photon ID
    # make dataframes by merging context, target and extra
    data_df = pd.concat([data, data_context, data_extra], axis=1)
    mc_df = pd.concat([mc, mc_context, mc_extra], axis=1)
    mc_corr_df = pd.concat([mc_corr, mc_context, mc_extra], axis=1)

    pho_id_data = calculate_photonid_mva(data_df, calo=calo)
    pho_id_mc = calculate_photonid_mva(mc_df, calo=calo)
    pho_id_mc_corr = calculate_photonid_mva(mc_corr_df, calo=calo)
    
    dump_main_plot(
        pho_id_data,
        pho_id_mc,
        vars_config["probe_mvaID"],
        output_dir="",
        subdetector=calo,
        mc_corr=pho_id_mc_corr,
        weights=mc_weights,
        extra_name="_one",
        labels=None,
        writer_epoch=(writer, epoch), 
    )