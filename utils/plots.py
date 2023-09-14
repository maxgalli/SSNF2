import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
hep.style.use("CMS")
import logging
logger = logging.getLogger(__name__)

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
    }
}


def dump_main_plot(
    data,
    mc_uncorr,
    variable_conf,
    output_dir,
    subdetector,
    mc_corr=None,
    weights=None,
    extra_name="",
):
    name = variable_conf["name"]
    title = variable_conf["title"] + "_" + subdetector
    x_label = variable_conf["x_label"]
    bins = variable_conf["bins"]
    range = variable_conf["range"]

    # specific ranges for EB and EE
    if name == "probe_sieie" and subdetector == "EE":
        range = [0.005, 0.04]

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
        label=f"MC - {subdetector} (uncorr.)",
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
            label=f"MC - {subdetector}",
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
        label=f"Data - {subdetector}",
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
    down.set_ylabel("Data / MC")
    down.set_xlim(range[0], range[1])
    down.set_ylim(0.5, 1.5)
    down.axhline(
        1,
        color="grey",
        linestyle="--",
    )
    y_minor_ticks = np.arange(0.5, 1.5, 0.1)
    down.set_yticks(y_minor_ticks, minor=True)
    down.grid(True, alpha=0.4, which="minor")
    up.legend()
    hep.cms.label(
        loc=0, data=True, llabel="Work in Progress", rlabel="", ax=up, pad=0.05
    )
    fig_name = name + "_" + subdetector + extra_name
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir + "/" + fig_name + "." + ext, bbox_inches="tight")
    plt.close(fig)