import warnings
import xgboost


phoid_files = {
    "eb": "/work/gallim/devel/CQRRelatedStudies/SSNF2/preprocess/XGB_Model_Barrel_SA_phoID_UL2017_woCorr.json",
    "ee": "/work/gallim/devel/CQRRelatedStudies/SSNF2/preprocess/XGB_Model_Endcap_SA_phoID_UL2017_woCorr.json"
}


def load_photonid_mva(fname):
    try:
        photonid_mva = xgboost.Booster()
        photonid_mva.load_model(fname)
    except xgboost.core.XGBoostError:
        warnings.warn(f"SKIPPING photonid_mva, could not find: {fname}")
        photonid_mva = None
    return photonid_mva


def calculate_photonid_mva(dataframe, calo):
    """Recompute PhotonIDMVA on-the-fly. This step is necessary considering that the inputs have to be corrected
    with the QRC process. Following is the list of features (barrel has 12, endcap two more):
    EB:
        events.Photon.energyRaw
        events.Photon.r9
        events.Photon.sieie
        events.Photon.etaWidth
        events.Photon.phiWidth
        events.Photon.sieip
        events.Photon.s4
        events.Photon.pfPhoIso03
        events.Photon.pfChargedIsoPFPV
        events.Photon.pfChargedIsoWorstVtx
        events.Photon.eta
        events.fixedGridRhoAll

    EE: EB +
        events.Photon.esEffSigmaRR
        events.Photon.esEnergyOverRawE
    """
    var_order = [
        "probe_energyRaw",
        "probe_r9",
        "probe_sieie",
        "probe_etaWidth",
        "probe_phiWidth",
        "probe_sieip",
        "probe_s4",
        "probe_pfPhoIso03",
        "probe_pfChargedIsoPFPV",
        "probe_pfChargedIsoWorstVtx",
        "probe_eta",
        "probe_fixedGridRhoAll",
    ]
    if calo == "ee":
        var_order += ["probe_esEffSigmaRR", "probe_esEnergyOverRawE"]

    photonid_mva = load_photonid_mva(phoid_files[calo])

    bdt_inputs = dataframe[var_order].to_numpy()
    tempmatrix = xgboost.DMatrix(bdt_inputs, feature_names=[var.replace("probe_", "") for var in var_order])

    mvaID = photonid_mva.predict(tempmatrix)

    # Only needed to compare to TMVA
    # mvaID = 1.0 - 2.0 / (1.0 + numpy.exp(2.0 * mvaID))

    # the previous transformation was not working correctly, peakin at about 0.7
    # since we can't really remember why that functional form was picked in the first place we decided
    # to switch to a simpler stretch of the output that works better, even though not perfectly.
    # Open for changes/ideas
    mvaID = -1 + 2 * mvaID

    return mvaID