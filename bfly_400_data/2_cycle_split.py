import os
import sys
import pandas as pd
import logging
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
import hdbscan
from lmfit import minimize, Parameters, fit_report
from typing import Tuple

from bfly400.utils.bfly_viz import bfly400_plot
from bfly400.utils.utils import calculate_observed_fraction

log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
log_level = logging.INFO
LOG = logging.getLogger(__name__)
LOG.setLevel(log_level)

# writing to stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(log_level)
handler.setFormatter(log_format)
LOG.addHandler(handler)

YRCUM = 1

DATA_FOLDER = "/home/amunozj/git_repos/Bfly_diagram/data/"
OUTPUT_FOLDER = "/home/amunozj/git_repos/Bfly_diagram/data/output_data/"
FIGURE_FOLDER = "/home/amunozj/git_repos/Bfly_diagram/figures/"

if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

if not os.path.exists(FIGURE_FOLDER):
    os.mkdir(FIGURE_FOLDER)


def normalized_KDE(
    BflyAllDF_NoSS: pd.DataFrame,
    ybinl: float = 60,
    ybin_size: float = 0.25,
    xbin1: float = 1600,
    xbin2: float = 2024,
    xbin_size: float = 0.5,
) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Function that calculates the normalized Exponential Kernel Density estimate used
    to identify the boundary between cycles

    Parameters
    ----------
    BflyAllDF_NoSS : pd.DataFrame
        Dataframe holding all data. It expects spotless days to be removed
    ybinl : float, optional
        Maximum latitude used to define the latitudinal grid in degrees, by default 60
    ybin_size : float, optional
        Size of the latitudinal grid in degrees, by default 0.25
    xbin1 : float, optional
        Lower limit of the time grid in years, by default 1600
    xbin2 : float, optional
        Upper limit of the time grid in years, by default 2024
    xbin_size : float, optional
        Size of the time grid in years, by default 0.5

    Returns
    -------
    Tuple[np.array, np.array, np.array, np.array, np.array]
        Returns the centers and edges of the latitude vs. time grid
        as well as density normalize with a running window
    """

    binsy = np.arange(-ybinl, ybinl + ybin_size, ybin_size)
    centersy = (binsy[1:] + binsy[0:-1]) / 2

    binsx = np.arange(xbin1, xbin2 + xbin_size, xbin_size)
    centersx = (binsx[1:] + binsx[0:-1]) / 2

    xv, yv = np.meshgrid(centersx, centersy, indexing="xy")
    positions = np.vstack([xv.ravel(), yv.ravel()])

    values = np.vstack(
        [BflyAllDF_NoSS["FRACYEAR"].values, BflyAllDF_NoSS["LATITUDE"].values]
    )

    kde = KernelDensity(bandwidth=0.75, kernel="exponential", algorithm="auto")

    kde.fit(values.T)

    Z2 = np.reshape(np.exp(kde.score_samples(positions.T)), xv.shape)
    mdn_width = 12
    KDE_norm = Z2 / median_filter(np.max(Z2, axis=0), mdn_width)[None, :]

    return centersx, centersy, binsx, binsy, KDE_norm


def find_bfly_separators(
    centersx: np.array,
    centersy: np.array,
    KDE_norm: np.array,
    distance: float = 17,
    prominence: float = 0.1,
    height: float = 0.6,
    min_cluster_size: int = 20,
    min_samples: int = 20,
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 5,
    alpha: float = 1.0,
) -> Tuple[np.array, np.array, object, object]:
    """
    Function that uses a normalized KDE to find the points that
    separate cycle wings.  It uses HDBSCAN clustering to deal
    with a non-uniform amount of points as different wings have
    different latitudinal extent.

    Parameters
    ----------
    centersx : np.array
        centers of the time grid
    centersy : np.array
        centers of the latitudinal grid
    KDE_norm : np.array
        Kernel density estimation of the butterfly diagram
    distance : float, optional
        Required minimal horizontal distance (>= 1) in samples
        between subsequent separators used in peak finding, by default 17
    prominence : float, optional
        Minimal prominence of separators used in peak finding, by default 0.1
    height : float, optional
        Minimal height of separators used in peak finding, by default 0.6
    min_cluster_size : int, optional
        Min cluster sized used in HDBSCAN, by default 20
    min_samples : int, optional
        Min samples used in HDBSCAN, by default 20
    cluster_selection_method : str, optional
        Cluster selection method used in HDBSCAN, by default "eom"
    cluster_selection_epsilon : float, optional
        Cluster selection epsilon used in HDBSCAN, by default 5
    alpha : float, optional
        Alpha used in HDBSCAN, by default 1.0

    Returns
    -------
    Tuple[np.array, np.array, object, object]
        arrays with all the points identified as separator candidates for the
        North and South hemispheres, as well as the clusters associated with each
        hemispheric cycle
    """

    boundary_listN = []
    boundary_listS = []
    for anchor_lat in np.arange(2, 35, 0.25):
        achor_iS = np.sum(centersy <= -anchor_lat) - 1
        achor_iN = np.sum(centersy <= anchor_lat)
        peaksN, properties = find_peaks(
            -KDE_norm[achor_iN, :],
            distance=distance,
            prominence=prominence,
            height=-height,
        )
        peaksS, properties = find_peaks(
            -KDE_norm[achor_iS, :],
            distance=distance,
            prominence=prominence,
            height=-height,
        )

        boundary_listN.append(
            np.concatenate(
                (
                    centersx[peaksN][:, None],
                    (centersx[peaksN] * 0 + centersy[achor_iN])[:, None],
                ),
                axis=1,
            )
        )
        boundary_listS.append(
            np.concatenate(
                (
                    centersx[peaksS][:, None],
                    (centersx[peaksS] * 0 + centersy[achor_iS])[:, None],
                ),
                axis=1,
            )
        )

    boundary_N = np.concatenate(boundary_listN, axis=0)
    boundary_S = np.concatenate(boundary_listS, axis=0)

    clustererN = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_method=cluster_selection_method,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        alpha=alpha,
    ).fit(boundary_N)

    clustererS = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_method=cluster_selection_method,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        alpha=alpha,
    ).fit(boundary_S)

    return boundary_N, boundary_S, clustererN, clustererS


def power_fit(
    params: object,
    boundary_N: np.array,
    labelsN: np.array,
    boundary_S: np.array,
    labelsS: np.array,
) -> np.array:
    """
    Power law used to fit the boundaries of all cycles

    Parameters
    ----------
    params : object
        Object containing all the fitting parameters, the starting values,
        and their boundaries
    boundary_N : np.array
        Points identified to be separators for the Northern hemisphere
    labelsN : np.array
        Cluster membership of boundary_N points
    boundary_S : np.array
        Points identified to be separators for the Southern hemisphere
    labelsS : np.array
        Cluster membership of boundary_S points

    Returns
    -------
    np.array
        Residuals of the model against the fitted points
    """

    residuals = []
    # Northern hemisphere
    for label in np.unique(labelsN):
        if label > -1:
            x = params[f"OSN{label}"] - boundary_N[labelsN == label, 0]
            model = params[f"SLN{label}"] * np.power(x, params["b"])
            residual = boundary_N[labelsN == label, 1] - model
            residuals.append(residual)

    for label in np.unique(labelsS):
        if label > -1:
            x = params[f"OSS{label}"] - boundary_S[labelsS == label, 0]
            model = -params[f"SLS{label}"] * np.power(x, params["b"])
            residual = boundary_S[labelsS == label, 1] - model
            residuals.append(residual)

    return np.concatenate(residuals)


def parse_fit_params(fit_output: object) -> pd.DataFrame:
    """
    Function that turns the output of the power law fit to the
    separators into a Pandas dataframe that can be saved as a csv

    Parameters
    ----------
    fit_output : object
        Output of the power law fit

    Returns
    -------
    pd.DataFrame
        Tabulated fitting parameters
    """    

    paramsDF = pd.DataFrame(
        {
            "keys": fit_output.params.valuesdict().keys(),
            "values": fit_output.params.valuesdict().values(),
        }
    )

    paramsDF_dic = {}
    for hem in ["N", "S"]:

        tmp1 = (
            paramsDF.loc[paramsDF["keys"].str.contains(f"OS{hem}", na=True), :]
            .sort_values("values", ascending=False)
            .reset_index(drop=True)
        )
        tmp1 = tmp1.rename(columns={"keys": f"keyI{hem}", "values": f"bndr_offst{hem}"})
        tmp1["index"] = tmp1[f"keyI{hem}"].apply(lambda x: x.split(f"OS{hem}")[1])
        tmp1 = tmp1.set_index("index").drop(columns=f"keyI{hem}")

        tmp2 = (
            paramsDF.loc[paramsDF["keys"].str.contains(f"SL{hem}", na=True), :]
            .sort_values("values", ascending=False)
            .reset_index(drop=True)
        )
        tmp2 = tmp2.rename(columns={f"keys": f"keyS{hem}", "values": f"bndr_slp{hem}"})
        tmp2["index"] = tmp2[f"keyS{hem}"].apply(lambda x: x.split(f"SL{hem}")[1])
        tmp2 = tmp2.set_index("index").drop(columns=f"keyS{hem}")

        paramsDF_dic[hem] = (
            pd.concat([tmp1, tmp2], axis=1)
            .sort_values(f"bndr_offst{hem}", ascending=False)
            .reset_index(drop=True)
        )

    cycle_fits = pd.DataFrame({"Cycle": np.arange(25, -11, -1)})
    cycle_fits = pd.concat([cycle_fits, paramsDF_dic["N"], paramsDF_dic["S"]], axis=1)
    cycle_fits["CycleN"] = cycle_fits["Cycle"].values.astype(int).astype(str)
    cycle_fits["bndr_expnt"] = fit_output.params["b"].value

    return cycle_fits


if __name__ == "__main__":

    gn = {}

    LOG.info("Read Group sunspot data...")

    gn["Svalgaard"] = pd.read_table(
        DATA_FOLDER + "input_data/GNbb2_y.txt", delimiter=r"\s+", engine="python"
    )

    gn["Usoskin"] = pd.read_table(
        DATA_FOLDER + "input_data/GNiu_y2.txt", delimiter=r"\s+", engine="python"
    )

    # -------------------------------------------------------------------------------------------
    LOG.info("Read consolidated Butterfly data...")
    BflyAllDF = pd.read_csv(
        OUTPUT_FOLDER + "1_BflyAll_gassendi_Vaquero.csv", parse_dates=True
    )

    BflyAllDF_NoSS = BflyAllDF.loc[
        np.logical_and(
            np.isfinite(BflyAllDF["LATITUDE"]), np.logical_not(BflyAllDF["AREA"] == 0)
        ),
        :,
    ]
    BfObsCv_NoSS, BfObsYr_NoSS = calculate_observed_fraction(
        BflyAllDF_NoSS, Y1=1600, Y2=2024, YrCum=YRCUM
    )

    # -------------------------------------------------------------------------------------------
    LOG.info("Calculating normalized KDE density...")

    centersx, centersy, binsx, binsy, KDE_norm = normalized_KDE(BflyAllDF_NoSS)

    # -------------------------------------------------------------------------------------------
    LOG.info("Identifying Bfly separators...")

    boundary_N, boundary_S, clustererN, clustererS = find_bfly_separators(
        centersx, centersy, KDE_norm
    )

    # -------------------------------------------------------------------------------------------
    LOG.info("Fitting Bfly separators...")

    power_params = Parameters()
    # Common exponent
    power_params.add("b", value=1.5, min=1.5)
    # Different offsets
    for label in np.unique(clustererN.labels_):
        if label > -1:
            power_params.add(
                f"OSN{label}",
                value=np.max(boundary_N[clustererN.labels_ == label, 0]) + 2,
                min=np.max(boundary_N[clustererN.labels_ == label, 0]),
            )
            power_params.add(f"SLN{label}", value=6, min=1.5)
    for label in np.unique(clustererS.labels_):
        if label > -1:
            power_params.add(
                f"OSS{label}",
                value=np.max(boundary_S[clustererS.labels_ == label, 0]) + 2,
                min=np.max(boundary_S[clustererS.labels_ == label, 0]),
            )
            power_params.add(f"SLS{label}", value=6, min=1.5)

    out = minimize(
        power_fit,
        power_params,
        method="BFGS",
        kws={
            "boundary_N": boundary_N,
            "labelsN": clustererN.labels_,
            "boundary_S": boundary_S,
            "labelsS": clustererS.labels_,
        },
    )

    print(fit_report(out))

    # -------------------------------------------------------------------------------------------
    LOG.info("Constructing fitting database...")

    cycle_fits = parse_fit_params(out)

    # -------------------------------------------------------------------------------------------
    LOG.info("Manual modifications to fit...")

    # ### Move last rows to -8 and -9
    cycle_fits.loc[[33, 34], ["bndr_offstN", "bndr_slpN"]] = cycle_fits.loc[
        [30, 31], ["bndr_offstN", "bndr_slpN"]
    ].values
    cycle_fits.loc[[33, 34], ["bndr_offstS", "bndr_slpS"]] = cycle_fits.loc[
        [31, 32], ["bndr_offstS", "bndr_slpS"]
    ].values

    # ### Shift 8 earlier
    cycle_fits.loc[17, "bndr_offstN"] = cycle_fits.loc[17, "bndr_offstN"] - 2
    cycle_fits.loc[17, "bndr_slpN"] = cycle_fits.loc[17, "bndr_slpN"] * 2

    # ### Increase 7 slope
    cycle_fits.loc[18, "bndr_slpN"] = cycle_fits.loc[18, "bndr_slpN"] * 3
    cycle_fits.loc[18, "bndr_slpS"] = cycle_fits.loc[18, "bndr_slpS"] * 3

    # ### Shift 6 south and increase slope
    cycle_fits.loc[19, "bndr_offstS"] = cycle_fits.loc[19, "bndr_offstS"] - 4
    cycle_fits.loc[19, "bndr_slpS"] = cycle_fits.loc[19, "bndr_slpS"] * 2

    # ### Increase the slope of 5 South
    cycle_fits.loc[20, "bndr_slpS"] = cycle_fits.loc[20, "bndr_slpS"] * 3

    # ### Increase slope of 4 North
    cycle_fits.loc[21, "bndr_slpN"] = cycle_fits.loc[21, "bndr_slpN"] * 2
    cycle_fits.loc[21, "bndr_offstN"] = cycle_fits.loc[21, "bndr_offstN"] - 2

    # ### Increase slope of 2 North
    cycle_fits.loc[23, "bndr_offstN"] = cycle_fits.loc[23, "bndr_offstN"] - 2
    cycle_fits.loc[23, "bndr_slpN"] = cycle_fits.loc[23, "bndr_slpN"] * 2.5

    cycle_fits.loc[23, "bndr_offstS"] = cycle_fits.loc[23, "bndr_offstS"] - 1
    cycle_fits.loc[23, "bndr_slpS"] = cycle_fits.loc[23, "bndr_slpS"] * 2

    # ### Increase slope of 1 South
    cycle_fits.loc[24, "bndr_slpS"] = cycle_fits.loc[24, "bndr_slpS"] * 2
    cycle_fits.loc[24, "bndr_offstS"] = cycle_fits.loc[24, "bndr_offstS"] - 1

    # ### Ensure that -3 North matches the South
    cycle_fits.loc[28, "bndr_offstN"] = cycle_fits.loc[28, "bndr_offstS"]
    cycle_fits.loc[28, "bndr_slpN"] = cycle_fits.loc[28, "bndr_slpS"]

    # ### Shift -4 earlier and match North to South
    cycle_fits.loc[29, "bndr_offstS"] = cycle_fits.loc[29, "bndr_offstS"]
    cycle_fits.loc[29, "bndr_offstN"] = cycle_fits.loc[29, "bndr_offstS"]

    # ### Shift -5 earlier and match North to South
    cycle_fits.loc[30, "bndr_offstS"] = cycle_fits.loc[30, "bndr_offstS"] + 12
    cycle_fits.loc[30, "bndr_offstN"] = cycle_fits.loc[30, "bndr_offstS"]

    # ### Add -6
    cycle_fits.loc[31, "bndr_offstS"] = cycle_fits.loc[30, "bndr_offstS"] - 11
    cycle_fits.loc[31, "bndr_slpS"] = cycle_fits.loc[30, "bndr_slpS"]

    cycle_fits.loc[31, "bndr_offstN"] = cycle_fits.loc[31, "bndr_offstS"]
    cycle_fits.loc[31, "bndr_slpN"] = cycle_fits.loc[31, "bndr_slpS"]

    # ### Add -7
    cycle_fits.loc[32, "bndr_offstS"] = cycle_fits.loc[31, "bndr_offstS"] - 13
    cycle_fits.loc[32, "bndr_slpS"] = cycle_fits.loc[31, "bndr_slpS"]

    cycle_fits.loc[32, "bndr_offstN"] = cycle_fits.loc[32, "bndr_offstS"] - 3
    cycle_fits.loc[32, "bndr_slpN"] = cycle_fits.loc[32, "bndr_slpS"]

    # ### Shift -8 North later
    cycle_fits.loc[33, "bndr_offstN"] = cycle_fits.loc[33, "bndr_offstN"] + 2
    cycle_fits.loc[33, "bndr_offstS"] = cycle_fits.loc[33, "bndr_offstN"]
    cycle_fits.loc[33, "bndr_slpS"] = cycle_fits.loc[3, "bndr_slpS"]

    # ### Shift -9 North earlier
    cycle_fits.loc[34, "bndr_offstN"] = cycle_fits.loc[34, "bndr_offstN"] - 2
    cycle_fits.loc[34, "bndr_slpS"] = cycle_fits.loc[34, "bndr_slpS"] * 2
    cycle_fits.loc[34, "bndr_slpN"] = cycle_fits.loc[34, "bndr_slpS"]

    # ### Rename Historic Cycles
    cycle_fits.loc[[29, 30, 31, 32], ["CycleN"]] = ["M4", "M3", "M2", "M1"]
    cycle_fits.loc[[33, 34, 35], ["CycleN"]] = ["BM3", "BM2", "BM1"]

    # ### Add cycle -10
    cycle_fits.loc[35, "bndr_offstS"] = cycle_fits.loc[34, "bndr_offstS"] - 11
    cycle_fits.loc[35, "bndr_slpS"] = cycle_fits.loc[34, "bndr_slpS"]

    cycle_fits.loc[35, "bndr_offstN"] = cycle_fits.loc[35, "bndr_offstS"]
    cycle_fits.loc[35, "bndr_slpN"] = cycle_fits.loc[35, "bndr_slpS"]

    # -------------------------------------------------------------------------------------------
    LOG.info("Assign cycle numbers to all spots...")

    BflyAllDF["CYCLE"] = np.nan
    BflyAllDF["CYCLEN"] = ""

    for index, row in cycle_fits.iterrows():
        x = row["bndr_offstN"] - BflyAllDF["FRACYEAR"]
        y = np.power(x, out.params["b"].value) * row["bndr_slpN"]

        maskN = np.logical_and(
            BflyAllDF["FRACYEAR"] >= row["bndr_offstN"], BflyAllDF["LATITUDE"] > 0
        )
        maskN = np.logical_or(maskN, BflyAllDF["LATITUDE"] >= y)

        x = row["bndr_offstS"] - BflyAllDF["FRACYEAR"]
        y = np.power(x, out.params["b"].value) * row["bndr_slpS"]

        maskS = np.logical_and(
            BflyAllDF["FRACYEAR"] >= row["bndr_offstS"], BflyAllDF["LATITUDE"] < 0
        )
        maskS = np.logical_or(maskS, BflyAllDF["LATITUDE"] <= -y)

        mask = np.logical_or(maskN, maskS)
        mask = np.logical_and(mask, np.isnan(BflyAllDF["CYCLE"]))

        BflyAllDF.loc[mask, "CYCLE"] = row["Cycle"]
        BflyAllDF.loc[mask, "CYCLEN"] = row["CycleN"]

    # Remove cycle assignment from Gassendi observations
    BflyAllDF.loc[BflyAllDF["OBSERVER"] == "Gassendi", "CYCLE"] = np.nan
    BflyAllDF.loc[BflyAllDF["OBSERVER"] == "Gassendi", "CYCLEN"] = ""
    BflyAllDF.loc[BflyAllDF["OBSERVER"] == "Marcgraf", "CYCLE"] = np.nan
    BflyAllDF.loc[BflyAllDF["OBSERVER"] == "Marcgraf", "CYCLEN"] = ""

    # -------------------------------------------------------------------------------------------
    LOG.info("Save data...")

    cycle_fits = cycle_fits.loc[
        :,
        [
            "Cycle",
            "CycleN",
            "bndr_expnt",
            "bndr_offstN",
            "bndr_slpN",
            "bndr_offstS",
            "bndr_slpS",
        ],
    ]
    BflyAllDF.to_csv(OUTPUT_FOLDER + "2_BflyAll_cycle_split.csv", index=False)
    cycle_fits.to_csv(OUTPUT_FOLDER + "2_cycle_split_params.csv", index=False)

    # -------------------------------------------------------------------------------------------
    LOG.info("Plot...")

    BfObsCv, BfObsYr = calculate_observed_fraction(
        BflyAllDF, Y1=1600, Y2=2024, YrCum=YRCUM
    )

    bfly400_plot(
        FIGURE_FOLDER + f"2_cycle_split.png",
        BflyAllDF,
        BfObsYr,
        BfObsCv,
        gn,
        YrCum=YRCUM,
        cycle_fits=cycle_fits,
        KDE_norm=KDE_norm,
        binsx=binsx,
        binsy=binsy,
    )
