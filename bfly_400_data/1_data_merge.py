# Modules
import os
import sys
import numpy as np
import pandas as pd
import datetime
import logging

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


def bad_line_printer(x:str)->str:
    """
    Convenience function to deal with problematic tabulated files.
    It prints result so that the human can find the problem and fix it

    Parameters
    ----------
    x : str
        Line read to print for diagnosis

    Returns
    -------
    str
        Input line
    """    
    print(x)
    return x


def bad_line_ignore(x:str)->str:    
    """
    Convenience function to deal with problematic tabulated files.
    It ignores the line and does nothing else

    Parameters
    ----------
    x : str
        Line read to print for diagnosis

    Returns
    -------
    str
        Input line
    """
    return x


if __name__ == "__main__":

    gn = {}

    LOG.info("Read Group sunspot data...")

    gn["Svalgaard"] = pd.read_table(
        DATA_FOLDER + "input_data/GNbb2_y.txt", delimiter=r"\s+", engine="python"
    )

    gn["Usoskin"] = pd.read_table(
        DATA_FOLDER + "input_data/GNiu_y2.txt", delimiter=r"\s+", engine="python"
    )

    # --------------------------------------------------------------------------------------------
    # AMJ 2019
    LOG.info("Read AMJ 2019...")

    BflyMod = pd.read_csv(
        DATA_FOLDER
        + "input_data/composite_sunspot_groups_daily_measurements_10_23.csv",
        quotechar='"',
    )
    BflyMod = BflyMod.loc[np.isfinite(BflyMod["survey"]), :]
    BflyMod["ORDINAL"] = BflyMod.apply(
        lambda x: datetime.datetime(
            int(x["year"]),
            int(x["month"]),
            int(x["day"]),
            int(x["hour"]),
            int(x["minute"]),
            0,
        ).toordinal(),
        axis=1,
    )

    BflyMod["FRACYEAR"] = BflyMod.apply(
        lambda x: x["year"]
        + (
            datetime.date(int(x["year"]), int(x["month"]), int(x["day"])).toordinal()
            - datetime.date(int(x["year"]), 1, 1).toordinal()
        )
        / (
            datetime.date(int(x["year"]) + 1, 1, 1).toordinal()
            - datetime.date(int(x["year"]), 1, 1).toordinal()
        ),
        axis=1,
    )

    BflyMod["DATE"] = BflyMod.apply(
        lambda x: datetime.datetime(
            int(x["year"]),
            int(x["month"]),
            int(x["day"]),
            int(x["hour"]),
            int(x["minute"]),
            0,
        ),
        axis=1,
    )

    BflyMod["MEASURER"] = "Munoz-Jaramillo"
    BflyMod["OBSERVER"] = "SC_SP_RG_DB_KM"
    BflyMod["SURVEY"] = ""
    BflyMod.loc[BflyMod["survey"] == 1, "SURVEY"] = "Schwabe"
    BflyMod.loc[BflyMod["survey"] == 2, "SURVEY"] = "Spoerer"
    BflyMod.loc[BflyMod["survey"] == 1002, "SURVEY"] = "KMAS"
    BflyMod.loc[BflyMod["survey"] == 1003, "SURVEY"] = "Debrecen"
    BflyMod.loc[BflyMod["survey"] == 1004, "SURVEY"] = "RGO"

    # Remove missing days and spotless days
    BflyMod = BflyMod.loc[np.isfinite(BflyMod["correctedArea"]), :]

    Bfly = {}

    for observer in np.unique(BflyMod["OBSERVER"]):
        Bfly[observer] = {}
        Bfly[observer]["DF"] = BflyMod.loc[BflyMod["OBSERVER"] == observer, :]

        Bfly[observer]["Lat"] = "latitude"
        Bfly[observer]["Lon"] = "longitude"
        Bfly[observer]["Area"] = "correctedArea"
        Bfly[observer]["Year"] = "year"
        Bfly[observer]["Month"] = "month"
        Bfly[observer]["Day"] = "day"
        Bfly[observer]["Hour"] = "hour"
        Bfly[observer]["Minute"] = "minute"

    # --------------------------------------------------------------------------------------------
    # Scheiner
    LOG.info("Read Scheiner...")

    Bfly["Scheiner"] = {}
    Bfly["Scheiner"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1611_1631_scheiner_v1.1_20160707.txt",
        delimiter=r"\s+",
        engine="python",
        on_bad_lines=bad_line_printer,
    )

    Bfly["Scheiner"]["DF"]["ORDINAL"] = Bfly["Scheiner"]["DF"].apply(
        lambda x: datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal(),
        axis=1,
    )

    Bfly["Scheiner"]["DF"]["FRACYEAR"] = Bfly["Scheiner"]["DF"].apply(
        lambda x: x["YYYY"]
        + (
            datetime.date(x["YYYY"], x["MM"], x["DD"]).toordinal()
            - datetime.date(x["YYYY"], 1, 1).toordinal()
        )
        / (
            datetime.date(x["YYYY"] + 1, 1, 1).toordinal()
            - datetime.date(x["YYYY"], 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Scheiner"]["DF"].loc[np.isnan(Bfly["Scheiner"]["DF"].loc[:, "HH"]), "HH"] = 12
    Bfly["Scheiner"]["DF"].loc[np.isnan(Bfly["Scheiner"]["DF"].loc[:, "MI"]), "MI"] = 0

    Bfly["Scheiner"]["DF"]["DATE"] = Bfly["Scheiner"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YYYY"]), int(x["MM"]), int(x["DD"]), int(x["HH"]), int(x["MI"]), 0
        ),
        axis=1,
    )

    Bfly["Scheiner"]["DF"]["OBSERVER"] = "Scheiner"
    Bfly["Scheiner"]["DF"]["CYCLE"] = np.nan

    Bfly["Scheiner"]["Lat"] = "BBB.B"
    Bfly["Scheiner"]["Lon"] = "LLL.L"
    Bfly["Scheiner"]["Area"] = "UMB"
    Bfly["Scheiner"]["Year"] = "YYYY"
    Bfly["Scheiner"]["Month"] = "MM"
    Bfly["Scheiner"]["Day"] = "DD"
    Bfly["Scheiner"]["Hour"] = "HH"
    Bfly["Scheiner"]["Minute"] = "MI"

    # --------------------------------------------------------------------------------------------
    # Galilei
    LOG.info("Read Galilei...")

    Bfly["Galilei"] = {}
    Bfly["Galilei"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1612_Galilei_position_vokhmyanin_2021_m2.txt",
        delimiter=r"\t+",
        skiprows=32,
        engine="python",
        on_bad_lines=bad_line_printer,
    )

    Bfly["Galilei"]["DF"]["ORDINAL"] = Bfly["Galilei"]["DF"].apply(
        lambda x: datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal(),
        axis=1,
    )

    Bfly["Galilei"]["DF"]["FRACYEAR"] = Bfly["Galilei"]["DF"].apply(
        lambda x: x["YYYY"]
        + (
            datetime.date(x["YYYY"], x["MM"], x["DD"]).toordinal()
            - datetime.date(x["YYYY"], 1, 1).toordinal()
        )
        / (
            datetime.date(x["YYYY"] + 1, 1, 1).toordinal()
            - datetime.date(x["YYYY"], 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Galilei"]["DF"].loc[np.isnan(Bfly["Galilei"]["DF"].loc[:, "HH"]), "HH"] = 12
    Bfly["Galilei"]["DF"].loc[np.isnan(Bfly["Galilei"]["DF"].loc[:, "MI"]), "MI"] = 0

    Bfly["Galilei"]["DF"]["DATE"] = Bfly["Galilei"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YYYY"]), int(x["MM"]), int(x["DD"]), int(x["HH"]), int(x["MI"]), 0
        ),
        axis=1,
    )

    Bfly["Galilei"]["DF"]["MEASURER"] = "Vokhmyanin"
    Bfly["Galilei"]["DF"]["OBSERVER"] = "Galilei"
    Bfly["Galilei"]["DF"]["CYCLE"] = np.nan

    Bfly["Galilei"]["Lat"] = "BBB.BB"
    Bfly["Galilei"]["Lon"] = "LLL.LL"
    Bfly["Galilei"]["Area"] = "Area_w"
    Bfly["Galilei"]["Year"] = "YYYY"
    Bfly["Galilei"]["Month"] = "MM"
    Bfly["Galilei"]["Day"] = "DD"
    Bfly["Galilei"]["Hour"] = "HH"
    Bfly["Galilei"]["Minute"] = "MI"

    # --------------------------------------------------------------------------------------------
    # Cigioli
    LOG.info("Read Cigioli...")

    Bfly["Cigioli"] = {}
    Bfly["Cigioli"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1612_cigioli_position_vokhmyanin_2021_m2.txt",
        delimiter=r"\t+",
        skiprows=31,
        engine="python",
        on_bad_lines=bad_line_printer,
    )

    Bfly["Cigioli"]["DF"]["ORDINAL"] = Bfly["Cigioli"]["DF"].apply(
        lambda x: datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal(),
        axis=1,
    )

    Bfly["Cigioli"]["DF"]["FRACYEAR"] = Bfly["Cigioli"]["DF"].apply(
        lambda x: x["YYYY"]
        + (
            datetime.date(x["YYYY"], x["MM"], x["DD"]).toordinal()
            - datetime.date(x["YYYY"], 1, 1).toordinal()
        )
        / (
            datetime.date(x["YYYY"] + 1, 1, 1).toordinal()
            - datetime.date(x["YYYY"], 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Cigioli"]["DF"].loc[np.isnan(Bfly["Cigioli"]["DF"].loc[:, "HH"]), "HH"] = 12
    Bfly["Cigioli"]["DF"].loc[np.isnan(Bfly["Cigioli"]["DF"].loc[:, "MI"]), "MI"] = 0

    Bfly["Cigioli"]["DF"]["DATE"] = Bfly["Cigioli"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YYYY"]), int(x["MM"]), int(x["DD"]), int(x["HH"]), int(x["MI"]), 0
        ),
        axis=1,
    )

    Bfly["Cigioli"]["DF"]["MEASURER"] = "Vokhmyanin"
    Bfly["Cigioli"]["DF"]["OBSERVER"] = "Cigioli"
    Bfly["Cigioli"]["DF"]["CYCLE"] = np.nan

    Bfly["Cigioli"]["Lat"] = "BBB.BB"
    Bfly["Cigioli"]["Lon"] = "LLL.LL"
    Bfly["Cigioli"]["Area"] = "Area_w"
    Bfly["Cigioli"]["Year"] = "YYYY"
    Bfly["Cigioli"]["Month"] = "MM"
    Bfly["Cigioli"]["Day"] = "DD"
    Bfly["Cigioli"]["Hour"] = "HH"
    Bfly["Cigioli"]["Minute"] = "MI"

    # --------------------------------------------------------------------------------------------
    # Cologna
    LOG.info("Read Cologna...")

    Bfly["Cologna"] = {}
    Bfly["Cologna"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1612_Cologna_position_vokhmyanin_2021_m2.txt",
        delimiter=r"\t+",
        skiprows=29,
        engine="python",
        on_bad_lines=bad_line_printer,
    )

    Bfly["Cologna"]["DF"]["ORDINAL"] = Bfly["Cologna"]["DF"].apply(
        lambda x: datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal(),
        axis=1,
    )

    Bfly["Cologna"]["DF"]["FRACYEAR"] = Bfly["Cologna"]["DF"].apply(
        lambda x: x["YYYY"]
        + (
            datetime.date(x["YYYY"], x["MM"], x["DD"]).toordinal()
            - datetime.date(x["YYYY"], 1, 1).toordinal()
        )
        / (
            datetime.date(x["YYYY"] + 1, 1, 1).toordinal()
            - datetime.date(x["YYYY"], 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Cologna"]["DF"].loc[np.isnan(Bfly["Cologna"]["DF"].loc[:, "HH"]), "HH"] = 12
    Bfly["Cologna"]["DF"].loc[np.isnan(Bfly["Cologna"]["DF"].loc[:, "MI"]), "MI"] = 0

    Bfly["Cologna"]["DF"]["DATE"] = Bfly["Cologna"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YYYY"]), int(x["MM"]), int(x["DD"]), int(x["HH"]), int(x["MI"]), 0
        ),
        axis=1,
    )

    Bfly["Cologna"]["DF"]["MEASURER"] = "Vokhmyanin"
    Bfly["Cologna"]["DF"]["OBSERVER"] = "Cologna"
    Bfly["Cologna"]["DF"]["CYCLE"] = np.nan

    Bfly["Cologna"]["Lat"] = "BBB.BB"
    Bfly["Cologna"]["Lon"] = "LLL.LL"
    Bfly["Cologna"]["Area"] = "Area"
    Bfly["Cologna"]["Year"] = "YYYY"
    Bfly["Cologna"]["Month"] = "MM"
    Bfly["Cologna"]["Day"] = "DD"
    Bfly["Cologna"]["Hour"] = "HH"
    Bfly["Cologna"]["Minute"] = "MI"

    # --------------------------------------------------------------------------------------------
    # Colonna
    LOG.info("Read Colonna...")

    Bfly["Colonna"] = {}
    Bfly["Colonna"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1612_Colonna_position_vokhmyanin_2021_m2.txt",
        delimiter=r"\t+",
        skiprows=27,
        engine="python",
        on_bad_lines=bad_line_printer,
    )

    Bfly["Colonna"]["DF"]["ORDINAL"] = Bfly["Colonna"]["DF"].apply(
        lambda x: datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal(),
        axis=1,
    )

    Bfly["Colonna"]["DF"]["FRACYEAR"] = Bfly["Colonna"]["DF"].apply(
        lambda x: x["YYYY"]
        + (
            datetime.date(x["YYYY"], x["MM"], x["DD"]).toordinal()
            - datetime.date(x["YYYY"], 1, 1).toordinal()
        )
        / (
            datetime.date(x["YYYY"] + 1, 1, 1).toordinal()
            - datetime.date(x["YYYY"], 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Colonna"]["DF"].loc[np.isnan(Bfly["Colonna"]["DF"].loc[:, "HH"]), "HH"] = 12
    Bfly["Colonna"]["DF"].loc[np.isnan(Bfly["Colonna"]["DF"].loc[:, "MI"]), "MI"] = 0

    Bfly["Colonna"]["DF"]["DATE"] = Bfly["Colonna"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YYYY"]), int(x["MM"]), int(x["DD"]), int(x["HH"]), int(x["MI"]), 0
        ),
        axis=1,
    )

    Bfly["Colonna"]["DF"]["MEASURER"] = "Vokhmyanin"
    Bfly["Colonna"]["DF"]["OBSERVER"] = "Colonna"
    Bfly["Colonna"]["DF"]["CYCLE"] = np.nan

    Bfly["Colonna"]["Lat"] = "BBB.BB"
    Bfly["Colonna"]["Lon"] = "LLL.LL"
    Bfly["Colonna"]["Area"] = "Area"
    Bfly["Colonna"]["Year"] = "YYYY"
    Bfly["Colonna"]["Month"] = "MM"
    Bfly["Colonna"]["Day"] = "DD"
    Bfly["Colonna"]["Hour"] = "HH"
    Bfly["Colonna"]["Minute"] = "MI"

    # --------------------------------------------------------------------------------------------
    # Harriot
    LOG.info("Read Harriot...")

    Bfly["Harriot"] = {}
    Bfly["Harriot"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1610_1613_Harriot_11207_2020_1604_MOESM3_ESM.txt",
        delimiter=r"\t+",
        skiprows=30,
        engine="python",
        encoding="unicode_escape",
        on_bad_lines=bad_line_printer,
    )

    Bfly["Harriot"]["DF"]["ORDINAL"] = Bfly["Harriot"]["DF"].apply(
        lambda x: datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal(),
        axis=1,
    )

    Bfly["Harriot"]["DF"]["FRACYEAR"] = Bfly["Harriot"]["DF"].apply(
        lambda x: x["YYYY"]
        + (
            datetime.date(x["YYYY"], x["MM"], x["DD"]).toordinal()
            - datetime.date(x["YYYY"], 1, 1).toordinal()
        )
        / (
            datetime.date(x["YYYY"] + 1, 1, 1).toordinal()
            - datetime.date(x["YYYY"], 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Harriot"]["DF"].loc[np.isnan(Bfly["Harriot"]["DF"].loc[:, "HH"]), "HH"] = 12
    Bfly["Harriot"]["DF"].loc[np.isnan(Bfly["Harriot"]["DF"].loc[:, "MI"]), "MI"] = 0

    Bfly["Harriot"]["DF"]["DATE"] = Bfly["Harriot"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YYYY"]), int(x["MM"]), int(x["DD"]), int(x["HH"]), int(x["MI"]), 0
        ),
        axis=1,
    )

    Bfly["Harriot"]["DF"]["MEASURER"] = "Vokhmyanin"
    Bfly["Harriot"]["DF"]["OBSERVER"] = "Harriot"
    Bfly["Harriot"]["DF"]["CYCLE"] = np.nan

    Bfly["Harriot"]["Lat"] = "BBB.BB"
    Bfly["Harriot"]["Lon"] = "LLL.LL"
    Bfly["Harriot"]["Area"] = "Area"
    Bfly["Harriot"]["Year"] = "YYYY"
    Bfly["Harriot"]["Month"] = "MM"
    Bfly["Harriot"]["Day"] = "DD"
    Bfly["Harriot"]["Hour"] = "HH"
    Bfly["Harriot"]["Minute"] = "MI"

    # --------------------------------------------------------------------------------------------
    # Smogulecki and Tarde
    LOG.info("Read Smogulecki and Tarde...")

    Bfly["Smogulecki"] = {}
    Bfly["Smogulecki"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1615_1625_Tarde-Smogulecki_group_positions.txt",
        delimiter=r"\t+",
        skiprows=10,
        engine="python",
        on_bad_lines=bad_line_printer,
    )

    Bfly["Smogulecki"]["DF"]["ORDINAL"] = (
        Bfly["Smogulecki"]["DF"]
        .apply(
            lambda x: datetime.date(int(x["DATE"]), 1, 1).toordinal()
            + np.round((x["DATE"] - int(x["DATE"])) * 365),
            axis=1,
        )
        .astype(int)
    )

    Bfly["Smogulecki"]["DF"]["YEAR"] = Bfly["Smogulecki"]["DF"]["DATE"].astype(int)
    Bfly["Smogulecki"]["DF"]["MONTH"] = Bfly["Smogulecki"]["DF"].apply(
        lambda x: datetime.date.fromordinal(int(x["ORDINAL"])).month,
        axis=1,
    )
    Bfly["Smogulecki"]["DF"]["DAY"] = Bfly["Smogulecki"]["DF"].apply(
        lambda x: datetime.date.fromordinal(int(x["ORDINAL"])).day,
        axis=1,
    )

    Bfly["Smogulecki"]["DF"]["HOUR"] = 12
    Bfly["Smogulecki"]["DF"]["MINUTE"] = 0

    Bfly["Smogulecki"]["DF"]["FRACYEAR"] = Bfly["Smogulecki"]["DF"]["DATE"]

    Bfly["Smogulecki"]["DF"]["DATE"] = Bfly["Smogulecki"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YEAR"]),
            int(x["MONTH"]),
            int(x["DAY"]),
            int(x["HOUR"]),
            int(x["MINUTE"]),
            0,
        ),
        axis=1,
    )

    Bfly["Smogulecki"]["DF"]["AREA"] = np.nan
    Bfly["Smogulecki"]["DF"]["LONGITUDE"] = np.nan

    Bfly["Smogulecki"]["DF"]["MEASURER"] = "Carrasco"
    Bfly["Smogulecki"]["DF"]["OBSERVER"] = Bfly["Smogulecki"]["DF"][
        "OBSERVER"
    ].str.capitalize()
    Bfly["Smogulecki"]["DF"]["CYCLE"] = np.nan

    Bfly["Smogulecki"]["Lat"] = "LATITUDE"
    Bfly["Smogulecki"]["Lon"] = "LONGITUDE"
    Bfly["Smogulecki"]["Area"] = "AREA"
    Bfly["Smogulecki"]["Year"] = "YEAR"
    Bfly["Smogulecki"]["Month"] = "MONTH"
    Bfly["Smogulecki"]["Day"] = "DAY"
    Bfly["Smogulecki"]["Hour"] = "HOUR"
    Bfly["Smogulecki"]["Minute"] = "MINUTE"

    Bfly["Tarde"] = Bfly["Smogulecki"].copy()
    Bfly["Tarde"]["DF"] = Bfly["Tarde"]["DF"].loc[
        Bfly["Tarde"]["DF"]["OBSERVER"] == "Tarde"
    ]

    Bfly["Smogulecki"]["DF"] = Bfly["Smogulecki"]["DF"].loc[
        Bfly["Smogulecki"]["DF"]["OBSERVER"] == "Smogulecki"
    ]

    # --------------------------------------------------------------------------------------------
    # Marcgraf and Tarde
    LOG.info("Read Marcgraf and Tarde...")

    Bfly["Marcgraf"] = {}
    Bfly["Marcgraf"]["DF"] = pd.read_csv(
        DATA_FOLDER + "2019_data/Bfly_Gassendi_Malapert_Macgraf.csv"
    )

    Bfly["Marcgraf"]["DF"]["ORDINAL"] = (
        Bfly["Marcgraf"]["DF"]
        .apply(
            lambda x: datetime.date(int(x["Year"]), 1, 1).toordinal()
            + np.round((x["Year"] - int(x["Year"])) * 365),
            axis=1,
        )
        .astype(int)
    )

    Bfly["Marcgraf"]["DF"]["Year"] = Bfly["Marcgraf"]["DF"]["Year"].astype(int)
    Bfly["Marcgraf"]["DF"]["MONTH"] = Bfly["Marcgraf"]["DF"].apply(
        lambda x: datetime.date.fromordinal(int(x["ORDINAL"])).month,
        axis=1,
    )
    Bfly["Marcgraf"]["DF"]["DAY"] = Bfly["Marcgraf"]["DF"].apply(
        lambda x: datetime.date.fromordinal(int(x["ORDINAL"])).day,
        axis=1,
    )

    Bfly["Marcgraf"]["DF"]["HOUR"] = 12
    Bfly["Marcgraf"]["DF"]["MINUTE"] = 0

    Bfly["Marcgraf"]["DF"]["FRACYEAR"] = Bfly["Marcgraf"]["DF"]["Year"]

    Bfly["Marcgraf"]["DF"]["DATE"] = Bfly["Marcgraf"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["Year"]),
            int(x["MONTH"]),
            int(x["DAY"]),
            int(x["HOUR"]),
            int(x["MINUTE"]),
            0,
        ),
        axis=1,
    )

    Bfly["Marcgraf"]["DF"]["AREA"] = np.nan
    Bfly["Marcgraf"]["DF"]["LONGITUDE"] = np.nan

    Bfly["Marcgraf"]["DF"]["MEASURER"] = "Vaquero"
    Bfly["Marcgraf"]["DF"]["CYCLE"] = np.nan

    Bfly["Marcgraf"]["Lat"] = "Lat"
    Bfly["Marcgraf"]["Lon"] = "LONGITUDE"
    Bfly["Marcgraf"]["Area"] = "AREA"
    Bfly["Marcgraf"]["Year"] = "Year"
    Bfly["Marcgraf"]["Month"] = "MONTH"
    Bfly["Marcgraf"]["Day"] = "DAY"
    Bfly["Marcgraf"]["Hour"] = "HOUR"
    Bfly["Marcgraf"]["Minute"] = "MINUTE"

    # --------------------------------------------------------------------------------------------
    # Mogling
    LOG.info("Read Mogling...")

    Bfly["Mogling"] = {}
    Bfly["Mogling"]["DF"] = pd.read_excel(
        DATA_FOLDER + "input_data/1626_1629_mogling_sunspot_position.xlsx"
    )

    Bfly["Mogling"]["DF"]["ORDINAL"] = Bfly["Mogling"]["DF"].apply(
        lambda x: datetime.date(
            int(x["Year"]), int(x["Month"]), int(x["Day"])
        ).toordinal(),
        axis=1,
    )

    Bfly["Mogling"]["DF"]["FRACYEAR"] = Bfly["Mogling"]["DF"].apply(
        lambda x: x["Year"]
        + (
            datetime.date(int(x["Year"]), int(x["Month"]), int(x["Day"])).toordinal()
            - datetime.date(int(x["Year"]), 1, 1).toordinal()
        )
        / (
            datetime.date(int(x["Year"]) + 1, 1, 1).toordinal()
            - datetime.date(int(x["Year"]), 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Mogling"]["DF"]["HOUR"] = 12
    Bfly["Mogling"]["DF"]["MINUTE"] = 0

    Bfly["Mogling"]["DF"]["DATE"] = Bfly["Mogling"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["Year"]),
            int(x["Month"]),
            int(x["Day"]),
            int(x["HOUR"]),
            int(x["MINUTE"]),
            0,
        ),
        axis=1,
    )

    Bfly["Mogling"]["DF"]["MEASURER"] = "Hayakawa"
    Bfly["Mogling"]["DF"]["OBSERVER"] = "Mogling"
    Bfly["Mogling"]["DF"]["CYCLE"] = np.nan

    Bfly["Mogling"]["DF"]["AREA"] = np.nan

    Bfly["Mogling"]["Lat"] = "Lat"
    Bfly["Mogling"]["Lon"] = "Lon"
    Bfly["Mogling"]["Area"] = "AREA"
    Bfly["Mogling"]["Year"] = "Year"
    Bfly["Mogling"]["Month"] = "Month"
    Bfly["Mogling"]["Day"] = "Day"
    Bfly["Mogling"]["Hour"] = "HOUR"
    Bfly["Mogling"]["Minute"] = "MINUTE"

    # --------------------------------------------------------------------------------------------
    # Malapert
    LOG.info("Read Malapert and Gassendi...")

    Bfly["Malapert"] = {}
    Bfly["Malapert"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1618_1626_Malapert_heliographic_coordinates.txt",
        delimiter=r"\t+",
        skiprows=11,
        engine="python",
        on_bad_lines=bad_line_printer,
    )

    Bfly["Malapert"]["DF"]["ORDINAL"] = Bfly["Malapert"]["DF"].apply(
        lambda x: datetime.date(
            int(x["YEAR"]), int(x["MONTH"]), int(x["DAY"])
        ).toordinal(),
        axis=1,
    )

    Bfly["Malapert"]["DF"]["FRACYEAR"] = Bfly["Malapert"]["DF"].apply(
        lambda x: x["YEAR"]
        + (
            datetime.date(int(x["YEAR"]), int(x["MONTH"]), int(x["DAY"])).toordinal()
            - datetime.date(int(x["YEAR"]), 1, 1).toordinal()
        )
        / (
            datetime.date(int(x["YEAR"]) + 1, 1, 1).toordinal()
            - datetime.date(int(x["YEAR"]), 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Malapert"]["DF"]["AREA"] = np.nan
    Bfly["Malapert"]["DF"]["LONGITUDE"] = np.nan

    Bfly["Malapert"]["DF"]["HOUR"] = 12
    Bfly["Malapert"]["DF"]["MINUTE"] = 0

    Bfly["Malapert"]["DF"]["DATE"] = Bfly["Malapert"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YEAR"]),
            int(x["MONTH"]),
            int(x["DAY"]),
            int(x["HOUR"]),
            int(x["MINUTE"]),
            0,
        ),
        axis=1,
    )

    Bfly["Malapert"]["DF"]["MEASURER"] = "Vaquero"
    Bfly["Malapert"]["DF"]["OBSERVER"] = "Malapert"
    Bfly["Malapert"]["DF"]["CYCLE"] = np.nan

    Bfly["Malapert"]["Lat"] = "LATITUDE"
    Bfly["Malapert"]["Lon"] = "LONGITUDE"
    Bfly["Malapert"]["Area"] = "AREA"
    Bfly["Malapert"]["Year"] = "YEAR"
    Bfly["Malapert"]["Month"] = "MONTH"
    Bfly["Malapert"]["Day"] = "DAY"
    Bfly["Malapert"]["Hour"] = "HOUR"
    Bfly["Malapert"]["Minute"] = "MINUTE"

    Bfly["Gassendi"] = Bfly["Marcgraf"].copy()
    Bfly["Gassendi"]["DF"] = Bfly["Gassendi"]["DF"].loc[
        Bfly["Gassendi"]["DF"]["OBSERVER"] == "Gassendi", :
    ]

    Bfly["Marcgraf"]["DF"] = Bfly["Marcgraf"]["DF"].loc[
        Bfly["Marcgraf"]["DF"]["OBSERVER"] == "Marcgraf"
    ]

    # # --------------------------------------------------------------------------------------------
    # # Gassendi
    # LOG.info("Read Gassendi...")

    # Bfly["Gassendi"] = {}
    # Bfly["Gassendi"]["DF"] = pd.read_table(
    #     data_folder + "input_data/1633_1638_Gassendi_position_zolotova.txt",
    #     delimiter=r"\t+",
    #     skiprows=33,
    #     engine="python",
    #     on_bad_lines=bad_line_printer,
    # )

    # Bfly["Gassendi"]["DF"]["ORDINAL"] = Bfly["Gassendi"]["DF"].apply(
    #     lambda x: datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal(),
    #     axis=1,
    # )

    # Bfly["Gassendi"]["DF"]["FRACYEAR"] = Bfly["Gassendi"]["DF"].apply(
    #     lambda x: x["YYYY"]
    #     + (
    #         datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal()
    #         - datetime.date(int(x["YYYY"]), 1, 1).toordinal()
    #     )
    #     / (
    #         datetime.date(int(x["YYYY"]) + 1, 1, 1).toordinal()
    #         - datetime.date(int(x["YYYY"]), 1, 1).toordinal()
    #     ),
    #     axis=1,
    # )

    # Bfly["Gassendi"]["DF"].loc[np.isnan(Bfly["Gassendi"]["DF"].loc[:,"HH"]),"HH"] = 12
    # Bfly["Gassendi"]["DF"].loc[np.isnan(Bfly["Gassendi"]["DF"].loc[:,"MI"]),"MI"] = 0

    # Bfly["Gassendi"]["DF"]["DATE"] = Bfly["Gassendi"]["DF"].apply(
    #     lambda x: datetime.datetime(int(x["YYYY"]), int(x["MM"]), int(x["DD"]), int(x["HH"]), int(x["MI"]),0),
    #     axis=1,
    # )

    # Bfly["Gassendi"]["DF"]["MEASURER"] = "Vokhmyanin"
    # Bfly["Gassendi"]["DF"]["OBSERVER"] = "Gassendi"
    # Bfly["Gassendi"]["DF"]["CYCLE"] = np.nan

    # Bfly["Gassendi"]["Lat"] = "BBB.BB"
    # Bfly["Gassendi"]["Lon"] = "LLL.LL"
    # Bfly["Gassendi"]["Area"] = "Area"
    # Bfly["Gassendi"]["Year"] = "YYYY"
    # Bfly["Gassendi"]["Month"] = "MM"
    # Bfly["Gassendi"]["Day"] = "DD"
    # Bfly["Gassendi"]["Hour"] = "HH"
    # Bfly["Gassendi"]["Minute"] = "MI"

    # --------------------------------------------------------------------------------------------
    # Hevelius
    LOG.info("Read Hevelius...")

    Bfly["Hevelius"] = {}
    Bfly["Hevelius"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1642_1644_Hevelius_positions_v1.0_20180412.txt",
        delimiter=r"\s+",
        skiprows=23,
        engine="python",
        on_bad_lines=bad_line_printer,
    )

    Bfly["Hevelius"]["DF"]["ORDINAL"] = Bfly["Hevelius"]["DF"].apply(
        lambda x: datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal(),
        axis=1,
    )

    Bfly["Hevelius"]["DF"]["FRACYEAR"] = Bfly["Hevelius"]["DF"].apply(
        lambda x: x["YYYY"]
        + (
            datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal()
            - datetime.date(int(x["YYYY"]), 1, 1).toordinal()
        )
        / (
            datetime.date(int(x["YYYY"]) + 1, 1, 1).toordinal()
            - datetime.date(int(x["YYYY"]), 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Hevelius"]["DF"].loc[np.isnan(Bfly["Hevelius"]["DF"].loc[:, "HH"]), "HH"] = 12
    Bfly["Hevelius"]["DF"].loc[np.isnan(Bfly["Hevelius"]["DF"].loc[:, "MI"]), "MI"] = 0

    Bfly["Hevelius"]["DF"]["DATE"] = Bfly["Hevelius"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YYYY"]), int(x["MM"]), int(x["DD"]), int(x["HH"]), int(x["MI"]), 0
        ),
        axis=1,
    )

    Bfly["Hevelius"]["DF"]["OBSERVER"] = "Hevelius"
    Bfly["Hevelius"]["DF"]["MEASURER"] = "Carrasco"
    Bfly["Hevelius"]["DF"]["CYCLE"] = np.nan

    Bfly["Hevelius"]["Lat"] = "BBB.B"
    Bfly["Hevelius"]["Lon"] = "LLL.L"
    Bfly["Hevelius"]["Area"] = "UMSH"
    Bfly["Hevelius"]["Year"] = "YYYY"
    Bfly["Hevelius"]["Month"] = "MM"
    Bfly["Hevelius"]["Day"] = "DD"
    Bfly["Hevelius"]["Hour"] = "HH"
    Bfly["Hevelius"]["Minute"] = "MI"

    # --------------------------------------------------------------------------------------------
    # Muller
    LOG.info("Read Muller...")

    Bfly["Muller"] = {}
    Bfly["Muller"]["DF"] = pd.read_excel(
        DATA_FOLDER + "input_data/1719_1720_JCM_data_210711.xlsx"
    )

    Bfly["Muller"]["DF"]["ORDINAL"] = Bfly["Muller"]["DF"].apply(
        lambda x: datetime.date(
            int(x["Year"]), int(x["Month"]), int(x["Day"])
        ).toordinal(),
        axis=1,
    )

    Bfly["Muller"]["DF"]["FRACYEAR"] = Bfly["Muller"]["DF"].apply(
        lambda x: x["Year"]
        + (
            datetime.date(int(x["Year"]), int(x["Month"]), int(x["Day"])).toordinal()
            - datetime.date(int(x["Year"]), 1, 1).toordinal()
        )
        / (
            datetime.date(int(x["Year"]) + 1, 1, 1).toordinal()
            - datetime.date(int(x["Year"]), 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Muller"]["DF"].loc[np.isnan(Bfly["Muller"]["DF"].loc[:, "Hour"]), "Hour"] = 12
    Bfly["Muller"]["DF"].loc[np.isnan(Bfly["Muller"]["DF"].loc[:, "Min"]), "Min"] = 0

    Bfly["Muller"]["DF"]["DATE"] = Bfly["Muller"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["Year"]),
            int(x["Month"]),
            int(x["Day"]),
            int(x["Hour"]),
            int(x["Min"]),
            0,
        ),
        axis=1,
    )

    Bfly["Muller"]["DF"]["Area"] = np.nan

    Bfly["Muller"]["DF"]["MEASURER"] = "Hayakawa"
    Bfly["Muller"]["DF"]["OBSERVER"] = "Muller"
    Bfly["Muller"]["DF"]["CYCLE"] = np.nan

    Bfly["Muller"]["Lat"] = "Lat. Ave"
    Bfly["Muller"]["Lon"] = "Lon. Ave (*)"
    Bfly["Muller"]["Area"] = "Area"
    Bfly["Muller"]["Year"] = "Year"
    Bfly["Muller"]["Month"] = "Month"
    Bfly["Muller"]["Day"] = "Day"
    Bfly["Muller"]["Hour"] = "Hour"
    Bfly["Muller"]["Minute"] = "Min"

    # --------------------------------------------------------------------------------------------
    # Ribes
    LOG.info("Read Ribes...")

    Bfly["Ribes"] = {}
    Bfly["Ribes"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1671_1719_Sunspot_Latitudes_Ribes.txt",
        delimiter=r"\t+",
        skiprows=13,
        engine="python",
        on_bad_lines=bad_line_printer,
        encoding="unicode_escape",
    )

    Bfly["Ribes"]["DF"]["ORDINAL"] = Bfly["Ribes"]["DF"].apply(
        lambda x: datetime.date(int(x["Date"]), 1, 1).toordinal()
        + np.round((x["Date"] - int(x["Date"])) * 365),
        axis=1,
    )

    Bfly["Ribes"]["DF"]["YEAR"] = Bfly["Ribes"]["DF"]["Date"].astype(int)
    Bfly["Ribes"]["DF"]["MONTH"] = Bfly["Ribes"]["DF"].apply(
        lambda x: datetime.date.fromordinal(int(x["ORDINAL"])).month,
        axis=1,
    )
    Bfly["Ribes"]["DF"]["DAY"] = Bfly["Ribes"]["DF"].apply(
        lambda x: datetime.date.fromordinal(int(x["ORDINAL"])).day,
        axis=1,
    )

    Bfly["Ribes"]["DF"]["HOUR"] = 12
    Bfly["Ribes"]["DF"]["MINUTE"] = 0

    Bfly["Ribes"]["DF"]["FRACYEAR"] = Bfly["Ribes"]["DF"]["Date"]

    Bfly["Ribes"]["DF"]["DATE"] = Bfly["Ribes"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YEAR"]),
            int(x["MONTH"]),
            int(x["DAY"]),
            int(x["HOUR"]),
            int(x["MINUTE"]),
            0,
        ),
        axis=1,
    )

    Bfly["Ribes"]["DF"]["area"] = np.nan
    Bfly["Ribes"]["DF"]["longitude"] = np.nan
    Bfly["Ribes"]["DF"]["CYCLE"] = np.nan

    Bfly["Ribes"]["Lat"] = "latitude"
    Bfly["Ribes"]["Lon"] = "longitude"
    Bfly["Ribes"]["Area"] = "area"
    Bfly["Ribes"]["Year"] = "YEAR"
    Bfly["Ribes"]["Month"] = "MONTH"
    Bfly["Ribes"]["Day"] = "DAY"
    Bfly["Ribes"]["Hour"] = "HOUR"
    Bfly["Ribes"]["Minute"] = "MINUTE"

    Bfly["Ribes"]["DF"]["OBSERVER"] = Bfly["Ribes"]["DF"]["Source"]
    Bfly["Ribes"]["DF"]["MEASURER"] = "Vaquero"

    # --------------------------------------------------------------------------------------------
    # Eimmart
    LOG.info("Read Eimmart...")

    Bfly["Eimmart"] = {}
    Bfly["Eimmart"]["DF"] = pd.read_excel(
        DATA_FOLDER + "input_data/1684_1718_eimmart_sunspot_data.xlsx"
    )

    Bfly["Eimmart"]["DF"]["ORDINAL"] = Bfly["Eimmart"]["DF"].apply(
        lambda x: datetime.date(
            int(x["YEAR"]), int(x["MONTH"]), int(x["DAY"])
        ).toordinal(),
        axis=1,
    )

    Bfly["Eimmart"]["DF"]["FRACYEAR"] = Bfly["Eimmart"]["DF"].apply(
        lambda x: x["YEAR"]
        + (
            datetime.date(int(x["YEAR"]), int(x["MONTH"]), int(x["DAY"])).toordinal()
            - datetime.date(int(x["YEAR"]), 1, 1).toordinal()
        )
        / (
            datetime.date(int(x["YEAR"]) + 1, 1, 1).toordinal()
            - datetime.date(int(x["YEAR"]), 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Eimmart"]["DF"]["HOUR"] = 12
    Bfly["Eimmart"]["DF"]["MINUTE"] = 0
    Bfly["Eimmart"]["DF"]["OBSERVER"] = Bfly["Eimmart"]["DF"][
        "OBSERVER"
    ].str.capitalize()

    Bfly["Eimmart"]["DF"]["DATE"] = Bfly["Eimmart"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YEAR"]),
            int(x["MONTH"]),
            int(x["DAY"]),
            int(x["HOUR"]),
            int(x["MINUTE"]),
            0,
        ),
        axis=1,
    )

    Bfly["Eimmart"]["DF"]["MEASURER"] = "Hayakawa"
    Bfly["Eimmart"]["DF"]["CYCLE"] = np.nan

    Bfly["Eimmart"]["Lat"] = "Heliographic Latitude"
    Bfly["Eimmart"]["Lon"] = "L"
    Bfly["Eimmart"]["Area"] = "AREAMSH"
    Bfly["Eimmart"]["Year"] = "YEAR"
    Bfly["Eimmart"]["Month"] = "MONTH"
    Bfly["Eimmart"]["Day"] = "DAY"
    Bfly["Eimmart"]["Hour"] = "HOUR"
    Bfly["Eimmart"]["Minute"] = "MINUTE"

    for observer in np.unique(Bfly["Eimmart"]["DF"]["OBSERVER"]):
        if observer != "Eimmart":
            Bfly[observer] = Bfly["Eimmart"].copy()
            Bfly[observer]["DF"] = Bfly[observer]["DF"].loc[
                Bfly[observer]["DF"]["OBSERVER"] == observer, :
            ]
    Bfly["Eimmart"]["DF"] = Bfly["Eimmart"]["DF"].loc[
        Bfly["Eimmart"]["DF"]["OBSERVER"] == "Eimmart", :
    ]

    # Define Spoerer1889 later to have it appear after Eimmart in the priority list
    Bfly["Spoerer1889"] = Bfly["Ribes"].copy()
    Bfly["Spoerer1889"]["DF"] = Bfly["Spoerer1889"]["DF"].loc[
        Bfly["Spoerer1889"]["DF"]["Source"] == "Spoerer1889", :
    ]

    Bfly["Ribes"]["DF"] = Bfly["Ribes"]["DF"].loc[
        Bfly["Ribes"]["DF"]["Source"] == "Ribes", :
    ]

    # --------------------------------------------------------------------------------------------
    # Siverus
    LOG.info("Read Siverus...")

    Bfly["Siverus"] = {}
    Bfly["Siverus"]["DF"] = pd.read_excel(
        DATA_FOLDER + "input_data/1671_Siverus_Cassini.xlsx"
    )

    Bfly["Siverus"]["DF"]["ORDINAL"] = Bfly["Siverus"]["DF"].apply(
        lambda x: datetime.date(
            int(x["YearG"]), int(x["MonthG"]), int(x["DayG"])
        ).toordinal(),
        axis=1,
    )

    Bfly["Siverus"]["DF"]["FRACYEAR"] = Bfly["Siverus"]["DF"].apply(
        lambda x: x["YearG"]
        + (
            datetime.date(int(x["YearG"]), int(x["MonthG"]), int(x["DayG"])).toordinal()
            - datetime.date(int(x["YearG"]), 1, 1).toordinal()
        )
        / (
            datetime.date(int(x["YearG"]) + 1, 1, 1).toordinal()
            - datetime.date(int(x["YearG"]), 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Siverus"]["DF"]["HourG"] = 12
    Bfly["Siverus"]["DF"]["MinuteG"] = 0

    Bfly["Siverus"]["DF"]["DATE"] = Bfly["Siverus"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YearG"]),
            int(x["MonthG"]),
            int(x["DayG"]),
            int(x["HourG"]),
            int(x["MinuteG"]),
            0,
        ),
        axis=1,
    )

    Bfly["Siverus"]["DF"]["area"] = np.nan

    Bfly["Siverus"]["DF"]["MEASURER"] = "Hayakawa"
    Bfly["Siverus"]["DF"]["CYCLE"] = np.nan

    Bfly["Siverus"]["Lat"] = "LatAverage"
    Bfly["Siverus"]["Lon"] = "LonAverage"
    Bfly["Siverus"]["Area"] = "area"
    Bfly["Siverus"]["Year"] = "YearG"
    Bfly["Siverus"]["Month"] = "MonthG"
    Bfly["Siverus"]["Day"] = "DayG"
    Bfly["Siverus"]["Hour"] = "HourG"
    Bfly["Siverus"]["Minute"] = "MinuteG"

    Bfly["Cassini"] = Bfly["Siverus"].copy()
    Bfly["Cassini"]["DF"] = Bfly["Cassini"]["DF"].loc[
        Bfly["Cassini"]["DF"]["OBSERVER"] == "Cassini", :
    ]
    Bfly["Siverus"]["DF"] = Bfly["Siverus"]["DF"].loc[
        Bfly["Siverus"]["DF"]["OBSERVER"] == "Siverus", :
    ]

    # --------------------------------------------------------------------------------------------
    # Hayakawa22
    LOG.info("Read Hayakawa22...")

    Bfly["Hayakawa22"] = {}
    Bfly["Hayakawa22"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1727_1748_sunspot_positions_1727_1748.txt",
        delimiter=r"\t",
        skiprows=29,
        engine="python",
        on_bad_lines=bad_line_printer,
        encoding="unicode_escape",
    )

    Bfly["Hayakawa22"]["DF"]["ORDINAL"] = Bfly["Hayakawa22"]["DF"].apply(
        lambda x: datetime.date(
            int(x["Year"]), int(x["Month"]), int(x["Day"])
        ).toordinal(),
        axis=1,
    )

    Bfly["Hayakawa22"]["DF"]["FRACYEAR"] = Bfly["Hayakawa22"]["DF"].apply(
        lambda x: x["Year"]
        + (
            datetime.date(int(x["Year"]), int(x["Month"]), int(x["Day"])).toordinal()
            - datetime.date(int(x["Year"]), 1, 1).toordinal()
        )
        / (
            datetime.date(int(x["Year"]) + 1, 1, 1).toordinal()
            - datetime.date(int(x["Year"]), 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Hayakawa22"]["DF"]["Hour"] = 12
    Bfly["Hayakawa22"]["DF"]["Minute"] = 0

    Bfly["Hayakawa22"]["DF"]["DATE"] = Bfly["Hayakawa22"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["Year"]),
            int(x["Month"]),
            int(x["Day"]),
            int(x["Hour"]),
            int(x["Minute"]),
            0,
        ),
        axis=1,
    )

    # Combine latitudes from different observers
    Bfly["Hayakawa22"]["DF"]["ComLatitude"] = Bfly["Hayakawa22"]["DF"]["Latitude"]
    mask = np.logical_and(
        np.isnan(Bfly["Hayakawa22"]["DF"]["ComLatitude"]),
        np.isfinite(Bfly["Hayakawa22"]["DF"]["Staudach"]),
    )
    Bfly["Hayakawa22"]["DF"].loc[mask, "ComLatitude"] = Bfly["Hayakawa22"]["DF"].loc[
        mask, "Staudach"
    ]
    mask = np.logical_and(
        np.isnan(Bfly["Hayakawa22"]["DF"]["ComLatitude"]),
        np.isfinite(Bfly["Hayakawa22"]["DF"]["Wargentin"]),
    )
    Bfly["Hayakawa22"]["DF"].loc[mask, "ComLatitude"] = Bfly["Hayakawa22"]["DF"].loc[
        mask, "Wargentin"
    ]

    # Remove days with spots, but no latitudes
    Bfly["Hayakawa22"]["DF"] = Bfly["Hayakawa22"]["DF"].loc[
        np.logical_or(
            Bfly["Hayakawa22"]["DF"]["G"] == 0,
            np.isfinite(Bfly["Hayakawa22"]["DF"]["ComLatitude"]),
        )
    ]

    Bfly["Hayakawa22"]["DF"]["Area"] = np.nan
    Bfly["Hayakawa22"]["DF"]["MEASURER"] = "Hayakawa"
    Bfly["Hayakawa22"]["DF"]["CYCLE"] = np.nan

    Bfly["Hayakawa22"]["Lat"] = "ComLatitude"
    Bfly["Hayakawa22"]["Lon"] = "Longitude"
    Bfly["Hayakawa22"]["Area"] = "Area"
    Bfly["Hayakawa22"]["Year"] = "Year"
    Bfly["Hayakawa22"]["Month"] = "Month"
    Bfly["Hayakawa22"]["Day"] = "Day"
    Bfly["Hayakawa22"]["Hour"] = "Hour"
    Bfly["Hayakawa22"]["Minute"] = "Minute"

    for observer in np.unique(Bfly["Hayakawa22"]["DF"]["OBSERVER"]):
        # Skip Carbone so that it has a lower priority
        if observer != "Carbone" and observer != "Staudach":
            Bfly[observer] = Bfly["Hayakawa22"].copy()
            Bfly[observer]["DF"] = Bfly[observer]["DF"].loc[
                Bfly[observer]["DF"]["OBSERVER"] == observer
            ]

    # Add Carbone at the end
    observer = "Carbone"
    Bfly[observer] = Bfly["Hayakawa22"].copy()
    Bfly[observer]["DF"] = Bfly[observer]["DF"].loc[
        Bfly[observer]["DF"]["OBSERVER"] == observer
    ]

    del Bfly["Hayakawa22"]

    # --------------------------------------------------------------------------------------------
    # Oriani
    LOG.info("Read Oriani...")

    Bfly["Oriani"] = {}
    Bfly["Oriani"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1778_1779_Oriani_heliographic_coordinates.txt",
        delimiter=r"\t+",
        skiprows=13,
        engine="python",
        on_bad_lines=bad_line_printer,
        encoding="unicode_escape",
    )

    Bfly["Oriani"]["DF"]["ORDINAL"] = Bfly["Oriani"]["DF"].apply(
        lambda x: datetime.date(
            int(x["YEAR"]), int(x["MONTH"]), int(x["DAY"])
        ).toordinal(),
        axis=1,
    )

    Bfly["Oriani"]["DF"]["FRACYEAR"] = Bfly["Oriani"]["DF"].apply(
        lambda x: x["YEAR"]
        + (
            datetime.date(x["YEAR"], x["MONTH"], x["DAY"]).toordinal()
            - datetime.date(x["YEAR"], 1, 1).toordinal()
        )
        / (
            datetime.date(x["YEAR"] + 1, 1, 1).toordinal()
            - datetime.date(x["YEAR"], 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Oriani"]["DF"]["HOUR"] = 12
    Bfly["Oriani"]["DF"]["MINUTE"] = 0

    Bfly["Oriani"]["DF"]["DATE"] = Bfly["Oriani"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YEAR"]),
            int(x["MONTH"]),
            int(x["DAY"]),
            int(x["HOUR"]),
            int(x["MINUTE"]),
            0,
        ),
        axis=1,
    )

    Bfly["Oriani"]["DF"]["AREA"] = np.nan
    Bfly["Oriani"]["DF"]["CYCLE"] = np.nan

    Bfly["Oriani"]["DF"]["MEASURER"] = "Carrasco"
    Bfly["Oriani"]["DF"]["OBSERVER"] = "Oriani"

    # Remove non-latitude measurements
    Bfly["Oriani"]["DF"] = Bfly["Oriani"]["DF"].loc[
        Bfly["Oriani"]["DF"]["LAT"] != "-", :
    ]

    Bfly["Oriani"]["Lat"] = "LAT"
    Bfly["Oriani"]["Lon"] = "LON"
    Bfly["Oriani"]["Area"] = "AREA"
    Bfly["Oriani"]["Year"] = "YEAR"
    Bfly["Oriani"]["Month"] = "MONTH"
    Bfly["Oriani"]["Day"] = "DAY"
    Bfly["Oriani"]["Hour"] = "HOUR"
    Bfly["Oriani"]["Minute"] = "MINUTE"

    # --------------------------------------------------------------------------------------------
    # Horrebow
    LOG.info("Read Horrebow...")

    Bfly["Horrebow"] = {}
    Bfly["Horrebow"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1761-1777_horrebow.txt",
        delimiter=r"\s+",
        engine="python",
        on_bad_lines=bad_line_printer,
        encoding="unicode_escape",
    )

    Bfly["Horrebow"]["DF"]["ORDINAL"] = Bfly["Horrebow"]["DF"].apply(
        lambda x: datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal(),
        axis=1,
    )

    Bfly["Horrebow"]["DF"]["FRACYEAR"] = Bfly["Horrebow"]["DF"].apply(
        lambda x: x["YYYY"]
        + (
            datetime.date(x["YYYY"], x["MM"], x["DD"]).toordinal()
            - datetime.date(x["YYYY"], 1, 1).toordinal()
        )
        / (
            datetime.date(x["YYYY"] + 1, 1, 1).toordinal()
            - datetime.date(x["YYYY"], 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Horrebow"]["DF"]["DATE"] = Bfly["Horrebow"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YYYY"]), int(x["MM"]), int(x["DD"]), int(x["HH"]), int(x["MI"]), 0
        ),
        axis=1,
    )

    Bfly["Horrebow"]["DF"]["OBSERVER"] = "Horrebow"
    Bfly["Horrebow"]["DF"]["MEASURER"] = "Jorgensen"
    Bfly["Horrebow"]["DF"]["CYCLE"] = np.nan
    Bfly["Horrebow"]["DF"]["Area"] = np.nan

    Bfly["Horrebow"]["Lat"] = "BBB.B"
    Bfly["Horrebow"]["Lon"] = "LLL.L"
    Bfly["Horrebow"]["Area"] = "Area"
    Bfly["Horrebow"]["Year"] = "YYYY"
    Bfly["Horrebow"]["Month"] = "MM"
    Bfly["Horrebow"]["Day"] = "DD"
    Bfly["Horrebow"]["Hour"] = "HH"
    Bfly["Horrebow"]["Minute"] = "MI"

    # --------------------------------------------------------------------------------------------
    # Zucconi
    LOG.info("Read Zucconi...")

    Bfly["Zucconi"] = {}
    Bfly["Zucconi"]["DF"] = pd.read_excel(
        DATA_FOLDER + "input_data/1754_1760_zucconi.xls"
    )

    Bfly["Zucconi"]["DF"]["ORDINAL"] = Bfly["Zucconi"]["DF"].apply(
        lambda x: datetime.date(
            int(x["Year"]), int(x["Month"]), int(x["Day"])
        ).toordinal(),
        axis=1,
    )

    Bfly["Zucconi"]["DF"]["FRACYEAR"] = Bfly["Zucconi"]["DF"].apply(
        lambda x: x["Year"]
        + (
            datetime.date(int(x["Year"]), int(x["Month"]), int(x["Day"])).toordinal()
            - datetime.date(int(x["Year"]), 1, 1).toordinal()
        )
        / (
            datetime.date(int(x["Year"]) + 1, 1, 1).toordinal()
            - datetime.date(int(x["Year"]), 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Zucconi"]["DF"]["AREA"] = np.nan
    Bfly["Zucconi"]["DF"]["LONGITUDE"] = np.nan

    Bfly["Zucconi"]["DF"]["HOUR"] = 12
    Bfly["Zucconi"]["DF"]["MINUTE"] = 0

    Bfly["Zucconi"]["DF"]["DATE"] = Bfly["Zucconi"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["Year"]),
            int(x["Month"]),
            int(x["Day"]),
            int(x["HOUR"]),
            int(x["MINUTE"]),
            0,
        ),
        axis=1,
    )

    Bfly["Zucconi"]["DF"]["MEASURER"] = "Carrasco"
    Bfly["Zucconi"]["DF"]["OBSERVER"] = "Zucconi"
    Bfly["Zucconi"]["DF"]["CYCLE"] = np.nan

    Bfly["Zucconi"]["Lat"] = "Latitude"
    Bfly["Zucconi"]["Lon"] = "LONGITUDE"
    Bfly["Zucconi"]["Area"] = "AREA"
    Bfly["Zucconi"]["Year"] = "Year"
    Bfly["Zucconi"]["Month"] = "Month"
    Bfly["Zucconi"]["Day"] = "Day"
    Bfly["Zucconi"]["Hour"] = "HOUR"
    Bfly["Zucconi"]["Minute"] = "MINUTE"

    # --------------------------------------------------------------------------------------------
    # Staudacher
    LOG.info("Read Staudacher...")

    Bfly["Staudacher"] = {}
    Bfly["Staudacher"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1749_1796_staudacher_v1.2_20150304_groupnames.txt",
        delimiter=r"\s+",
        skiprows=30,
        engine="python",
        on_bad_lines=bad_line_ignore,
    )

    Bfly["Staudacher"]["DF"]["ORDINAL"] = Bfly["Staudacher"]["DF"].apply(
        lambda x: datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal(),
        axis=1,
    )

    Bfly["Staudacher"]["DF"]["FRACYEAR"] = Bfly["Staudacher"]["DF"].apply(
        lambda x: x["YYYY"]
        + (
            datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal()
            - datetime.date(int(x["YYYY"]), 1, 1).toordinal()
        )
        / (
            datetime.date(int(x["YYYY"]) + 1, 1, 1).toordinal()
            - datetime.date(int(x["YYYY"]), 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Staudacher"]["DF"].loc[
        np.isnan(Bfly["Staudacher"]["DF"].loc[:, "HH"]), "HH"
    ] = 12
    Bfly["Staudacher"]["DF"].loc[
        np.isnan(Bfly["Staudacher"]["DF"].loc[:, "MI"]), "MI"
    ] = 0

    Bfly["Staudacher"]["DF"]["DATE"] = Bfly["Staudacher"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YYYY"]), int(x["MM"]), int(x["DD"]), int(x["HH"]), int(x["MI"]), 0
        ),
        axis=1,
    )

    Bfly["Staudacher"]["DF"]["Area"] = np.nan
    Bfly["Staudacher"]["DF"]["MEASURER"] = "Arlt"
    Bfly["Staudacher"]["DF"]["OBSERVER"] = "Staudacher"
    Bfly["Staudacher"]["DF"]["CYCLE"] = np.nan

    Bfly["Staudacher"]["Lat"] = "BBB.B"
    Bfly["Staudacher"]["Lon"] = "LLL.L"
    Bfly["Staudacher"]["Area"] = "Area"
    Bfly["Staudacher"]["Year"] = "YYYY"
    Bfly["Staudacher"]["Month"] = "MM"
    Bfly["Staudacher"]["Day"] = "DD"
    Bfly["Staudacher"]["Hour"] = "HH"
    Bfly["Staudacher"]["Minute"] = "MI"

    # --------------------------------------------------------------------------------------------
    # Prantner
    LOG.info("Read Prantner...")

    Bfly["Prantner"] = {}
    Bfly["Prantner"]["DF"] = pd.read_table(
        DATA_FOLDER
        + "input_data/1804_1844_Hayakawa_2021_ApJ_919_1_whole_data_of_Prantner.txt",
        delimiter=r"\s+",
        skiprows=25,
        engine="python",
        on_bad_lines=bad_line_ignore,
    )

    Bfly["Prantner"]["DF"]["ORDINAL"] = Bfly["Prantner"]["DF"].apply(
        lambda x: datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal(),
        axis=1,
    )

    Bfly["Prantner"]["DF"]["FRACYEAR"] = Bfly["Prantner"]["DF"].apply(
        lambda x: x["YYYY"]
        + (
            datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal()
            - datetime.date(int(x["YYYY"]), 1, 1).toordinal()
        )
        / (
            datetime.date(int(x["YYYY"]) + 1, 1, 1).toordinal()
            - datetime.date(int(x["YYYY"]), 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Prantner"]["DF"].loc[np.isnan(Bfly["Prantner"]["DF"].loc[:, "HH"]), "HH"] = 12
    Bfly["Prantner"]["DF"].loc[Bfly["Prantner"]["DF"]["HH"] > 23, "HH"] = 12
    Bfly["Prantner"]["DF"]["MI"] = 0

    Bfly["Prantner"]["DF"]["DATE"] = Bfly["Prantner"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YYYY"]), int(x["MM"]), int(x["DD"]), int(x["HH"]), int(x["MI"]), 0
        ),
        axis=1,
    )

    Bfly["Prantner"]["DF"]["MEASURER"] = "Hayakawa-etal"
    Bfly["Prantner"]["DF"]["OBSERVER"] = "Prantner"
    Bfly["Prantner"]["DF"]["CYCLE"] = np.nan

    Bfly["Prantner"]["Lat"] = "BB"
    Bfly["Prantner"]["Lon"] = "LL"
    Bfly["Prantner"]["Area"] = "TOTAL_AREA"
    Bfly["Prantner"]["Year"] = "YYYY"
    Bfly["Prantner"]["Month"] = "MM"
    Bfly["Prantner"]["Day"] = "DD"
    Bfly["Prantner"]["Hour"] = "HH"
    Bfly["Prantner"]["Minute"] = "MI"

    # --------------------------------------------------------------------------------------------
    # Linderer
    LOG.info("Read Linderer...")

    Bfly["Linderer"] = {}
    Bfly["Linderer"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1800_1827_lindener_v1.1_20230126.txt",
        delimiter=r"\s+",
        skiprows=55,
        engine="python",
        on_bad_lines=bad_line_ignore,
    )

    Bfly["Linderer"]["DF"]["ORDINAL"] = Bfly["Linderer"]["DF"].apply(
        lambda x: datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal(),
        axis=1,
    )

    Bfly["Linderer"]["DF"]["FRACYEAR"] = Bfly["Linderer"]["DF"].apply(
        lambda x: x["YYYY"]
        + (
            datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal()
            - datetime.date(int(x["YYYY"]), 1, 1).toordinal()
        )
        / (
            datetime.date(int(x["YYYY"]) + 1, 1, 1).toordinal()
            - datetime.date(int(x["YYYY"]), 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Linderer"]["DF"].loc[np.isnan(Bfly["Linderer"]["DF"].loc[:, "HH"]), "HH"] = 12
    Bfly["Linderer"]["DF"].loc[np.isnan(Bfly["Linderer"]["DF"].loc[:, "MI"]), "MI"] = 0

    Bfly["Linderer"]["DF"]["DATE"] = Bfly["Linderer"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YYYY"]), int(x["MM"]), int(x["DD"]), int(x["HH"]), int(x["MI"]), 0
        ),
        axis=1,
    )

    Bfly["Linderer"]["DF"]["Area"] = np.nan
    Bfly["Linderer"]["DF"]["MEASURER"] = "Hayakawa-etal"
    Bfly["Linderer"]["DF"]["OBSERVER"] = "Linderer"
    Bfly["Linderer"]["DF"]["CYCLE"] = np.nan

    Bfly["Linderer"]["Lat"] = "LAT"
    Bfly["Linderer"]["Lon"] = "CMD"
    Bfly["Linderer"]["Area"] = "Area"
    Bfly["Linderer"]["Year"] = "YYYY"
    Bfly["Linderer"]["Month"] = "MM"
    Bfly["Linderer"]["Day"] = "DD"
    Bfly["Linderer"]["Hour"] = "HH"
    Bfly["Linderer"]["Minute"] = "MI"

    # --------------------------------------------------------------------------------------------
    # Derfflinger
    LOG.info("Read Derfflinger...")

    Bfly["Derfflinger"] = {}
    Bfly["Derfflinger"]["DF"] = pd.read_table(
        DATA_FOLDER + "input_data/1802_1824_derfflinger_supplemet_positions_2.txt",
        delimiter=r"\s+",
        skiprows=42,
        engine="python",
        on_bad_lines=bad_line_ignore,
    )

    Bfly["Derfflinger"]["DF"]["ORDINAL"] = Bfly["Derfflinger"]["DF"].apply(
        lambda x: datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal(),
        axis=1,
    )

    Bfly["Derfflinger"]["DF"]["FRACYEAR"] = Bfly["Derfflinger"]["DF"].apply(
        lambda x: x["YYYY"]
        + (
            datetime.date(int(x["YYYY"]), int(x["MM"]), int(x["DD"])).toordinal()
            - datetime.date(int(x["YYYY"]), 1, 1).toordinal()
        )
        / (
            datetime.date(int(x["YYYY"]) + 1, 1, 1).toordinal()
            - datetime.date(int(x["YYYY"]), 1, 1).toordinal()
        ),
        axis=1,
    )

    Bfly["Derfflinger"]["DF"].loc[
        np.isnan(Bfly["Derfflinger"]["DF"].loc[:, "HH"]), "HH"
    ] = 12
    Bfly["Derfflinger"]["DF"].loc[
        np.isnan(Bfly["Derfflinger"]["DF"].loc[:, "MI"]), "MI"
    ] = 0

    Bfly["Derfflinger"]["DF"]["DATE"] = Bfly["Derfflinger"]["DF"].apply(
        lambda x: datetime.datetime(
            int(x["YYYY"]), int(x["MM"]), int(x["DD"]), int(x["HH"]), int(x["MI"]), 0
        ),
        axis=1,
    )

    Bfly["Derfflinger"]["DF"]["Area"] = np.nan
    Bfly["Derfflinger"]["DF"]["MEASURER"] = "Hayakawa-etal"
    Bfly["Derfflinger"]["DF"]["OBSERVER"] = "Derfflinger"
    Bfly["Derfflinger"]["DF"]["CYCLE"] = np.nan

    Bfly["Derfflinger"]["Lat"] = "Lat"
    Bfly["Derfflinger"]["Lon"] = "Lon"
    Bfly["Derfflinger"]["Area"] = "Area"
    Bfly["Derfflinger"]["Year"] = "YYYY"
    Bfly["Derfflinger"]["Month"] = "MM"
    Bfly["Derfflinger"]["Day"] = "DD"
    Bfly["Derfflinger"]["Hour"] = "HH"
    Bfly["Derfflinger"]["Minute"] = "MI"

    # --------------------------------------------------------------------------------------------
    # Concatenate all observers and set each observation to 1
    LOG.info("Concatenate all observers...")

    observations = pd.DataFrame(columns=["ORDINAL", "OBSERVER"])
    for observer in Bfly.keys():
        observations = pd.concat(
            [observations, Bfly[observer]["DF"].loc[:, ["ORDINAL", "OBSERVER"]]]
        ).reset_index(drop=True)
    observations["OBSERVATION"] = 1

    # Get unique observers per day
    observers_each_day = observations.groupby(["ORDINAL", "OBSERVER"]).mean()

    # Get the number of unique observers per day
    observers_per_day = observers_each_day.groupby("ORDINAL").sum()

    # Get the days with more than one unique observer
    index = observers_per_day.loc[observers_per_day["OBSERVATION"] > 1, :].index

    # Retrieve the ordered list of days with more than one observer
    overlaps = observations.loc[observations["ORDINAL"].isin(index), :]

    # Assemble lists of observers in the order of priority
    observer_priority_list = []
    for ordinal in np.unique(overlaps["ORDINAL"]):
        tmp = overlaps.loc[overlaps["ORDINAL"] == ordinal, ["ORDINAL", "OBSERVER"]]
        tmp = tmp.loc[~tmp.duplicated("OBSERVER"), :]
        observer_sequence = None
        for index, row in tmp.iterrows():
            if observer_sequence is None:
                observer_sequence = row["OBSERVER"]
            else:
                observer_sequence = observer_sequence + "->" + row["OBSERVER"]
        observer_priority_list.append(observer_sequence)

    sequences = (
        pd.DataFrame(np.unique(observer_priority_list), columns=["Sequence"])
        .sort_values("Sequence")
        .reset_index(drop=True)
    )
    sequences.to_csv(OUTPUT_FOLDER + "0_observer_priority_Sequences.csv", index=False)

    columns = [
        "DATE",
        "FRACYEAR",
        "ORDINAL",
        "YEAR",
        "MONTH",
        "DAY",
        "HOUR",
        "MINUTE",
        "LATITUDE",
        "LONGITUDE",
        "AREA",
        "CYCLE",
        "OBSERVER",
        "MEASURER",
    ]
    BflyAllDF = None

    for observer in Bfly.keys():
        observer_df = Bfly[observer]["DF"]
        lat = Bfly[observer]["Lat"]
        lon = Bfly[observer]["Lon"]
        area = Bfly[observer]["Area"]
        year = Bfly[observer]["Year"]
        month = Bfly[observer]["Month"]
        day = Bfly[observer]["Day"]
        hour = Bfly[observer]["Hour"]
        minute = Bfly[observer]["Minute"]

        tmp = observer_df.loc[
            :,
            [
                "DATE",
                "FRACYEAR",
                "ORDINAL",
                year,
                month,
                day,
                hour,
                minute,
                lat,
                lon,
                area,
                "CYCLE",
                "OBSERVER",
                "MEASURER",
            ],
        ]
        rename_mapper = {
            year: "YEAR",
            month: "MONTH",
            day: "DAY",
            hour: "HOUR",
            minute: "MINUTE",
            lat: "LATITUDE",
            lon: "LONGITUDE",
            area: "AREA",
        }
        tmp = tmp.rename(columns=rename_mapper)

        if BflyAllDF is None:
            BflyAllDF = tmp
        else:
            tmp = tmp.loc[~np.in1d(tmp["ORDINAL"], BflyAllDF["ORDINAL"]), :]
            BflyAllDF = pd.concat([BflyAllDF, tmp], ignore_index=True, join="inner")

    BflyAllDF = BflyAllDF.sort_values(by=["DATE"]).reset_index(drop=True)
    BflyAllDF["ORDINAL"] = BflyAllDF["ORDINAL"].astype(int)
    BflyAllDF["LATITUDE"] = BflyAllDF["LATITUDE"].astype(float)
    BflyAllDF.to_csv(OUTPUT_FOLDER + "1_BflyAll_gassendi_Vaquero.csv", index=False)

    BfObsCv, BfObsYr = calculate_observed_fraction(
        BflyAllDF, Y1=1600, Y2=2024, YrCum=YRCUM
    )

    bfly400_plot(FIGURE_FOLDER + f'1_data_merge.png', BflyAllDF, BfObsYr, BfObsCv, gn, YrCum=YRCUM)
