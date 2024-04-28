import datetime
import numpy as np



def ordinal2fracy(Ord):
    YEAR = datetime.date.fromordinal(Ord).year
    return YEAR + (Ord - datetime.date(YEAR, 1, 1).toordinal()) / (
        datetime.date(YEAR + 1, 1, 1).toordinal()
        - datetime.date(YEAR, 1, 1).toordinal()
    )


def calculate_observed_fraction(Bfly, Y1 = 1600, Y2 = 2025, YrCum = 1, slss_ds=None):

    vordinal2fracy = np.vectorize(ordinal2fracy)

    # Calculate a unique list of days represented in the butterfly diagram
    ObsDsBfly = np.unique(Bfly["Ordinal"])

    if slss_ds is not None:
        ObsDsBfly = np.concatenate([ObsDsBfly, np.unique(slss_ds['Ordinal'])])

    # Calculate the corresponding fractional year
    BfObsFy = vordinal2fracy(ObsDsBfly)

    # Set up repository variables
    BfObsYr = np.arange(Y1, Y2 + YrCum, YrCum)
    BfObsCv = BfObsYr.copy().astype(float) * 0

    for i in np.arange(0, BfObsYr.shape[0]):
        # Calculate number of days in year
        NdaysYr = (
            datetime.date(BfObsYr[i] + YrCum, 1, 1).toordinal()
            - datetime.date(BfObsYr[i], 1, 1).toordinal()
        )

        # Find number of observations in first year of the bin
        NdaysOb = np.sum(np.floor(BfObsFy) == BfObsYr[i])

        # Add other years to the bin
        for j in np.arange(1, YrCum):
            NdaysOb = NdaysOb + np.sum(np.floor(BfObsFy) == BfObsYr[i] + j)

        # Calculate coverage
        BfObsCv[i] = NdaysOb / NdaysYr
    
    return BfObsCv, BfObsYr


