import numpy as np
import colorcet as cc
import matplotlib.pyplot as plt
from matplotlib import colors as clrs
import pandas as pd

FIGDPI = 100

# Color mask
colorsMsk = [(0.65, 0.5, 0.35), (0.60, 0.60, 0.4), (0.42, 0.5, 0.56), (0.5, 0.5, 0.5)]

# Colormap for Bfly coverage
colors = [(0.65, 0.5, 0.35), (0.60, 0.60, 0.4), (0.42, 0.5, 0.56), (0.5, 0.5, 0.5)]
color0 = (0.61, 0.38, 0.38)
cmapMsk = clrs.LinearSegmentedColormap.from_list("cmapMsk", colors, N=5)
norm = clrs.Normalize(vmin=0, vmax=0.15, clip=True)

# Font Size
font = {"family": "sans-serif", "weight": "normal", "size": 30}

# Size definitions
dpi = 300
pxx = 10000  # Horizontal size of each panel
pxy = 1500  # Vertical size of each panel

nph = 1  # Number of horizontal panels
npv = 2  # Number of vertical panels

# Padding
padv = 300  # Vertical padding in pixels
padv2 = 0  # Vertical padding in pixels between panels
padh = 600  # Horizontal padding in pixels at the edge of the figure
padh2 = 350  # Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = npv * pxy + 2 * padv + (npv - 1) * padv2  # Vertical size of figure in pixels
fszh = nph * pxx + 2 * padh + (nph - 1) * padh2  # Horizontal size of figure in pixels

# Conversion to relative unites
ppxx = pxx / fszh
ppxy = pxy / fszv
ppadv = padv / fszv  # Vertical padding in relative units
ppadv2 = padv2 / fszv  # Vertical padding in relative units
ppadh = padh / fszh  # Horizontal padding the edge of the figure in relative units
ppadh2 = padh2 / fszh  # Horizontal padding between panels in relative units


def bfly400_plot(
    save_path,
    BflyAllDF,
    BfObsYr,
    BfObsCv,
    GN,
    KDE_norm:np.array = None,
    binsx:np.array = None,
    binsy:np.array = None,
    cycle_fits: pd.DataFrame = None,
    median_lat_dict: dict = None,
    YrCum: float = 1,
    MaskMaxAl: float = 0.8,
    Y1:float=1605,
    Y2:float=2030,

):

    fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))
    plt.rc("font", **font)
    cmap = cc.cm.glasbey_dark

    ax3 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

    # Plot transparency mask
    for i in np.arange(0, BfObsYr.shape[0]):
        if BfObsCv[i] == 0:
            clr = color0
        else:
            clr = colorsMsk[(int(np.min([3, np.floor(BfObsCv[i] / 0.05)])))]
        ax3.fill(
            [BfObsYr[i], BfObsYr[i], BfObsYr[i] + YrCum, BfObsYr[i] + YrCum],
            [-60, 60, 60, -60],
            color=clr,
            alpha=(1 - BfObsCv[i]) * MaskMaxAl,
            zorder=0,
            edgecolor="none",
        )

    ax3.scatter(BflyAllDF["FRACYEAR"], BflyAllDF["LATITUDE"], s=20, color="k")

    if cycle_fits is not None:
        color_offset = 70
        light_lim = 0.5
        for index, row in cycle_fits.iterrows():

            c = cmap.colors[index + color_offset]
            luma = 0.212 * c[0] + 0.701 * c[1] + 0.087 * c[2]

            # search for the next color that is dark enough
            if luma > light_lim:

                while luma > light_lim:
                    color_offset = color_offset + 1
                    c = cmap.colors[index + color_offset]
                    luma = 0.212 * c[0] + 0.701 * c[1] + 0.087 * c[2]

            # Plot sunspots
            mask_cyc = BflyAllDF["CYCLEN"] == row["CycleN"]
            ax3.scatter(
                BflyAllDF.loc[mask_cyc, "FRACYEAR"],
                BflyAllDF.loc[mask_cyc, "LATITUDE"],
                s=20,
                color=c,
            )

            if median_lat_dict is not None:
                # Plot median path
                x = median_lat_dict[row["CycleN"]]["N"]["Year"]
                y = median_lat_dict[row["CycleN"]]["N"]["Median"]
                yh = median_lat_dict[row["CycleN"]]["N"]["MedianH"]
                yl = median_lat_dict[row["CycleN"]]["N"]["MedianL"]
                ax3.fill_between(x, yl, y2=yh, step="mid", ec="None", fc="k", alpha=0.4)
                ax3.plot(x, y, ds="steps-mid", c="k")

                x = median_lat_dict[row["CycleN"]]["S"]["Year"]
                y = median_lat_dict[row["CycleN"]]["S"]["Median"]
                yh = median_lat_dict[row["CycleN"]]["S"]["MedianH"]
                yl = median_lat_dict[row["CycleN"]]["S"]["MedianL"]
                ax3.fill_between(x, yl, y2=yh, step="mid", ec="None", fc="k", alpha=0.4)
                ax3.plot(x, y, ds="steps-mid", c="k")

            if 'cntrd_tau' in row.keys():

                if row["CycleN"] != "M3":

                    # plot fits
                    cycle = row["Cycle"] + 10
                    x = np.arange(-15, 10, 0.1)
                    b = row["cntrd_tau"]

                    if row["CycleN"] != "M2" and row["CycleN"] != "M4":
                        # Northern hemisphere
                        x0 = row["cntrd_offstN"]
                        dx = row["cntrd_offstN_sig"]
                        y = 10 * np.exp(-x / b)
                        ax3.plot(x + x0, y, c="r")
                        ax3.fill_betweenx(
                            y,
                            x1=x + x0 - dx,
                            x2=x + x0 + dx,
                            color="r",
                            alpha=0.25,
                            edgecolor="none",
                        )

                    # Southern hemisphere
                    x0 = row["cntrd_offstS"]
                    dx = row["cntrd_offstS_sig"]
                    y = -10 * np.exp(-x / b)
                    ax3.plot(x + x0, y, c="r")
                    ax3.fill_betweenx(
                        y,
                        x1=x + x0 - dx,
                        x2=x + x0 + dx,
                        color="r",
                        alpha=0.25,
                        edgecolor="none",
                    )

            ax3.text(
                row["bndr_offstS"],
                -47,
                row["CycleN"],
                color=c,
                ha="right",
                zorder=11,
                va="top",
            )

    ax3.set_xlim(left=Y1, right=Y2)
    ax3.set_ylim(top=59.5, bottom=-59.5)
    ax3.set_ylabel("Latitude (o)")
    ax3.grid(color=(0.5, 0.5, 0.5), linestyle="--", linewidth=1, axis="x", which="both")
    # ax3.set_facecolor('k')
    ax3.annotate(
        "(b)", xy=(1, 0.99), xycoords="axes fraction", fontsize=60, va="top", ha="right"
    )

    ax3.plot([Y1, Y2], np.array([1, 1]) * 0, c="k", lw=2, ls="--")
    ax3.plot([Y1, Y2], np.array([1, 1]) * 10, c="k", lw=2, ls=":")
    ax3.plot([Y1, Y2], np.array([1, 1]) * -10, c="k", lw=2, ls=":")

    ax4 = fig.add_axes([ppadh, ppadv + ppxy, ppxx, ppxy])

    for observer in GN.keys():
        ax4.plot(GN[observer]["Year"], GN[observer]["GN"], label=observer)

    ax4.legend(frameon=False, loc="upper left")

    ax4.set_xlim(left=Y1, right=Y2)
    ax4.set_ylim(top=15, bottom=0)

    ax4.xaxis.tick_top()
    ax4.xaxis.set_label_position("top")
    ax4.grid(color=(0.5, 0.5, 0.5), linestyle="--", linewidth=1, axis="x", which="both")

    ax4.annotate(
        "(a)", xy=(1, 0.99), xycoords="axes fraction", fontsize=60, va="top", ha="right"
    )

    # Add Colorbar
    axcb3 = fig.add_axes([ppadh + ppxx, ppadv, ppadh2, ppxy])

    axcb3.fill(
        [0, 0, 1, 1], [0, 1, 1, 0], color=color0, alpha=MaskMaxAl, edgecolor="none"
    )
    axcb3.annotate(
        "0%",
        xy=(0.5, 0.5),
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=27,
        zorder=3,
    )
    for i in range(1, 4):
        clr = colorsMsk[i - 1]
        axcb3.fill(
            [0, 0, 1, 1],
            [i, i + 1, i + 1, i],
            color=clr,
            alpha=(1 - i / 20) * MaskMaxAl,
            edgecolor="none",
        )
        axcb3.annotate(
            "<" + str(5 * i) + "%",
            xy=(0.5, i + 0.5),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=27,
            zorder=3,
        )

    axcb3.set_ylim(bottom=0, top=4)
    axcb3.set_xlim(left=0, right=1)

    axcb3.set_xticks([])
    axcb3.set_yticks([])
    axcb3.set_xlabel("Obsv.\nCoverage")

    # Add Colorbar
    axcb2 = fig.add_axes([ppadh + ppxx, ppadv + ppxy, ppadh2, ppxy])
    for i in range(1, 6):
        axcb2.fill(
            [0, 0, 1, 1],
            [i, i + 1, i + 1, i],
            color=(0.5, 0.5, 0.5),
            alpha=(1 - i / 5) * MaskMaxAl,
            edgecolor="none",
        )
        axcb2.annotate(
            str(20 * i) + "%",
            xy=(0.5, i + 0.5),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=27,
            zorder=3,
        )

    axcb2.set_ylim(bottom=1, top=6)
    axcb2.set_xlim(left=0, right=1)

    axcb2.set_xticks([])
    axcb2.set_yticks([])

    if cycle_fits is not None and KDE_norm is not None and binsx is not None and binsy is not None:

        cmap = 'gray_r'
        ax2 = fig.add_axes([ppadh, ppadv-ppxy, ppxx, ppxy])
        ax2.pcolor(binsx, binsy, KDE_norm, cmap=cmap, vmax=np.max(KDE_norm)/1.75, vmin=np.max(KDE_norm)/200)
        ax2.plot([Y1, Y2], np.array([1,1])*0, c='c', lw=2, ls='--')

        # Plot transparency mask
        for i in np.arange(0, BfObsYr.shape[0]):
            if BfObsCv[i] == 0:
                ax2.fill(
                    [BfObsYr[i], BfObsYr[i], BfObsYr[i] + YrCum, BfObsYr[i] + YrCum],
                    [-60, 60, 60, -60],
                    color=clr,
                    alpha=1,
                    zorder=10,
                    edgecolor="none",
                )

                
        ax2.set_xlim(left=Y1, right=Y2)
        ax2.set_ylim(top=59.5, bottom=-59.5)
        ax2.set_ylabel("Latitude (o)")
        ax2.grid(color=(0.5, 0.5, 0.5), linestyle="--", linewidth=1, axis="x", which="both")
        ax2.set_xticklabels([])

        cmap = cc.cm.glasbey_dark

        color_offset = 70
        light_lim = 0.5
        for index, row in cycle_fits.iterrows():

            c = cmap.colors[index + color_offset]
            luma = 0.212 * c[0] + 0.701 * c[1] + 0.087 * c[2]

            # search for the next color that is dark enough
            if luma > light_lim:

                while luma > light_lim:
                    color_offset = color_offset + 1
                    c = cmap.colors[index + color_offset]
                    luma = 0.212 * c[0] + 0.701 * c[1] + 0.087 * c[2]

            xr = row['bndr_offstN'] - np.arange(0, 15, 0.01)
            x = row['bndr_offstN'] - xr
            y = np.power(x, row['bndr_expnt'])*row['bndr_slpN']
            ax2.plot(xr,y, zorder=22, lw=2, color = c)

            xr = row['bndr_offstS'] - np.arange(0, 15, 0.01)
            x = row['bndr_offstS'] - xr
            y = np.power(x, row['bndr_expnt'])*row['bndr_slpS']
            ax2.plot(xr,-y, zorder=22, lw=2, color = c)
            ax2.text(row['bndr_offstS'], -47, row['CycleN'], color=c, ha='right', zorder=11, va='top')    




    fig.savefig(save_path, dpi=FIGDPI, bbox_inches='tight')
    plt.close(fig)
