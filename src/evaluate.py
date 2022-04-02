import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plotting(
        ax, title,
        x_label, y_label,
        xactual, yactual, actual_label,
        x_binary=None, y_binary=None,
        x_sdf=None, y_sdf=None,
        label_binary="",
        label_sdf="",
):
    ax.set_title = title
    ax.plot(xactual, yactual, label=actual_label)

    if x_binary is not None and y_binary is not None:
        ax.plot(x_binary, y_binary, "x", markersize=4, label=label_binary)

    if x_sdf is not None and y_sdf is not None:
        ax.plot(x_sdf, y_sdf, "+", markersize=4, label=label_sdf)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()


def rsquare(y, yhat):
    res = []
    total = []
    ymean = sum(y) / len(y)

    for i in range(len(yhat)):
        res.append((y[i] - yhat[i]) ** 2)
        total.append((y[i] - ymean) ** 2)

    # sum of squares of residuals
    ssres = sum(res)

    #  total sum of squares
    sstot = sum(total)

    return 1 - ssres / sstot


if __name__ == "__main__":
    pred_sdf = pd.read_csv("../prediction_sdf.csv")
    pred_binary = pd.read_csv("../prediction_binary.csv")
    pred_nama = pred_binary["name"].unique()[0]

    aktual = pd.read_csv("../out.csv")
    aktual = aktual.loc[aktual["name"].str.replace(" ", "") == pred_nama]
    aktual = aktual[["alpha", "cl", "cd", "cm"]]
    aktual = aktual.sort_values("alpha")

    # plt.style.use("bmh")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
    fig_score, ((ax1_score, ax2_score, ax3_score), (ax4_score, ax5_score, ax6_score)) = plt.subplots(2, 3,
                                                                                                     figsize=(10, 8))
    fig.tight_layout(pad=4.0)
    fig_score.tight_layout(pad=4.0)

    plt.suptitle(pred_nama, fontsize=14)

    plotting(
        ax1, "Cl - Alpha", "alpha", "Cl",
        aktual["alpha"].to_numpy(), aktual["cl"].to_numpy(), "Actual",
        x_binary=pred_binary["alpha"].to_numpy(), y_binary=pred_binary["cl"].to_numpy(),
        x_sdf=pred_sdf["alpha"].to_numpy(), y_sdf=pred_sdf["cl"].to_numpy(),
        label_binary="Binary", label_sdf="SDF"
    )

    plotting(
        ax2, "Cd - Alpha", "alpha", "Cd",
        aktual["alpha"].to_numpy(), aktual["cd"].to_numpy(), "Actual",
        x_binary=pred_binary["alpha"].to_numpy(), y_binary=pred_binary["cd"].to_numpy(),
        x_sdf=pred_sdf["alpha"].to_numpy(), y_sdf=pred_sdf["cd"].to_numpy(),
        label_binary="Binary", label_sdf="SDF"
    )

    plotting(
        ax3, "Cm - Alpha", "alpha", "Cm",
        aktual["alpha"].to_numpy(), aktual["cm"].to_numpy(), "Actual",
        x_binary=pred_binary["alpha"].to_numpy(), y_binary=pred_binary["cm"].to_numpy(),
        x_sdf=pred_sdf["alpha"].to_numpy(), y_sdf=pred_sdf["cm"].to_numpy(),
        label_binary="Binary", label_sdf="SDF"
    )

    # Score Cl
    x1 = aktual["cl"].to_numpy()
    y1_binary = pred_binary["cl"].to_numpy()
    y1_sdf = pred_sdf["cl"].to_numpy()

    # if y1_binary.size > x1.size:
    #     y1_binary = y1_binary[:(x1.size - y1_binary.size)]

    coef1_binary = np.polyfit(x1, y1_binary, 1)
    poly1_binary = np.poly1d(coef1_binary)
    rsq1_binary = rsquare(x1, y1_binary)

    plotting(
        ax1_score, "Score Cl", "Actual Cl", "Predicted Cl",
        x1, poly1_binary(x1), f"$r^2_{{Binary}} = {round(rsq1_binary, 2)}$",
        x_binary=x1, y_binary=y1_binary
    )

    coef1_sdf = np.polyfit(x1, y1_sdf, 1)
    poly1_sdf = np.poly1d(coef1_sdf)
    rsq1_sdf = rsquare(x1, y1_sdf)

    plotting(
        ax4_score, "Score Cl", "Actual Cl", "Predicted Cl",
        x1, poly1_sdf(x1), f"$r^2_{{SDF}} = {round(rsq1_sdf, 2)}$",
        x_sdf=x1, y_sdf=y1_sdf
    )

    # Score Cd
    x2 = aktual["cd"].to_numpy()
    y2_binary = pred_binary["cd"].to_numpy()
    y2_sdf = pred_sdf["cd"].to_numpy()

    coef2_binary = np.polyfit(x2, y2_binary, 1)
    poly2_binary = np.poly1d(coef2_binary)
    rsq2_binary = rsquare(x2, y2_binary)

    plotting(
        ax2_score, "Score Cd", "Actual Cd", "Predicted Cd",
        x2, poly2_binary(x2), f"$r^2_{{Binary}} = {round(rsq2_binary, 2)}$",
        x_binary=x2, y_binary=y2_binary
    )

    coef2_sdf = np.polyfit(x2, y2_sdf, 1)
    poly2_sdf = np.poly1d(coef2_sdf)
    rsq2_sdf = rsquare(x2, y2_sdf)

    plotting(
        ax5_score, "Score Cd", "Actual Cd", "Predicted Cd",
        x2, poly2_sdf(x2), f"$r^2_{{SDF}} = {round(rsq2_sdf, 2)}$",
        x_sdf=x2, y_sdf=y2_sdf
    )

    # Score Cm
    x3 = aktual["cm"].to_numpy()
    y3_binary = pred_binary["cm"].to_numpy()
    y3_sdf = pred_sdf["cm"].to_numpy()

    coef3_binary = np.polyfit(x3, y3_binary, 1)
    poly3_binary = np.poly1d(coef3_binary)
    rsq3_binary = rsquare(x3, y3_binary)

    plotting(
        ax3_score, "Score Cm", "Actual Cm", "Predicted Cm",
        x3, poly3_binary(x3), f"$r^2_{{Binary}} = {round(rsq3_binary, 2)}$",
        x_binary=x3, y_binary=y3_binary
    )

    coef3_sdf = np.polyfit(x3, y3_sdf, 1)
    poly3_sdf = np.poly1d(coef3_sdf)
    rsq3_sdf = rsquare(x3, y3_sdf)

    plotting(
        ax6_score, "Score Cm", "Actual Cm", "Predicted Cm",
        x3, poly3_sdf(x3), f"$r^2_{{SDF}} = {round(rsq3_sdf, 2)}$",
        x_sdf=x3, y_sdf=y3_sdf
    )

    plt.show()
