import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plotting(
        ax,
        title,
        xactual, yactual,
        xpred, ypred,
        actual_label, pred_label,
        x_label, y_label,
        marker
):
    ax.set_title = title
    ax.plot(xactual, yactual, label=actual_label)
    ax.plot(xpred, ypred, marker, label=pred_label)
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
    pred = pd.read_csv("../prediction.csv")
    pred_nama = pred["name"].unique()[0]

    aktual = pd.read_csv("../out.csv")
    aktual = aktual.loc[aktual["name"].str.replace(" ", "") == pred_nama]
    aktual = aktual[["alpha", "cl", "cd", "cm"]]
    aktual = aktual.sort_values("alpha")

    # plt.style.use("bmh")
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 8))
    fig.tight_layout(pad=4.0)

    plt.suptitle(pred_nama, fontsize=14)

    plotting(
        ax1,
        "Cl - Alpha",
        aktual["alpha"].to_numpy(), aktual["cl"].to_numpy(),
        pred["alpha"].to_numpy(), pred["cl"].to_numpy(),
        "Actual", "Prediction",
        "alpha", "Cl", "x"
    )

    plotting(
        ax2,
        "Cd - Alpha",
        aktual["alpha"].to_numpy(), aktual["cd"].to_numpy(),
        pred["alpha"].to_numpy(), pred["cd"].to_numpy(),
        "Actual", "Prediction",
        "alpha", "Cd", "x"
    )

    plotting(
        ax3,
        "Cm - Alpha",
        aktual["alpha"].to_numpy(), aktual["cm"].to_numpy(),
        pred["alpha"].to_numpy(), pred["cm"].to_numpy(),
        "Actual", "Prediction",
        "alpha", "Cm", "x"
    )

    x4 = aktual["cl"].to_numpy()
    y4 = pred["cl"].to_numpy()

    if y4.size > x4.size:
        y4 = y4[:(x4.size - y4.size)]

    coef4 = np.polyfit(x4, y4, 1)
    poly4 = np.poly1d(coef4)
    rsq5 = rsquare(x4, y4)
    plotting(
        ax4,
        "",
        x4, poly4(x4),
        x4, y4,
        f"$r^2 = {round(rsq5, 2)}$", "",
        "Actual Cl", "Predicted Cl", "o"
    )

    x5 = aktual["cd"].to_numpy()
    y5 = pred["cd"].to_numpy()

    if y5.size > x5.size:
        y5 = y5[:(x5.size - y5.size)]

    coef5 = np.polyfit(x5, y5, 1)
    poly5 = np.poly1d(coef5)
    rsq5 = rsquare(x5, y5)
    plotting(
        ax5,
        "",
        x5, poly5(x5),
        x5, y5,
        f"$r^2 = {round(rsq5, 2)}$", "",
        "Actual Cd", "Predicted Cd", "o"
    )

    x6 = aktual["cm"].to_numpy()
    y6 = pred["cm"].to_numpy()

    if y6.size > x6.size:
        y6 = y6[:(x6.size - y6.size)]

    coef6 = np.polyfit(x6, y6, 1)
    poly6 = np.poly1d(coef6)
    rsq6 = rsquare(x6, y6)
    plotting(
        ax6,
        "",
        x6, poly6(x6),
        x6, y6,
        f"$r^2 = {round(rsq6, 2)}$", "",
        "Actual Cm", "Predicted Cm", "o"
    )

    plt.show()
