import glob
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt


def count_data():
    path = "aero"
    aeros = glob.glob(f"{path}/*.txt")

    for aero in aeros:
        ddf = pd.read_fwf(
            aero,
            skiprows=11,
            header=None,
            names=["alpha", "cl", "cd", "cdp", "cm", "TopXtr", "BotXtr", "Cpmin", "Chinge", "XCp"]
        )

        name = aero.replace("aero\\", "").replace("_T1_Re0.500_M0.00_N5.0.txt", "")
        print(f"{name}: {ddf.alpha.count()}")


def show_train():
    df = pd.read_csv("training.csv", delimiter="\t")

    plt.style.use("bmh")
    mpl.rcParams['lines.linewidth'] = 1
    fig, ax= plt.subplots()

    ax.plot(df["train_loss_bin"].to_numpy(), color="red", label="TrainLossGray")
    ax.plot(df["valid_loss_bin"].to_numpy(), ":", color="red", label="ValidLossGray")

    ax.plot(df["train_loss_un"].to_numpy(), color="green", label="TrainLossUnstruct")
    ax.plot(df["valid_loss_un"].to_numpy(), ":", color="green", label="ValidLossUnstruct")

    ax.plot(df["train_loss_sdf"].to_numpy(), color="blue", label="TrainLossSDF")
    ax.plot(df["valid_loss_sdf"].to_numpy(), ":", color="blue", label="ValidLossSDF")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.legend()
    plt.show()
    plt.close(fig)

def plot_prediction(mode="cl"):
    df = pd.read_csv("prediction.csv", delimiter="\t")

    plt.style.use("bmh")
    mpl.rcParams['lines.linewidth'] = 1
    fig, ax= plt.subplots()

    ax.plot(df["alpha"].to_numpy(), df[f"{mode}_actual"].to_numpy(), color="k", label="actual")
    ax.plot(df["alpha"].to_numpy(), df[f"{mode}_bin"].to_numpy(), marker="x", linestyle="None", color="red", label="gray")
    ax.plot(df["alpha"].to_numpy(), df[f"{mode}_un"].to_numpy(), marker="+", linestyle="None", color="blue", label="unstructured")
    ax.plot(df["alpha"].to_numpy(), df[f"{mode}_sdf"].to_numpy(), marker=".", linestyle="None", color="green", label="sdf")

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(mode)
    ax.legend()
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    plot_prediction("cm")
