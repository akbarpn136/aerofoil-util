import glob
import numpy as np
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
            names=["alpha", "cl", "cd", "cdp", "cm",
                   "TopXtr", "BotXtr", "Cpmin", "Chinge", "XCp"]
        )

        name = aero.replace("aero\\", "").replace(
            "_T1_Re0.500_M0.00_N5.0.txt", "")
        print(f"{name}: {ddf.alpha.count()}")


def show_train():
    df = pd.read_csv("training.csv", delimiter="\t")

    mpl.use('pgf')
    mpl.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'font.family': 'serif',
        'font.size': 25,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(df["train_loss_2"].to_numpy(),
            color="red", label="TrainLoss_Aerofoil2BN2FC")
    ax.plot(df["valid_loss_2"].to_numpy(), ":",
            color="red", label="ValidLoss_Aerofoil2BN2FC")

    ax.plot(df["train_loss_3"].to_numpy(),
            color="green", label="TrainLoss_Aerofoil3BN2FC")
    ax.plot(df["valid_loss_3"].to_numpy(), ":",
            color="green", label="ValidLoss_Aerofoil3BN2FC")

    ax.plot(df["train_loss_4"].to_numpy(),
            color="blue", label="TrainLoss_Aerofoil4BN2FC")
    ax.plot(df["valid_loss_4"].to_numpy(), ":",
            color="blue", label="ValidLoss_Aerofoil4BN2FC")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(linestyle='--')
    fig.savefig(f'training.pgf')
    plt.close(fig)


def show_vartrain():
    bin = pd.read_csv("training_binary_4Conv_BN_2FC.csv")
    um = pd.read_csv("training_mesh_4Conv_BN_2FC.csv")
    sdf = pd.read_csv("training_sdf_4Conv_BN_2FC.csv")

    mpl.use('pgf')
    mpl.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'font.family': 'serif',
        'font.size': 25,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(bin["train_loss"].to_numpy(),
            color="red", label="TrainLoss_Gray")
    ax.plot(bin["valid_loss"].to_numpy(), ":",
            color="red", label="ValidLoss_Gray")

    ax.plot(um["train_loss"].to_numpy(),
            color="green", label="TrainLoss_UM")
    ax.plot(um["valid_loss"].to_numpy(), ":",
            color="green", label="ValidLoss_UM")

    ax.plot(sdf["train_loss"].to_numpy(),
            color="blue", label="TrainLoss_SDF")
    ax.plot(sdf["valid_loss"].to_numpy(), ":",
            color="blue", label="ValidLoss_SDF")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(linestyle='--')
    fig.savefig(f'vartraining.pgf')
    plt.close(fig)


def plot_prediction(mode="cl"):
    df = pd.read_csv("actual.csv")
    bin = pd.read_csv("prediction_binary.csv")
    um = pd.read_csv("prediction_mesh.csv")
    sdf = pd.read_csv("prediction_sdf.csv")

    mpl.use('pgf')
    mpl.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'font.family': 'serif',
        'font.size': 25,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.plot(
        df["alpha"].to_numpy(),
        df[f"{mode}"].to_numpy(),
        color="k",
        label="actual"
    )

    ax.plot(
        bin["alpha"].to_numpy(),
        bin[f"{mode}"].to_numpy(),
        marker="x",
        linestyle="None",
        color="red",
        label="Gray"
    )

    ax.plot(
        um["alpha"].to_numpy(),
        um[f"{mode}"].to_numpy(),
        marker="+",
        linestyle="None",
        color="blue",
        label="UM"
    )

    ax.plot(
        sdf["alpha"].to_numpy(),
        sdf[f"{mode}"].to_numpy(),
        marker=".",
        linestyle="None",
        color="green",
        label="SDF"
    )

    ax.set_xlabel(r"$\alpha$")
    if mode == 'cl':
        ax.set_ylabel(r'$c_l$')
    elif mode == 'cd':
        ax.set_ylabel(r'$c_d$')
    else:
        ax.set_ylabel(r'$c_m$')
    ax.legend()
    ax.grid(linestyle='--')
    fig.savefig(f'pred_{mode}.pgf')
    plt.close(fig)


def plot_score(mode='cl', render='Grayscale'):
    df = pd.read_csv("actual.csv")
    bin = pd.read_csv("prediction_binary.csv")
    um = pd.read_csv("prediction_mesh.csv")
    sdf = pd.read_csv("prediction_sdf.csv")

    mpl.use('pgf')
    mpl.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'font.family': 'serif',
        'font.size': 25,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    fig, ax = plt.subplots(figsize=(10, 10))

    actual = df[mode].to_numpy()

    if render == 'Grayscale':
        predicted = bin[mode].to_numpy()
    elif render == 'UM':
        predicted = um[mode].to_numpy()
    else:
        predicted = sdf[mode].to_numpy()

    coef = np.polyfit(actual, predicted, 1)
    polyy = np.poly1d(coef)
    ax.set_title(f'Score using {render}')
    ax.plot(
        actual,
        predicted,
        marker=".",
        color="green",
        linestyle="None"
    )

    ax.plot(
        actual,
        polyy(actual),
        color="k"
    )

    if mode == 'cl':
        ax.set_xlabel(r"Actual $C_l$")
        ax.set_ylabel(r"Predicted $C_l$")
    elif mode == 'cd':
        ax.set_xlabel(r"Actual $C_d$")
        ax.set_ylabel(r"Predicted $C_d$")
    else:
        ax.set_xlabel(r"Actual $C_m$")
        ax.set_ylabel(r"Predicted $C_m$")

    ax.grid(linestyle='--')
    fig.savefig(f'score_{mode}_{render}.pgf')
    plt.close(fig)


if __name__ == "__main__":
    plot_score(mode='cm', render='SDF')
