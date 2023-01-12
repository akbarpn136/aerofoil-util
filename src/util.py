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
    df = pd.read_csv("prediction.csv", delimiter="\t")

    plt.style.use("bmh")
    mpl.rcParams['lines.linewidth'] = 1
    fig, ax = plt.subplots()

    ax.plot(df["alpha"].to_numpy(),
            df[f"{mode}_actual"].to_numpy(), color="k", label="actual")
    ax.plot(df["alpha"].to_numpy(), df[f"{mode}_bin"].to_numpy(
    ), marker="x", linestyle="None", color="red", label="gray")
    ax.plot(df["alpha"].to_numpy(), df[f"{mode}_un"].to_numpy(
    ), marker="+", linestyle="None", color="blue", label="unstructured")
    ax.plot(df["alpha"].to_numpy(), df[f"{mode}_sdf"].to_numpy(
    ), marker=".", linestyle="None", color="green", label="sdf")

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(mode)
    ax.legend()
    plt.show()
    plt.close(fig)


def plot_score():
    df = pd.read_csv("prediction.csv", delimiter="\t")

    plt.style.use("bmh")
    mpl.rcParams['lines.linewidth'] = 1

    fig, ((ax1, ax2, ax3),
          (ax4, ax5, ax6),
          (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(10, 10))

    # First row
    clactual_bin = df["cl_actual"].to_numpy()
    clpredicted_bin = df[f"cl_bin"].to_numpy()
    coef_bin1 = np.polyfit(clactual_bin, clpredicted_bin, 1)
    poly_bin1 = np.poly1d(coef_bin1)
    ax1.set_title("Grayscale", fontsize=12)
    ax1.plot(clactual_bin, clpredicted_bin, marker=".",
             color="green", linestyle="None")
    ax1.plot(clactual_bin, poly_bin1(clactual_bin), color="k")
    ax1.set_xlabel(r"Actual $C_l$", fontsize=9)
    ax1.set_ylabel(r"Predicted $C_l$", fontsize=9)

    cdactual_bin = df["cd_actual"].to_numpy()
    cdpredicted_bin = df[f"cd_bin"].to_numpy()
    coef_bin2 = np.polyfit(cdactual_bin, cdpredicted_bin, 1)
    poly_bin2 = np.poly1d(coef_bin2)
    ax2.set_title("Grayscale", fontsize=12)
    ax2.plot(cdactual_bin, cdpredicted_bin, marker=".",
             color="green", linestyle="None")
    ax2.plot(cdactual_bin, poly_bin2(cdactual_bin), color="k")
    ax2.set_xlabel(r"Actual $C_d$", fontsize=9)
    ax2.set_ylabel(r"Predicted $C_d$", fontsize=9)

    cmactual_bin = df["cm_actual"].to_numpy()
    cmpredicted_bin = df[f"cm_bin"].to_numpy()
    coef_bin3 = np.polyfit(cmactual_bin, cmpredicted_bin, 1)
    poly_bin3 = np.poly1d(coef_bin3)
    ax3.set_title("Grayscale", fontsize=12)
    ax3.plot(cmactual_bin, cmpredicted_bin, marker=".",
             color="green", linestyle="None")
    ax3.plot(cmactual_bin, poly_bin3(cmactual_bin), color="k")
    ax3.set_xlabel(r"Actual $C_m$", fontsize=9)
    ax3.set_ylabel(r"Predicted $C_m$", fontsize=9)

    # Second row
    clactual_um = df["cl_actual"].to_numpy()
    clpredicted_um = df[f"cl_um"].to_numpy()
    coef_um1 = np.polyfit(clactual_um, clpredicted_um, 1)
    poly_um1 = np.poly1d(coef_um1)
    ax4.set_title("Unstructured Mesh", fontsize=12)
    ax4.plot(clactual_um, clpredicted_um, marker=".",
             color="green", linestyle="None")
    ax4.plot(clactual_um, poly_um1(clactual_um), color="k")
    ax4.set_xlabel(r"Actual $C_l$", fontsize=9)
    ax4.set_ylabel(r"Predicted $C_l$", fontsize=9)

    cdactual_um = df["cd_actual"].to_numpy()
    cdpredicted_um = df[f"cd_um"].to_numpy()
    coef_um2 = np.polyfit(cdactual_um, cdpredicted_um, 1)
    poly_um2 = np.poly1d(coef_um2)
    ax5.set_title("Unstructured Mesh", fontsize=12)
    ax5.plot(cdactual_um, cdpredicted_um, marker=".",
             color="green", linestyle="None")
    ax5.plot(cdactual_um, poly_um2(cdactual_um), color="k")
    ax5.set_xlabel(r"Actual $C_d$", fontsize=9)
    ax5.set_ylabel(r"Predicted $C_d$", fontsize=9)

    cmactual_um = df["cm_actual"].to_numpy()
    cmpredicted_um = df[f"cm_um"].to_numpy()
    coef_um3 = np.polyfit(cmactual_um, cmpredicted_um, 1)
    poly_um3 = np.poly1d(coef_um3)
    ax6.set_title("Unstructured Mesh", fontsize=12)
    ax6.plot(cmactual_um, cmpredicted_um, marker=".",
             color="green", linestyle="None")
    ax6.plot(cmactual_um, poly_um3(cmactual_um), color="k")
    ax6.set_xlabel(r"Actual $C_m$", fontsize=9)
    ax6.set_ylabel(r"Predicted $C_m$", fontsize=9)

    # Third row
    clactual_sdf = df["cl_actual"].to_numpy()
    clpredicted_sdf = df[f"cl_sdf"].to_numpy()
    coef_sdf1 = np.polyfit(clactual_sdf, clpredicted_sdf, 1)
    poly_sdf1 = np.poly1d(coef_sdf1)
    ax7.set_title("SDF", fontsize=12)
    ax7.plot(clactual_sdf, clpredicted_sdf, marker=".",
             color="green", linestyle="None")
    ax7.plot(clactual_sdf, poly_sdf1(clactual_sdf), color="k")
    ax7.set_xlabel(r"Actual $C_l$", fontsize=9)
    ax7.set_ylabel(r"Predicted $C_l$", fontsize=9)

    cdactual_sdf = df["cd_actual"].to_numpy()
    cdpredicted_sdf = df[f"cd_sdf"].to_numpy()
    coef_sdf2 = np.polyfit(cdactual_sdf, cdpredicted_sdf, 1)
    poly_sdf2 = np.poly1d(coef_sdf2)
    ax8.set_title("SDF", fontsize=12)
    ax8.plot(cdactual_sdf, cdpredicted_sdf, marker=".",
             color="green", linestyle="None")
    ax8.plot(cdactual_sdf, poly_sdf2(cdactual_sdf), color="k")
    ax8.set_xlabel(r"Actual $C_d$", fontsize=9)
    ax8.set_ylabel(r"Predicted $C_d$", fontsize=9)

    cmactual_sdf = df["cm_actual"].to_numpy()
    cmpredicted_sdf = df[f"cm_sdf"].to_numpy()
    coef_sdf3 = np.polyfit(cmactual_sdf, cmpredicted_sdf, 1)
    poly_sdf3 = np.poly1d(coef_sdf3)
    ax9.set_title("SDF", fontsize=12)
    ax9.plot(cmactual_sdf, cmpredicted_sdf, marker=".",
             color="green", linestyle="None")
    ax9.plot(cmactual_sdf, poly_sdf3(cmactual_sdf), color="k")
    ax9.set_xlabel(r"Actual $C_m$", fontsize=9)
    ax9.set_ylabel(r"Predicted $C_m$", fontsize=9)

    plt.subplots_adjust(
        left=0.1,
        bottom=0.1,
        right=0.9,
        top=0.9,
        wspace=0.4,
        hspace=0.5
    )
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    show_vartrain()
