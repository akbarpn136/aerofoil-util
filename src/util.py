import glob
import pandas as pd


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


if __name__ == "__main__":
    count_data()
