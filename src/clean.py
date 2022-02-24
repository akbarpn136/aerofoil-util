import itertools
import os
import shutil
import pandas as pd
from mpire import WorkerPool


def _rename(*payload):
    pth = payload[0]
    img = payload[1]

    try:
        os.rename(f"{pth}/{img}", f"tmp/{img}")
    except FileNotFoundError:
        pass

    try:
        shutil.rmtree("out")
        os.rename("tmp", f"{pth}")
    except FileNotFoundError:
        pass
    except OSError:
        pass


if __name__ == "__main__":
    filename = "../out.csv"
    df = pd.read_csv(f"{filename}")
    images = df["img"].tolist()
    path = "../out"
    isExist = os.path.exists("tmp")
    paramlist = list(itertools.product([path], images))

    if not isExist:
        os.makedirs("tmp")

    print("Clean unused airfoil images")
    with WorkerPool(n_jobs=os.cpu_count()) as pool:
        pool.map(_rename, paramlist, progress_bar=True)
