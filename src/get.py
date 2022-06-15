import os
import glob
import itertools
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def _process(payload):
    pth = payload[0]
    fname = payload[1]
    knd = payload[2]
    file = fname.replace(pth, "") \
        .replace("\\", "") \
        .replace("/", "")

    split = file.split("_")
    dat_name = split[0]
    name = dat_name.replace(" ", "")
    re = split[2].replace("Re", "")
    re = int(float(re) * 10 ** 6)
    ma = split[3].replace("M", "")
    ma = float(ma)
    image = f"{name}_{knd}_{re}_{ma}"

    ddf = pd.read_csv(
        fname,
        skiprows=11,
        header=None,
        names=["alpha", "cl", "cd", "cdp", "cm", "TopXtr", "BotXtr", "Cpmin", "Chinge", "XCp"]
    )

    ddf = ddf[["alpha", "cl", "cd", "cm"]]
    ddf["name"] = dat_name
    ddf["re"] = re
    ddf["ma"] = ma
    ddf["img"] = ddf.apply(lambda row: f"{image}_{int(row['alpha'])}.jpg", axis=1)

    return ddf


if __name__ == "__main__":
    kind = "stack"
    path = "aero"
    filename = "out.csv"
    aeros = glob.glob(f"{path}/*.csv")
    paramlist = list(itertools.product([path], aeros, [kind]))

    with Pool(processes=cpu_count()) as pool:
        results = [x for x in tqdm(pool.imap(_process, paramlist),
                                   total=len(aeros))]

        dt = pd.concat(results, axis=0)
        files_present = glob.glob(filename)

        if len(files_present) != 0:
            os.remove(filename)

        dt.to_csv(filename, index=False)
