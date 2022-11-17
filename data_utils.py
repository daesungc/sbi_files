from glob import glob

import pandas as pd


def concat_dfs(search_str, savename):
    dat_files = glob(search_str)
    dfs = [pd.read_csv(f) for f in dat_files]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(savename, index=False)
