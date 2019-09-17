import glob
import pandas as pd
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt

def inception_error_log(base_dir):
    dirs = sorted(glob.glob(base_dir + "_case*"))
    for i, d in enumerate(dirs):
        if i == 0: continue
        df = pd.read_csv(d + "/inception_score.csv").sort_values(by=["epoch"])
        dir_name = d.replace("\\", "/").split("/")[-1]
        case_no=re.search('case([0-9]*)', dir_name).group(1)
        with open(d + "/logs.pkl", "rb") as fp:
            logs = pickle.load(fp)
        xlab = np.arange(len(logs["d_loss"]))
        plt.plot(xlab[::5], df["inception_score"].values, label="Incetpion Score")
        plt.plot(xlab, logs["d_loss"], label="D loss")
        plt.plot(xlab, logs["g_loss"], label="G loss")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    inception_error_log("results/cifar")