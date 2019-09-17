import torch
import torchvision
import glob
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import models
import models.resnet_size_32, models.resnet_size_48, models.resnet_size_96
import numpy as np
from PIL import Image

def inception_log_graph(base_dir):
    csvs = sorted(glob.glob(base_dir + "_case*/inception_score.csv"))
    datax = []
    datay = []
    cases = []
    model_paths = []
    for c in csvs:
        df = pd.read_csv(c).sort_values(by=["epoch"])
        dir_name = os.path.dirname(c).replace("\\", " / ").split(" / ")[-1]
        regex=re.search('case([0-9]*)', dir_name).group(1)
        cases.append(regex)
        x = df["epoch"].values
        y = df["inception_score"].values
        max_epoch = x[np.argmax(np.array(y))]
        model_paths.append(os.path.dirname(c) + f"/models/gen_epoch_{max_epoch:03}.pytorch")        
        datax.append(x)
        datay.append(y)        
        title=dir_name.replace("_case" + regex, "")

    plt.clf()
    for x, y, c in zip(datax, datay, cases):
        plt.plot(x, y, label="case " + c)
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Inception score")
    plt.savefig("graph/" + title + ".png")

    return model_paths

def sampling_and_interpolation(base_dir):
    experiment_case = base_dir.split("/")[1]
    if experiment_case == "cifar":
        ms = [models.resnet_size_32.Generator(enable_conditional=i in [0, 1, 2, 3, 4, 5],
                                              use_self_attention=i in [2, 5, 6, 7]) for i in range(8)]
        xs = [torch.randn(50, 128) for i in range(8)]
        cs = [torch.eye(10)[torch.arange(10).repeat(5)] if i in [0, 1, 2, 3, 4, 5] else None for i in range(8)]
    elif experiment_case == "cifar_lrscale":
        ms = [models.resnet_size_32.Generator(enable_conditional=True,
                                              use_self_attention=True) for i in range(4)]
        xs = [torch.randn(50, 128) for i in range(4)]
        cs = [torch.eye(10)[torch.arange(10).repeat(5)] for i in range(4)]
    elif experiment_case == "stl":
        ms = [models.resnet_size_48.Generator(enable_conditional=True,
                                              use_self_attention=i == 2) for i in range(3)]
        xs = [torch.randn(50, 128) for i in range(3)]
        cs = [torch.eye(10)[torch.arange(10).repeat(5)] for i in range(3)]
    elif experiment_case == "anime":
        ms = [models.resnet_size_96.Generator(n_classes_g=176) for i in range(1)]
        xs = [torch.randn(50, 128) for i in range(1)]
        cs = [torch.eye(176)[torch.randint(0, 176, (50,))] for i in range(1)]
    elif experiment_case == "flower":
        ms = [models.resnet_size_96.Generator(n_classes_g=102) for i in range(1)]
        xs = [torch.randn(50, 128) for i in range(1)]
        cs = [torch.eye(102)[torch.randint(0, 102, (50,))] for i in range(1)]

    # animefaceの場合、Inception Scoreが意味がないので（ドメイン相違）最後の係数を使う
    # flowerはanimefaceよりInceptionが使えないわけではないが、ISがあまり当てにならない
    if experiment_case == "anime":
        paths = [f"{base_dir}_case{i}/models/gen_epoch_0250.pytorch" for i in range(1)]
    elif experiment_case == "flower":
        paths = [f"{base_dir}_case{i}/models/gen_epoch_0050.pytorch" for i in range(1)]
    else:
        paths = inception_log_graph(base_dir)
        

    for m, x, c, p in zip(ms, xs, cs, paths):
        if experiment_case in ["anime", "flower"]:
            state_dict = torch.load(p)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v # remove module.
            m.load_state_dict(new_state_dict)
        else:
            m.load_state_dict(torch.load(p))

        # sampling
        y = m(x, c) if c is not None else m(x)
        n_row = int(np.rint(np.sqrt(len(x) // 2)))
        sam_img = torchvision.utils.make_grid(y, normalize=True, range=(-1.0, 1.0), nrow=n_row)
        # interpolation
        k = torch.arange(n_row, dtype=torch.float) / (n_row - 1)
        k = k.view(1, n_row, 1)
        n_col = len(x) // n_row
        x1 = x[:n_col].view(n_col, 1, -1)
        x2 = x[n_col:(2 * n_col)].view(n_col, 1, -1)        
        interx = k * x1 + (1 - k) * x2
        interx = interx.view(-1, interx.size(2))
        if c is not None:
            interc = c[:n_col].view(n_col, 1, -1).expand(n_col, n_row, -1).contiguous().view(-1, c.size(1))
            y = m(interx, interc)
        else:
            y = m(interx)
        # "interpolation_" + os.path.dirname(p).replace("\\", "/").split("/")[-2]+".png"
        inter_img = torchvision.utils.make_grid(y, normalize=True, range=(-1.0, 1.0), nrow=n_row)

        # padding
        pad = torch.zeros(sam_img.size(0), sam_img.size(1), 20)

        # concat
        img = torch.cat([sam_img, pad, inter_img], dim=2).permute([1, 2, 0]).detach().numpy()
        img = (img * 255.0).astype(np.uint8)
        with Image.fromarray(img) as img:
            img.save("sampling_interpolation/"+ os.path.dirname(p).replace("\\", "/").split("/")[-2]+".png")        

if __name__ == "__main__":
    sampling_and_interpolation("results/flower")