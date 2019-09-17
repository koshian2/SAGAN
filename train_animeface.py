import torch
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import os
import pickle
import statistics
import glob
import shutil

import losses
import models.resnet_size_96 as resnet96
from inception_score import inceptions_score_all_weights

def load_animeface(batch_size):
    # 前処理
    for dir in sorted(glob.glob("thumb/*")):
        imgs = glob.glob(dir + "/*.png")
        if len(imgs) == 0:
            shutil.rmtree(dir)

    trans = transforms.Compose([
        transforms.Resize(size=(96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(root="./data/thumb", transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

def train(cases):
    # Anime face (96x96)
    assert cases == 0 # 1 case only

    output_dir = f"anime_case{cases}"

    batch_size = 128
    device = "cuda"

    dataloader = load_animeface(batch_size)

    model_G = resnet96.Generator(n_classes_g=176)
    model_D = resnet96.Discriminator(n_classes_d=176)
    model_G, model_D = model_G.to(device), model_D.to(device)

    model_G, model_D = torch.nn.DataParallel(model_G), torch.nn.DataParallel(model_D)

    param_G = torch.optim.Adam(model_G.parameters(), lr=0.00005, betas=(0, 0.9))
    param_D = torch.optim.Adam(model_D.parameters(), lr=0.0002, betas=(0, 0.9))

    gan_loss = losses.HingeLoss(batch_size, device)

    n_epoch = 251

    result = {"d_loss": [], "g_loss": []}
    n = len(dataloader)
    onehot_encoding = torch.eye(176).to(device)

    for epoch in range(n_epoch):
        log_loss_D, log_loss_G = [], []

        for i, (real_img, labels) in tqdm(enumerate(dataloader), total=n):
            batch_len = len(real_img)
            if batch_len != batch_size: continue
            real_img = real_img.to(device)
            real_onehots = onehot_encoding[labels.to(device)] # conditional
            
            # train G
            param_G.zero_grad()
            param_D.zero_grad()

            rand_X = torch.randn(batch_len, 128).to(device)
            fake_onehots = onehot_encoding[torch.randint(0, 176, (batch_size,))].to(device)            
            fake_img = model_G(rand_X, fake_onehots)
            g_out = model_D(fake_img, fake_onehots)

            loss = gan_loss(g_out, "gen")
            log_loss_G.append(loss.item())
            # backprop
            loss.backward()
            param_G.step()

            # train D
            param_G.zero_grad()
            param_D.zero_grad()
            # train real
            d_out_real = model_D(real_img, real_onehots)
            loss_real = gan_loss(d_out_real, "dis_real")
            # train fake
            rand_X = torch.randn(batch_len, 128).to(device)
            fake_onehots = onehot_encoding[torch.randint(0, 176, (batch_len,))].to(device)
            fake_img = model_G(rand_X, fake_onehots).detach() # important not to backprop

            d_out_fake = model_D(fake_img, fake_onehots)
            loss_fake = gan_loss(d_out_fake, "dis_fake")
            loss = loss_real + loss_fake
            log_loss_D.append(loss.item())
            # backprop
            loss.backward()
            param_D.step()

        # ログ
        result["d_loss"].append(statistics.mean(log_loss_D))
        result["g_loss"].append(statistics.mean(log_loss_G))
        print(f"epoch = {epoch}, g_loss = {result['g_loss'][-1]}, d_loss = {result['d_loss'][-1]}")        
            
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        torchvision.utils.save_image(fake_img[:25], f"{output_dir}/epoch_{epoch:03}.png",
                                    nrow=5, padding=5, normalize=True, range=(-1.0, 1.0))

        # 係数保存
        if not os.path.exists(output_dir + "/models"):
            os.mkdir(output_dir+"/models")
        if epoch % 5 == 0:
            torch.save(model_G.state_dict(), f"{output_dir}/models/gen_epoch_{epoch:04}.pytorch")
            torch.save(model_D.state_dict(), f"{output_dir}/models/dis_epoch_{epoch:04}.pytorch")

    # ログ
    with open(output_dir + "/logs.pkl", "wb") as fp:
        pickle.dump(result, fp)
    
if __name__ == "__main__":
    train(0)
