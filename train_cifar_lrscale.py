import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import os
import pickle
import statistics
import glob
import numpy as np

import losses
import models.resnet_size_32 as cifar_resnet
from inception_score import inceptions_score_all_weights

def load_cifar(batch_size):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True,
                transform=trans, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                shuffle=True, num_workers=4)
    return dataloader

def train(cases):
    ## cifar-10
    # すべてのケースでConditional, TTUR, Attentionを使いHinge lossとする
    # case 0
    # batch_size = 64, lr /= 4.0
    # case 1
    # batch_size = 128, lr /= 2.0
    # case 2
    # batch_size = 64, lr = same
    # case 3
    # batch_size = 128, lr = same

    output_dir = f"cifar_lrscale_case{cases}"

    if cases in [0, 2]: batch_size = 64
    elif cases in [1, 3]: batch_size = 128

    lr_scale = 1.0
    if cases == 0: lr_scale = 0.25
    elif cases == 1: lr_scale = 0.5

    device = "cuda"
    
    torch.backends.cudnn.benchmark = True

    gan_loss = losses.HingeLoss(batch_size, device)

    nb_epoch = 101
    
    print("--- Conditions ---")
    print("- Case : ", cases)
    print("batch_size :", batch_size, ", lr_scale :", lr_scale)
    
    dataloader = load_cifar(batch_size)

    model_G = cifar_resnet.Generator(enable_conditional=True, use_self_attention=True)
    model_D = cifar_resnet.Discriminator(enable_conditional=True, use_self_attention=True)
    model_G, model_D = model_G.to(device), model_D.to(device)

    param_G = torch.optim.Adam(model_G.parameters(), lr=0.0001 * lr_scale, betas=(0, 0.9))
    param_D = torch.optim.Adam(model_D.parameters(), lr=0.0004 * lr_scale, betas=(0, 0.9))    

    result = {"d_loss": [], "g_loss": []}
    n = len(dataloader)
    onehot_encoding = torch.eye(10).to(device)

    for epoch in range(nb_epoch):
        log_loss_D, log_loss_G = [], []

        for i, (real_img, labels) in tqdm(enumerate(dataloader), total=n):
            batch_len = len(real_img)
            if batch_len != batch_size: continue

            real_img = real_img.to(device)
            real_onehots = onehot_encoding[labels.to(device)]  # conditional
                        
            # train G
            param_G.zero_grad()
            param_D.zero_grad()

            rand_X = torch.randn(batch_len, 128).to(device)
            fake_onehots = torch.eye(10)[torch.randint(0, 10, (batch_len,))].to(device)
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
            fake_onehots = torch.eye(10)[torch.randint(0, 10, (batch_len,))].to(device)            
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
        torchvision.utils.save_image(fake_img, f"{output_dir}/epoch_{epoch:03}.png",
                                    nrow=16, padding=2, normalize=True, range=(-1.0, 1.0))

        # 係数保存
        if not os.path.exists(output_dir + "/models"):
            os.mkdir(output_dir+"/models")
        if epoch % 5 == 0:
            torch.save(model_G.state_dict(), f"{output_dir}/models/gen_epoch_{epoch:03}.pytorch")
            torch.save(model_D.state_dict(), f"{output_dir}/models/dis_epoch_{epoch:03}.pytorch")

    # ログ
    with open(output_dir + "/logs.pkl", "wb") as fp:
        pickle.dump(result, fp)

def evaluate(cases):
    inceptions_score_all_weights("cifar_lrscale_case" + str(cases), cifar_resnet.Generator,
                                100, 100, n_classes=10,
                                enable_conditional=True, use_self_attention=True)

if __name__ == "__main__":
    for i in range(4):
        evaluate(i)
