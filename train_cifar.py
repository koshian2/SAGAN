import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import os
import pickle
import statistics
import glob

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
    # うまくいかないケース
    # lr=0.0002, n_dis=5, beta1~0

    ## cifar-10
    # case 0
    # conditional, ttur:False, attention:False, hinge_loss
    # case 1
    # conditional, ttur:True,  attention:False, hinge_loss
    # case 2
    # conditional, ttur:True,  attention:True,  hinge_loss

    output_dir = f"cifar_case{cases}"

    batch_size = 64
    device = "cuda"
    
    torch.backends.cudnn.benchmark = True

    enable_conditional = cases in [0, 1, 2]
    use_ttur = cases in [1, 2]
    use_attention = cases in [2]
    gan_loss = losses.HingeLoss(batch_size, device) if cases in [0, 1, 2] else losses.DCGANCrossEntropy(batch_size, device)
    
    print("--- Conditions ---")
    print("- Case : ", cases)
    print("conditional :", enable_conditional, ", ttur :", use_ttur,
          ", attention :", use_attention, ", loss :", gan_loss)
    
    dataloader = load_cifar(batch_size)

    model_G = cifar_resnet.Generator(enable_conditional=enable_conditional, use_self_attention=use_attention)
    model_D = cifar_resnet.Discriminator(enable_conditional=enable_conditional, use_self_attention=use_attention)
    model_G, model_D = model_G.to(device), model_D.to(device)

    param_G = torch.optim.Adam(model_G.parameters(), lr=0.0001 if use_ttur else 0.0002, betas=(0, 0.9))
    param_D = torch.optim.Adam(model_D.parameters(), lr=0.0004 if use_ttur else 0.0002, betas=(0, 0.9))

    result = {"d_loss": [], "g_loss": []}
    n = len(dataloader)
    onehot_encoding = torch.eye(10).to(device)

    for epoch in range(301):
        log_loss_D, log_loss_G = [], []

        for i, (real_img, labels) in tqdm(enumerate(dataloader), total=n):
            batch_len = len(real_img)
            if batch_len != batch_size: continue

            real_img = real_img.to(device)
            if enable_conditional:
                real_onehots = onehot_encoding[labels.to(device)]  # conditional
            else:
                real_onehots = None  # non conditional
                        
            # G -> Dでfake_imgを使いまわしすると、DのタイミングでGがアップデートされており、分布のミスマッチを起こすが
            # D -> Gでfake_imgを使いまわしても、GのタイミングでGはアップデートされておらず、分布のミスマッチを起こさない
            # もっと言えば、サンプリングするとき、onehotをrealと変えると分布の偏りを起こさないかも（？）

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
                                    nrow=8, padding=2, normalize=True, range=(-1.0, 1.0))

        # 係数保存
        if not os.path.exists(output_dir + "/models"):
            os.mkdir(output_dir+"/models")
        if epoch % 5 == 0:
            torch.save(model_G.state_dict(), f"{output_dir}/models/gen_epoch_{epoch:03}.pytorch")
            torch.save(model_D.state_dict(), f"{output_dir}/models/dis_epoch_{epoch:03}.pytorch")

    # ログ
    with open(output_dir + "/logs.pkl", "wb") as fp:
        pickle.dump(result, fp)

if __name__ == "__main__":
    train(1)
