import torch

class HingeLoss():
    def __init__(self, batch_size, device, precision="float"):
        self.ones = torch.ones(batch_size, 1)
        self.zeros = torch.zeros(batch_size, 1)
        if precision == "half":
            self.ones = self.ones.half()
            self.zeros = self.zeros.half()
        self.ones, self.zeros = self.ones.to(device), self.zeros.to(device)  

    def __call__(self, logits, condition):
        assert condition in ["gen", "dis_real", "dis_fake"]
        batch_len = len(logits)
        if condition == "gen":
            return -torch.mean(logits)
        elif condition == "dis_real":
            minval = torch.min(logits - 1, self.zeros[:batch_len])
            return -torch.mean(minval)
        else:
            minval = torch.min(-logits - 1, self.zeros[:batch_len])
            return - torch.mean(minval)

class DCGANCrossEntropy():
    def __init__(self, batch_size, device, precision="float"):
        self.ones = torch.ones(batch_size, 1).to(device)
        self.zeros = torch.zeros(batch_size, 1).to(device)
        if precision == "half":
            self.ones = self.ones.half()
            self.zeros = self.zeros.half()       
        self.loss_func = torch.nn.BCEWithLogitsLoss()

    def __call__(self, logits, condition):
        assert condition in ["gen", "dis_real", "dis_fake"]
        batch_len = len(logits)
        if condition == "gen" or condition == "dis_real":
            return self.loss_func(logits, self.ones[:batch_len])
        else:
            return self.loss_func(logits, self.zeros[:batch_len])
