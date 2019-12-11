import torch

if __name__ == "__main__":
    filename_scripted = 'retinaface_scripted.pt'
    net = torch.jit.load(filename_scripted)
