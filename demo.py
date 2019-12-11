import torch

if __name__ == "__main__":
    filename_scripted = 'retinaface_scripted.pt'
    model = torch.jit.load(filename_scripted)
