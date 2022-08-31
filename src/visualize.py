import glob
import torch
from PIL import Image
from scipy import ndimage

from src.services.arch.conv3 import Aerofoil3BN2FC


def VisualizeTensor(tensor, idx, ch=0, allkernels=False,):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    x = tensor[idx].numpy().transpose((1, 2, 0))

    return x


if __name__ == "__main__":
    airfoilname = "Aerofoil5000"
    kind = "sdf"
    num_channel = 3
    all_files = glob.glob(f"out/{airfoilname}_{kind}*.jpg")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Aerofoil3BN2FC(num_channel=num_channel).to(dev)
    model.load_state_dict(
        torch.load("aerofoil_stack_Aerofoil3BN2FC_aug.pt", map_location=dev)
    )
    model.eval()

    layer = 0
    # Option based on Aerofoil3BN2FC: conv1, conv2, and conv3
    filter = model.conv3[layer].weight.data.cpu()

    for i in range(filter.shape[0]):
        x = VisualizeTensor(filter, i)
        img = Image.open(all_files[0])
        out = ndimage.convolve(img, x)
        out = Image.fromarray(out, "RGB").convert("L")
        out.save(f"./result/{100 + i}.jpg")
