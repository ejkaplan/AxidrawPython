import axi
import axi.device
import math
from PIL import Image
import requests
import numpy as np
from matplotlib import pyplot as plt

FLOYD_STEINBERG = (np.array([[0, 0, 7], [3, 5, 1]]),)
JARVIS = (np.array([[0, 0, 0, 7, 5], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]]),)
STUCKI = (np.array([[0, 0, 0, 8, 4], [2, 4, 8, 4, 2], [1, 2, 4, 2, 1]]),)
SIERRA = (np.array([[0, 0, 0, 5, 3], [2, 4, 5, 4, 2], [0, 2, 3, 2, 0]]),)
ATKINSON = (np.array([[0, 0, 0, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]]), 8)
BURKES = (np.array([[0, 0, 0, 8, 4], [2, 4, 8, 4, 2]]),)


def dither(pixels, diffusion, divisor=None):
    if divisor is None:
        divisor = diffusion.sum()
    diffusion = diffusion / divisor
    out = np.copy(pixels)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            old_val = out[y, x]
            new_val = 255 if old_val > 255 / 2 else 0
            error = old_val - new_val
            out[y, x] = new_val
            local_diff = diffusion * error
            if x - diffusion.shape[1] // 2 < 0:
                local_diff = local_diff[:, diffusion.shape[1] // 2 - x :]
            elif x + diffusion.shape[1] // 2 >= out.shape[1]:
                local_diff = local_diff[
                    :, : -(diffusion.shape[1] // 2 - (out.shape[1] - x - 1))
                ]
            if y + diffusion.shape[0] - 1 >= out.shape[0]:
                local_diff = local_diff[: -(diffusion.shape[0] - (out.shape[0] - y)), :]
            out_slice = out[
                y : y + local_diff.shape[0],
                max(0, x - diffusion.shape[1] // 2) : min(
                    out.shape[1], x + diffusion.shape[1] // 2 + 1
                ),
            ]
            out_slice += local_diff
    return out


def resize(img, width=0, height=0):
    if width == height == 0:
        raise ValueError("At least one of width or height must be provided")
    if width == 0:
        width = round(img.width * (height / img.height))
    elif height == 0:
        height = round(img.height * (width / img.width))
    return img.resize(size=(width, height), resample=Image.NEAREST)


def main():
    axi.device.MAX_VELOCITY = 2
    url = (
        "https://pbs.twimg.com/profile_images/1309133913953099776/PEgTVuQB_400x400.jpg"
    )
    img = Image.open(requests.get(url, stream=True).raw).convert("L")
    img = resize(img, width=100)
    pixels = np.asarray(img, dtype="float64")
    dithered = dither(pixels, *ATKINSON)
    out_img = Image.fromarray(np.uint8(dithered))
    out_img = resize(out_img, width=600)
    out_img.save("dither.png")


if __name__ == "__main__":
    main()
