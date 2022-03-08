from io import BytesIO

import axi
import click
import numpy as np
import requests
from PIL import ImageEnhance, Image
from axi import Drawing
from matplotlib import pyplot as plt

jarvis = np.array([[0, 0, 0, 7, 5], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]]) / 48


def snap(val: float, levels: np.ndarray) -> tuple[float, float, int]:
    diff = np.abs(levels - val)
    idx = np.argmin(diff)
    snapped_val = levels[idx]
    return snapped_val, val - snapped_val, idx


def set_ratio(pil_img, ratio):
    width, height = pil_img.size
    current_ratio = width / height
    if current_ratio < ratio:  # Needs to be wider
        new_size = (round(height * ratio), height)
    else:
        new_size = (width, round(width / ratio))
    result = Image.new(pil_img.mode, new_size, 255)
    result.paste(pil_img, (0, 0))
    return result


def dither(
    img: np.ndarray, kernel: np.ndarray, levels: np.ndarray, original_size: tuple
) -> tuple[np.ndarray, np.ndarray]:
    working_img = np.copy(img).astype("float64")
    out = np.ones(img.shape) * levels.size
    for r in range(original_size[1]):
        for c in range(original_size[0]):
            new_val, error, idx = snap(working_img[r, c], levels)
            out[r, c] = idx
            working_img[r, c] = new_val
            diffuse = error * kernel
            r_upper_bound = min(original_size[1] - r - 1, kernel.shape[0])
            c_lower_bound = min(c, kernel.shape[1] // 2)
            c_upper_bound = min(original_size[0] - c - 1, kernel.shape[1] // 2 + 1)
            working_img[
                r : r + r_upper_bound, c - c_lower_bound : c + c_upper_bound
            ] += diffuse[0:r_upper_bound, 2 - c_lower_bound : 2 + c_upper_bound]
    return out, working_img


def hatch(
    img: Image,
    brightness: float,
    contrast: float,
    gap: float,
    res: float,
) -> Drawing:
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)
    original_size = img.size
    width, height = img.width, img.height
    img = np.asarray(img)
    master_mask, _ = dither(
        img,
        jarvis,
        np.array(
            [
                1 * 255 // 9,
                2 * 255 // 9,
                3 * 255 // 9,
                4 * 255 // 9,
                5 * 255 // 9,
                6 * 255 // 9,
                7 * 255 // 9,
                8 * 255 // 9,
                255,
            ]
        ),
        original_size,
    )
    vert_mask_0 = master_mask <= 0
    vert_mask_1 = master_mask <= 1
    horiz_mask_0 = master_mask <= 2
    horiz_mask_1 = master_mask <= 3
    diag_a_mask_0 = master_mask <= 4
    diag_a_mask_1 = master_mask <= 5
    diag_b_mask_0 = master_mask <= 6
    diag_b_mask_1 = master_mask <= 7
    sqrt_2 = np.sqrt(2)
    paths = []
    # vertical lines
    for i, x in enumerate(np.arange(0, width, gap)):
        mask = vert_mask_0 if i % 2 == 0 else vert_mask_1
        path = []
        for y in np.arange(0, height, res):
            if mask[int(y), int(x)]:
                path.append((x, y))
            else:
                if len(path) >= 2:
                    paths.append(path)
                path = []
        if len(path) >= 2:
            paths.append(path)
    # horizontal lines
    for i, y in enumerate(np.arange(0, height, gap)):
        mask = horiz_mask_0 if i % 2 == 0 else horiz_mask_1
        path = []
        for x in np.arange(0, width, res):
            if mask[int(y), int(x)]:
                path.append((x, y))
            else:
                if len(path) >= 2:
                    paths.append(path)
                path = []
        if len(path) >= 2:
            paths.append(path)
    # # UL-DR diagonal lines
    for i, y in enumerate(np.arange(-width, height, gap * sqrt_2)):
        mask = diag_a_mask_0 if i % 2 == 0 else diag_a_mask_1
        path = []
        for x in np.arange(0, width, res / sqrt_2):
            if not 0 <= y + x < height:
                continue
            if mask[int(y + x), int(x)]:
                path.append((x, y + x))
            else:
                if len(path) >= 2:
                    paths.append(path)
                path = []
        if len(path) >= 2:
            paths.append(path)
    # # UR-DL diagonal lines
    for i, y in enumerate(np.arange(-width, height, gap * sqrt_2)):
        mask = diag_b_mask_0 if i % 2 == 0 else diag_b_mask_1
        path = []
        for x in np.arange(0, width, res / sqrt_2):
            if not 0 <= y + (width - x) < height:
                continue
            if mask[int(y + (width - x)), int(x)]:
                path.append((x, y + (width - x)))
            else:
                if len(path) >= 2:
                    paths.append(path)
                path = []
        if len(path) >= 2:
            paths.append(path)
    return Drawing(paths)


@click.command()
@click.option("-t", "--test", is_flag=True)
@click.option("-w", "--width", prompt=True, type=float)
@click.option("-h", "--height", prompt=True, type=float)
@click.option("-m", "--margin", prompt=True, type=float)
@click.option("-g", "--line_gap", prompt=True, type=float, default=5)
@click.option("-r", "--line_res", prompt=True, type=float, default=2)
@click.option("-b", "--brightness", prompt=True, type=float, default=1.1)
@click.option("-c", "--contrast", prompt=True, type=float, default=1.1)
def main(
    test: bool,
    width: float,
    height: float,
    margin: float,
    line_gap: float,
    line_res: float,
    brightness: float,
    contrast: float,
):
    img = Image.open('family.jpg').convert("L")
    drawing = hatch(img, brightness, contrast, line_gap, line_res)
    drawing = drawing.scale_to_fit(width, height, margin).center(width, height)
    if test or axi.device.find_port() is None:
        im = drawing.render(bounds=(0, 0, width, height))
        im.write_to_png("hatch.png")
    else:
        axi.draw(drawing)


if __name__ == "__main__":
    main()
