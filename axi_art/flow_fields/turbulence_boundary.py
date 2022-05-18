import axi
import click
import numpy as np
from PIL import Image, ImageFilter
from axi import Drawing

from axi_art.flow_fields.flow_fields import (
    curl_noise,
    derive_grid_shape,
    blend_vector_fields,
    render_flow_field,
)


def logistic(x: float, midpoint: float, steepness: float) -> float:
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))


def resize_and_center_image(
    grid_shape: tuple[int, int], img: Image.Image, shrink_factor: float, blur: float
) -> np.ndarray:
    img = img.convert("L")
    scale = shrink_factor * min(grid_shape[0] / img.width, grid_shape[1] / img.height)
    img = img.resize((int(scale * img.width), int(scale * img.height)))
    img = img.filter(ImageFilter.GaussianBlur(blur))
    img = np.asarray(img)
    out = 255 * np.ones(grid_shape)
    x, y = out.shape[1] // 2 - img.shape[0] // 2, out.shape[0] // 2 - img.shape[1] // 2
    out[y : y + img.shape[1], x : x + img.shape[0]] = img.T
    return out / 255


@click.command()
@click.option("-t", "--test", is_flag=True)
@click.option("-w", "--width", prompt=True, type=float)
@click.option("-h", "--height", prompt=True, type=float)
@click.option("-m", "--margin", prompt=True, type=float, default=0.5)
@click.option("-r0", "--grid-res-0", prompt=True, type=int, default=1)
@click.option("-r0", "--grid-res-1", prompt=True, type=int, default=6)
@click.option("-l", "--line-length", prompt=True, type=int, default=200)
@click.option("-l", "--line-separation", prompt=True, type=float, default=4)
@click.option("-cn", "--n-colors", prompt=True, type=int, default=3)
@click.option("-cc", "--color-cohesion", prompt=True, type=float, default=2)
@click.option("-b", "--img-blur", prompt=True, type=float, default=5)
def main(
    test: bool,
    width: float,
    height: float,
    margin: float,
    grid_res_0: int,
    grid_res_1: int,
    line_length: int,
    line_separation: float,
    n_colors: int,
    color_cohesion: float,
    img_blur: float,
):
    grid_shape_0, grid_res_0 = derive_grid_shape(width, height, margin, grid_res_0)
    grid_shape_1, grid_res_1 = derive_grid_shape(width, height, margin, grid_res_1)
    curl0 = curl_noise(grid_shape_0, grid_res_0) * 0.5
    curl1 = curl_noise(grid_shape_1, grid_res_1)
    img = Image.open("heart.png")
    img = 1 - resize_and_center_image(curl1.shape[:2], img, 0.8, img_blur)
    field = blend_vector_fields(curl0, curl1, lambda x, y: img[y, x])
    layers = render_flow_field(
        field, line_length, line_separation, n_colors, color_cohesion
    )
    layers = Drawing.multi_scale_to_fit(layers, width, height, margin)
    layers = [layer.sort_paths() for layer in layers]
    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(layers, bounds=(0, 0, width, height))
        im.write_to_png("turbulence_boundary_preview.png")
        im.finish()
    else:
        axi.draw_layers(layers)


if __name__ == "__main__":
    main()
