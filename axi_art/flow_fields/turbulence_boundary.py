import axi
import click
import numpy as np
from axi import Drawing

from axi_art.flow_fields.flow_fields import curl_noise, derive_grid_shape, blend_vector_fields, render_flow_field


def logistic(x: float, midpoint: float, steepness: float) -> float:
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))


@click.command()
@click.option('-t', '--test', is_flag=True)
@click.option('-w', '--width', prompt=True, type=float)
@click.option('-h', '--height', prompt=True, type=float)
@click.option('-m', '--margin', prompt=True, type=float, default=0.5)
@click.option('-sx', '--grid-dpi', prompt=True, type=int, default=25)
@click.option('-ry', '--grid-res', prompt=True, type=int, default=5)
@click.option('-l', '--line-length', prompt=True, type=int, default=200)
@click.option('-l', '--line-separation', prompt=True, type=float, default=5)
@click.option('-cn', '--n-colors', prompt=True, type=int, default=3)
@click.option('-cc', '--color-cohesion', prompt=True, type=float, default=1.2)
@click.option('-mid', '--midpoint', prompt=True, type=float, default=-0.2)
@click.option('-st', '--steepness', prompt=True, type=float, default=4)
def main(test: bool, width: float, height: float, margin: float,
         grid_dpi: float, grid_res: int,
         line_length: int, line_separation: float,
         n_colors: int, color_cohesion: float,
         midpoint: float, steepness: float):
    grid_shape, grid_res = derive_grid_shape(width, height, margin, grid_dpi, grid_res)
    curl = curl_noise(grid_shape, grid_res)
    horizontal = np.ones(curl.shape) * [0.5, 0]
    field = blend_vector_fields(horizontal, curl, lambda x, y: logistic(x / grid_shape[0], midpoint, steepness))
    layers = render_flow_field(field, line_length, line_separation, n_colors, color_cohesion)
    layers = Drawing.multi_scale_to_fit(layers, width, height, margin)
    layers = [layer.sort_paths() for layer in layers]
    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(layers, bounds=(0, 0, width, height))
        im.write_to_png('turbulence_boundary_preview.png')
        im.finish()
    else:
        axi.draw_layers(layers)


if __name__ == '__main__':
    main()
