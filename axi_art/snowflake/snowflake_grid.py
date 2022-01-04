import axi
import click
import numpy as np
from axi import Drawing
from tqdm import tqdm

from axi_art.snowflake.snowflake import snowflake, overlay


@click.command()
@click.option('-t', '--test', is_flag=True)
@click.option('-w', '--width', prompt=True, type=float)
@click.option('-h', '--height', prompt=True, type=float)
@click.option('-m', '--margin', prompt=True, type=float)
@click.option('-r', '--rows', prompt=True, type=int)
@click.option('-c', '--cols', prompt=True, type=int)
@click.option('-co', '--colors', prompt=True, type=int)
@click.option('-ps', '--points_small', prompt=True, type=int, default=20)
@click.option('-pb', '--points_big', prompt=True, type=int, default=30)
@click.option('-ts', '--theta_small', prompt=True, type=float, default=0.2)
@click.option('-tb', '--theta_big', prompt=True, type=float, default=0.18)
def flake_grid(test: bool, width: float, height: float, margin: float, rows: int, cols: int, colors: int, 
               points_small: int, points_big: int, theta_small: float, theta_big: float):
    layers = [Drawing() for _ in range(colors)]
    axi.device.MAX_VELOCITY = 2
    flake_size = (width / cols, height / rows)
    with tqdm(total=rows * cols) as pbar:
        for x in np.linspace(0, width, cols, endpoint=False):
            for y in np.linspace(0, height, rows, endpoint=False):
                small_flake = snowflake(points_small, 1.5, 10000, 6, theta_small).rotate(np.pi / 6)
                big_flake = snowflake(points_big, 1.5, 10000, 6, theta_big)
                small_flake = overlay(big_flake, small_flake)
                curr_layers = [small_flake,
                               big_flake]
                curr_layers = Drawing.multi_scale_to_fit(curr_layers, *flake_size, max(flake_size) * 0.05)
                curr_layers = [layer.translate(x, y).sort_paths().join_paths(0.001) for layer in curr_layers]
                rand_ints = np.random.choice(colors, size=2, replace=False)
                for i, layer in enumerate(curr_layers):
                    layers[rand_ints[i]].add(layer)
                pbar.update(1)
    layers = Drawing.multi_scale_to_fit(layers, width, height, margin)
    layers = [layer.sort_paths().join_paths(0.001) for layer in layers]

    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(layers, bounds=(0, 0, width, height))
        im.write_to_png('snowflake_preview.png')
        im.finish()
    else:
        axi.draw_layers(layers)


if __name__ == '__main__':
    flake_grid()
