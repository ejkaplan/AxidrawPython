import axi
import click
import numpy as np
from axi import Drawing
from tqdm import tqdm

from axi_art.snowflake.snowflake import snowflake, overlay


@click.command()
@click.option('--test', is_flag=True)
@click.option('--width', prompt=True, type=float)
@click.option('--height', prompt=True, type=float)
@click.option('--margin', prompt=True, type=float)
@click.option('--rows', prompt=True, type=int)
@click.option('--cols', prompt=True, type=int)
@click.option('--colors', prompt=True, type=int)
def flake(test: bool, width: float, height: float, margin: float, rows: int, cols: int, colors: int):
    layers = [Drawing() for _ in range(colors)]
    axi.device.MAX_VELOCITY = 2
    flake_size = (width / cols, height / rows)
    with tqdm(total=rows*cols) as pbar:
        for x in np.linspace(0, width, cols, endpoint=False):
            for y in np.linspace(0, height, rows, endpoint=False):
                small_flake = snowflake(20, 1.5, 10000, 6, 0.2).rotate(np.pi / 6)
                big_flake = snowflake(30, 1.5, 10000, 6, 0.18)
                small_flake = overlay(big_flake, small_flake)
                curr_layers = [small_flake,
                               big_flake]
                curr_layers = Drawing.multi_scale_to_fit(curr_layers, *flake_size, max(flake_size) * 0.05)
                curr_layers = [layer.translate(x, y).sort_paths().join_paths(0.001) for layer in curr_layers]
                rand_ints = np.random.choice(colors, size=2, replace=False)
                for i, layer in enumerate(curr_layers):
                    layers[rand_ints[i]].add(layer)
                pbar.update(1)
    layers = Drawing.multi_scale_to_fit(layers, width, height, 0.5)
    layers = [layer.sort_paths().join_paths(0.001) for layer in layers]

    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(layers, bounds=(0, 0, width, height))
        im.write_to_png('snowflake_preview.png')
        im.finish()
    else:
        axi.draw_layers(layers)


if __name__ == '__main__':
    flake()
