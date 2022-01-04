import axi
import numpy as np
from axi import Drawing
from tqdm import tqdm

from axi_art.snowflake.snowflake import snowflake, overlay

TEST = True
WIDTH = 11
HEIGHT = 8.5
MARGIN = 0.5
ROWS = 5
COLS = 7
N_COLORS = 3


def main():
    layers = [Drawing() for _ in range(N_COLORS)]
    axi.device.MAX_VELOCITY = 2
    flake_size = (WIDTH / COLS, HEIGHT / ROWS)
    with tqdm(total=ROWS*COLS) as pbar:
        for x in np.linspace(0, WIDTH, COLS, endpoint=False):
            for y in np.linspace(0, HEIGHT, ROWS, endpoint=False):
                small_flake = snowflake(20, 1.5, 10000, 6, 0.2).rotate(np.pi / 6)
                big_flake = snowflake(30, 1.5, 10000, 6, 0.18)
                small_flake = overlay(big_flake, small_flake)
                curr_layers = [small_flake,
                               big_flake]
                curr_layers = Drawing.multi_scale_to_fit(curr_layers, *flake_size, max(flake_size) * 0.05)
                curr_layers = [layer.translate(x, y).sort_paths().join_paths(0.001) for layer in curr_layers]
                rand_ints = np.random.choice(N_COLORS, size=2, replace=False)
                for i, layer in enumerate(curr_layers):
                    layers[rand_ints[i]].add(layer)
                pbar.update(1)
    layers = Drawing.multi_scale_to_fit(layers, WIDTH, HEIGHT, 0.5)
    layers = [layer.sort_paths().join_paths(0.001) for layer in layers]

    if TEST or axi.device.find_port() is None:
        im = Drawing.render_layers(layers, bounds=(0, 0, WIDTH, HEIGHT))
        im.write_to_png('snowflake_preview.png')
        im.finish()
    else:
        axi.draw_layers(layers)


if __name__ == '__main__':
    main()
