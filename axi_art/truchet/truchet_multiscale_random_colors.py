import axi
import click
import numpy as np
from axi import Drawing

from axi_art.truchet.truchet_multiscale import truchet_corner_circles, truchet_corner_circle_highlights, \
    truchet_crossed_lines, truchet_crossed_lines_highlights, Grid, make_grid


def render_random_colors(grid: Grid, p_turn: float, colors: int) -> list[Drawing]:
    layers = [Drawing() for _ in range(colors)]
    for box in grid.boxes:
        if np.random.random() < p_turn:
            tile = truchet_corner_circles(box.size, 2 * box.size)
            highlight = truchet_corner_circle_highlights(box.size, 2 * box.size)
        else:
            tile = truchet_crossed_lines(box.size, 2 * box.size)
            highlight = truchet_crossed_lines_highlights(box.size, 2 * box.size)
        curr_layers = np.random.choice(layers, size=2, replace=False)
        tile = tile.translate(-box.size / 2, -box.size / 2)
        angle = np.random.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        tile = tile.rotate(angle)
        tile = tile.translate(box.c + box.size / 2, box.r + box.size / 2)
        curr_layers[0].add(tile)
        highlight = highlight.translate(-box.size / 2, -box.size / 2)
        highlight = highlight.rotate(angle)
        highlight = highlight.translate(box.c + box.size / 2, box.r + box.size / 2)
        curr_layers[1].add(highlight)
    return layers


@click.command()
@click.option("-t", "--test", is_flag=True)
@click.option("-w", "--width", prompt=True, type=float)
@click.option("-h", "--height", prompt=True, type=float)
@click.option("-m", "--margin", prompt=True, type=float)
@click.option("-r", "--rows", prompt=True, type=int)
@click.option("-b", "--max-block-size", prompt=True, type=int)
@click.option("-p", "--prob_turn", prompt=True, type=float)
@click.option("-c", "--colors", prompt=True, type=int)
def main(
        test: bool,
        width: float,
        height: float,
        margin: float,
        rows: int,
        max_block_size: int,
        prob_turn: float,
        colors: int,
):
    grid = make_grid(rows, round(rows * height / width), max_block_size)
    layers = render_random_colors(grid, prob_turn, colors)
    layers = Drawing.multi_scale_to_fit(list(layers), width, height, padding=margin)
    layers = [layer.join_paths(0.05).sort_paths() for layer in layers]
    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(layers, bounds=(0, 0, width, height))
        im.write_to_png("truchet_random_colors.png")
    else:
        axi.draw_layers(layers)


if __name__ == "__main__":
    main()