import axi
import click
import numpy as np
from axi import Drawing

from axi_art.truchet.truchet_multiscale import make_grid

facemap = {
    (0, 0): 0,
    (0, 1): 0,
    (0, 2): 1,
    (0, 3): 2,
    (1, 0): 1,
    (1, 3): 2,
    (2, 0): 2,
    (2, 3): 1,
    (3, 0): 2,
    (3, 1): 1,
    (3, 2): 0,
    (3, 3): 0,
}

face_colors = [(0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)]


def flexagon(
    rows: int, cols: int, max_block_size: int, prob_turn: float, front: bool
) -> list[Drawing]:
    layers = [Drawing() for _ in range(3)]
    for r in range(4):
        for c in range(4):
            if (r, c) not in facemap:
                continue
            cell_layers = [
                layer.translate(c * cols, r * rows)
                for layer in make_grid(rows, cols, max_block_size).render(prob_turn)
            ]
            colors = face_colors[facemap[(r, c)] + (0 if front else 3)]
            layers[colors[0]].add(cell_layers[0])
            layers[colors[1]].add(cell_layers[1])
    return layers


@click.command()
@click.option("-t", "--test", is_flag=True)
@click.option("-w", "--width", prompt=True, type=float)
@click.option("-h", "--height", prompt=True, type=float)
@click.option("-m", "--margin", prompt=True, type=float)
@click.option("-r", "--rows", prompt=True, type=int)
@click.option("-b", "--max-block-size", prompt=True, type=int)
@click.option("-p", "--prob_turn", prompt=True, type=float)
def main(
    test: bool,
    width: float,
    height: float,
    margin: float,
    rows: int,
    max_block_size: int,
    prob_turn: float,
):
    front = flexagon(
        rows, round(rows * height / width), max_block_size, prob_turn, True
    )
    front = Drawing.multi_scale_to_fit(list(front), width, height, padding=margin)
    front = [layer.join_paths(0.05).sort_paths() for layer in front]
    back = flexagon(
        rows, round(rows * height / width), max_block_size, prob_turn, False
    )
    back = Drawing.multi_scale_to_fit(list(back), width, height, padding=margin)
    back = [layer.join_paths(0.05).sort_paths() for layer in back]
    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(front, bounds=(0, 0, width, height))
        im.write_to_png("truchet_flex_front.png")
        im = Drawing.render_layers(back, bounds=(0, 0, width, height))
        im.write_to_png("truchet_flex_back.png")
    else:
        print("FRONT")
        axi.draw_layers(front)
        print("BACK")
        axi.draw_layers(back)


if __name__ == "__main__":
    main()
