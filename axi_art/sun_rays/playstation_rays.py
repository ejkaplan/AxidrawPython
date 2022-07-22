import axi
import click
import numpy as np
from axi import Drawing

from axi_art.sun_rays.sun_rays import radial_drawing, n_gon, x_shape
from shapely.affinity import rotate


@click.command()
@click.option("-t", "--test", is_flag=True)
@click.option("-w", "--width", prompt=True, type=float, default=7.87)
@click.option("-h", "--height", prompt=True, type=float, default=7.87)
@click.option("-m", "--margin", prompt=True, type=float, default=0.5)
@click.option("-s", "--spokes", prompt=True, type=int, default=100)
@click.option("-r", "--radius", prompt=True, type=float, default=2)
def main(
    test: bool, width: float, height: float, margin: float, spokes: int, radius: float
):
    dw, dh = width - 2 * margin, height - 2 * margin
    centers = [
        (dw / 2 + radius * np.cos(angle), dh / 2 + radius * np.sin(angle))
        for angle in np.linspace(0, 2 * np.pi, 4, endpoint=False)
    ]
    emitters = [
        n_gon(150, *centers[0], dw / 8),  # Circle
        x_shape(*centers[1], dw / 30, dw / 8, 0.2),  # X
        rotate(n_gon(4, *centers[2], dw / 7), 45, "center"),  # Square
        rotate(n_gon(3, *centers[3], dw / 8), 30, "center"),  # Triangle
    ]
    drawings = radial_drawing(dw, dh, emitters, spokes)
    drawings = [drawing.repeat().sort_paths() for drawing in drawings]
    drawings = Drawing.multi_scale_to_fit(drawings, width, height, margin)

    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(drawings, bounds=(0, 0, width, height))
        im.write_to_png("suns.png")
    else:
        axi.draw_layers(drawings)


if __name__ == "__main__":
    main()
