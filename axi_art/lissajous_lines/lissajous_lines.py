import axi
import numpy as np

PAPER_WIDTH = 5.5
PAPER_HEIGHT = 4.25
PAPER_MARGIN = 0.5


def lissajous(lo, hi, samples):
    while True:
        x_coef, y_coef = np.random.uniform(1, 5, 2)
        x_off, y_off = np.random.uniform(-100, 100, 2)
        if np.abs(x_coef - y_coef) > 0.5:
            break
    w = PAPER_WIDTH - 2 * PAPER_MARGIN
    h = PAPER_HEIGHT - 2 * PAPER_MARGIN
    return [(w * np.cos(x_coef * theta + x_off), h * np.sin(y_coef * theta + y_off)) for theta in
            np.linspace(lo, hi, samples)]


def lissajous_lines(lo, hi, samples):
    curves = [lissajous(lo, hi, samples), lissajous(lo, hi, samples)]
    paths = [[curves[0][i], curves[1][i]] for i in range(samples)]
    return axi.Drawing(paths)


def make_drawing():
    drawing = lissajous_lines(0, np.pi, 300).sort_paths()
    drawing = drawing.scale_to_fit(PAPER_WIDTH, PAPER_HEIGHT, PAPER_MARGIN)
    drawing = drawing.center(PAPER_WIDTH, PAPER_HEIGHT)
    img = drawing.render(scale=150, margin=0.2)
    img.write_to_png('lissajous.png')
    return drawing


SEED = False
TEST = False


def main():
    seed = SEED if SEED else np.random.randint(2 ** 31)
    print(f"The random seed is {seed}")
    np.random.seed(seed)
    drawing = make_drawing()
    if axi.device.find_port() is not None and not TEST:
        axi.draw(drawing)


if __name__ == "__main__":
    main()
