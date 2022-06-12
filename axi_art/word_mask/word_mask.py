import axi
from PIL import Image
from axi import Drawing

from axi_art.utils import Font


def map(val: float, lo0: float, hi0: float, lo1: float, hi1: float):
    p = (val - lo0) / (hi0 - lo0)
    return lo1 + p * (hi1 - lo1)


def letter_grid(
    font, mask: Image, text: str, width: float, height: float
) -> list[Drawing]:
    letters: dict[str, Drawing] = {c: font.text(c) for c in set(text)}
    row_height = 1.3 * max(d.height for d in letters.values())
    out = [Drawing(), Drawing()]
    y = 0
    row_count = 0
    while y < height:
        x = -row_height/2
        i = row_count % len(text)
        while x < width:
            if (c := text[i]) != " ":
                d = letters[c].translate(x, y)
                try:
                    mask_x = int(map(x + d.width / 2, 0, width, 0, mask.size[0]))
                    mask_y = int(map(y + d.height / 2, 0, height, 0, mask.size[1]))
                    layer = mask.getpixel((mask_x, mask_y)) > 127
                except IndexError:
                    layer = 1
                out[layer] = Drawing.combine([out[layer], d])
                x += 1.3 * d.width
            else:
                x += 0.3 * row_height
            i = (i + 1) % len(text)
        y += row_height
        row_count += 1
    return out


def rect(x: float, y: float, w: float, h: float) -> Drawing:
    paths = [[(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]]
    return Drawing(paths)


def main():
    im = Image.open("name.png").convert("1")
    f = Font(axi.FUTURAL, 7)
    layers = letter_grid(f, im, "they them ", 3.5, 2.25)
    layers = [layer.translate(0.5, 0.5) for layer in layers]
    layers[0] = Drawing.combine([layers[0], rect(0.5, 0.5, 3.5, 2.25)])
    if axi.device.find_port() is None:
        im = Drawing.render_layers(layers, bounds=(0, 0, 11, 8.5))
        # im = Drawing.render(text_drawings[0], bounds=(0, 0, 11, 8.5))
        im.write_to_png("nametag.png")
    else:
        axi.draw_layers(layers)


if __name__ == "__main__":
    main()
