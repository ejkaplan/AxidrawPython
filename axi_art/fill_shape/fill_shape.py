import numpy as np
from axi import shapely_to_paths, Drawing
from freetype import Face
from shapely.geometry import MultiLineString, Polygon, Point
from shapely.ops import split


def load_character(face: Face, char: str) -> Polygon:
    face.load_char(char)
    shape = Polygon(face.glyph.outline.points)
    return shape


def shade_shape(shape: Polygon, gap: float, resolution: float) -> Drawing:
    print(shape.bounds)
    min_x, min_y, max_x, max_y = shape.bounds
    paths = []
    for y in np.arange(min_y, max_y, gap):
        line_start = None
        line_end = None
        for x in np.arange(min_x, max_x, resolution):
            point = Point(x, y)
            if shape.contains(point):
                if line_start is None:
                    line_start = (x, y)
                line_end = (x, y)
            else:
                if line_start and line_start != line_end:
                    paths.append([line_start, line_end])
                line_start, line_end = None, None
        if line_start and line_start != line_end:
            paths.append([line_start, line_end])
    return Drawing(paths).scale(1, -1)


def main():
    face = Face('C:/Windows/Fonts/TAHOMA.ttf')
    face.set_char_size(20)
    letter = load_character(face, 'B')
    print(letter.exterior)
    drawing = shade_shape(letter, 0.3, 0.1)
    drawing = drawing.scale_to_fit(8, 8, 1).center(8, 8)
    im = drawing.render(bounds=(0, 0, 8, 8))
    im.write_to_png("circle.png")


if __name__ == "__main__":
    main()

