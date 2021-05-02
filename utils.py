import itertools
import string

import axi


def merge_paths(paths):
    print(f"BEFORE MERGE: {len(paths) - 1} pen lifts")
    frontier = paths.copy()
    finished = []
    while len(frontier) > 0:
        curr = frontier.pop(0)
        changed = False
        for other in frontier:
            new_path = None
            if curr[0] == other[0]:
                new_path = curr[:0:-1] + other
            elif curr[0] == other[-1]:
                new_path = other[:-1] + curr
            elif curr[-1] == other[0]:
                new_path = curr[:-1] + other
            elif curr[-1] == other[-1]:
                new_path = curr + other[-2::-1]
            if new_path is not None:
                frontier.remove(other)
                frontier.append(new_path)
                changed = True
                break
        if not changed:
            finished.append(curr)
    print(f"AFTER MERGE: {len(finished) - 1} pen lifts")
    return finished


def offset_paths(paths, off_x, off_y):
    return [[(p[0] + off_x, p[1] + off_y) for p in path] for path in paths]


def map_range(val, a0, a1, b0, b1):
    p = (val-a0) / (a1-a0)
    return b0 + p * (b1 - b0)


def word_wrap(text, width, measure_func):
    result = []
    for line in text.split('\n'):
        fields = itertools.groupby(line, lambda x: x.isspace())
        fields = [''.join(g) for _, g in fields]
        if len(fields) % 2 == 1:
            fields.append('')
        x = ''
        for a, b in zip(fields[::2], fields[1::2]):
            w, _ = measure_func(x + a)
            if w > width:
                if x == '':
                    result.append(a)
                    continue
                else:
                    result.append(x)
                    x = ''
            x += a + b
        if x != '':
            result.append(x)
    result = [x.strip() for x in result]
    return result


class Font(object):
    def __init__(self, font, point_size):
        self.font = font
        self.max_height = axi.Drawing(axi.text(string.printable, font)).height
        # self.cap_height = axi.Drawing(axi.text('H', font)).height
        height = point_size / 72
        self.scale = height / self.max_height

    def text(self, text):
        d = axi.Drawing(axi.text(text, self.font))
        d = d.scale(self.scale)
        return d

    def justify_text(self, text, width):
        d = self.text(text)
        w = d.width
        spaces = text.count(' ')
        if spaces == 0 or w >= width:
            return d
        e = ((width - w) / spaces) / self.scale
        d = axi.Drawing(axi.text(text, self.font, extra=e))
        d = d.scale(self.scale)
        return d

    def measure(self, text):
        return self.text(text).size

    def wrap(self, text, width, line_spacing=1, align=0, justify=False):
        lines = word_wrap(text, width, self.measure)
        ds = [self.text(line) for line in lines]
        max_width = max(d.width for d in ds)
        if justify:
            jds = [self.justify_text(line, max_width) for line in lines]
            ds = jds[:-1] + [ds[-1]]
        spacing = line_spacing * self.max_height * self.scale
        result = axi.Drawing()
        y = 0
        for d in ds:
            if align == 0:
                x = 0
            elif align == 1:
                x = max_width - d.width
            else:
                x = max_width / 2 - d.width / 2
            result.add(d.translate(x, y))
            y += spacing
        return result


def vertical_stack(ds, spacing=0, center=True):
    result = axi.Drawing()
    y = 0
    for d in ds:
        if center:
            d = d.origin().translate(-d.width / 2, y)
        else:
            d = d.origin().translate(0, y)
        result.add(d)
        y += d.height + spacing
    return result


def horizontal_stack(ds, spacing=0):
    result = axi.Drawing()
    x = 0
    for d in ds:
        d = d.origin().translate(x, -d.height / 2)
        result.add(d)
        x += d.width + spacing
    return result
