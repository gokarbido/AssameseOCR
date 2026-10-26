"""Large, bounded video labels using scalable fonts for both languages."""
from functools import lru_cache

from PIL import ImageFont


def font_size(frame_height):
    return max(28, min(64, round(frame_height / 22.5)))


def place_label(box, frame_shape, width, height, occupied, boxes):
    """Place a whole bilingual block, or leave it to the sidebar if crowded."""
    frame_height, frame_width = frame_shape[:2]
    if width > frame_width or height > frame_height:
        return None
    gap = 8
    x_values = [box[0], 0, frame_width - width, box[2] + gap, box[0] - width - gap]
    y_values = [box[1] - height - gap, box[3] + gap, 0, frame_height - height, box[1]]
    for rectangle in occupied:
        x_values.extend((rectangle[2] + gap, rectangle[0] - width - gap))
        y_values.extend((rectangle[3] + gap, rectangle[1] - height - gap))
    xs = list(dict.fromkeys(int(max(0, min(x, frame_width - width))) for x in x_values))
    ys = list(dict.fromkeys(int(max(0, min(y, frame_height - height))) for y in y_values))

    def distance(position):
        x, y = position
        dx = max(box[0] - (x + width), x - box[2], 0)
        dy = max(box[1] - (y + height), y - box[3], 0)
        return dx + dy + abs(x - box[0]) * 0.001

    candidates = sorted(((x, y) for x in xs for y in ys), key=distance)

    def overlaps(rectangle, other):
        return (rectangle[0] < other[2] + 4 and rectangle[2] > other[0] - 4
                and rectangle[1] < other[3] + 4 and rectangle[3] > other[1] - 4)

    # Prefer free video space; allow region interiors only when necessary.
    # Never cover another label or separate the Assamese and English lines.
    for obstacles in (list(occupied) + list(boxes), occupied):
        for x, y in candidates:
            rectangle = (x, y, x + width, y + height)
            if not any(overlaps(rectangle, other) for other in obstacles):
                occupied.append(rectangle)
                return x, y
    return None


@lru_cache(maxsize=32)
def display_font(size, lang, preferred=None):
    english = ('/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
               '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf')
    assamese = ('/usr/share/fonts/truetype/lohit-assamese/Lohit-Assamese.ttf',
                '/usr/share/fonts/truetype/lohit-assamese/lohit_assamese.ttf',
                '/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf')
    candidates = ((preferred,) + assamese + english if lang == 'as' else english + (preferred,))
    for path in candidates:
        if path:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Older local Pillow versions; container requires Pillow >=10.4.
        return ImageFont.load_default()


def label_layout(draw, text, frame_shape, lang, preferred=None):
    height, width = frame_shape[:2]
    font = display_font(font_size(height), lang, preferred)
    padding, spacing = 6, 6
    available = max(1, width - 2 * padding)
    lines, current = [], ''
    for word in text.split():
        candidate = f'{current} {word}'.strip()
        if current and draw.textlength(candidate, font=font) > available:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > 2:
        lines = lines[:2]
        lines[-1] += '…'
    # Keep very long words within the video; the sidebar retains the full text.
    for index, line in enumerate(lines):
        if draw.textlength(line, font=font) > available:
            while line and draw.textlength(line + '…', font=font) > available:
                line = line[:-1]
            lines[index] = line + '…'
    wrapped = '\n'.join(lines)
    bounds = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing)
    return dict(text=wrapped, font=font, bounds=bounds, padding=padding, spacing=spacing,
                width=min(width, bounds[2] - bounds[0] + padding * 2),
                height=min(height, bounds[3] - bounds[1] + padding * 2))
