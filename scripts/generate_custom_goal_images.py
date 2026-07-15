#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 256
HEIGHT = 178
BACKGROUND = (255, 255, 255)
INK = (0, 0, 255)
LINE_WIDTH = 8
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "target_drawings_tasks"


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
        "LiberationSans-Regular.ttf",
    ]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    return image, ImageDraw.Draw(image)


def _draw_centered_text(draw: ImageDraw.ImageDraw, text: str, font_size: int = 84) -> None:
    font = _load_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (WIDTH - text_width) / 2
    y = (HEIGHT - text_height) / 2 - 6
    draw.text((x, y), text, fill=INK, font=font)


def _draw_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]]) -> None:
    draw.line(points, fill=INK, width=LINE_WIDTH, joint="curve")


def _draw_one(draw: ImageDraw.ImageDraw, x: int, y: int, scale: float = 1.0) -> None:
    top = (x + int(8 * scale), y + int(6 * scale))
    mid = (x + int(22 * scale), y + int(22 * scale))
    bottom = (x + int(22 * scale), y + int(114 * scale))
    base_left = (x, y + int(114 * scale))
    base_right = (x + int(42 * scale), y + int(114 * scale))
    _draw_polyline(draw, [top, mid, bottom])
    _draw_polyline(draw, [base_left, base_right])


def _draw_five(draw: ImageDraw.ImageDraw, x: int, y: int, scale: float = 1.0) -> None:
    p1 = (x + int(54 * scale), y)
    p2 = (x + int(8 * scale), y)
    p3 = (x + int(8 * scale), y + int(46 * scale))
    p4 = (x + int(48 * scale), y + int(46 * scale))
    p5 = (x + int(58 * scale), y + int(58 * scale))
    p6 = (x + int(58 * scale), y + int(96 * scale))
    p7 = (x + int(46 * scale), y + int(114 * scale))
    p8 = (x + int(10 * scale), y + int(114 * scale))
    _draw_polyline(draw, [p1, p2, p3, p4, p5, p6, p7, p8])


def _draw_a(draw: ImageDraw.ImageDraw, x: int, y: int, scale: float = 1.0) -> None:
    apex = (x + int(34 * scale), y)
    left_base = (x, y + int(114 * scale))
    right_base = (x + int(68 * scale), y + int(114 * scale))
    cross_left = (x + int(16 * scale), y + int(62 * scale))
    cross_right = (x + int(52 * scale), y + int(62 * scale))
    _draw_polyline(draw, [left_base, apex, right_base])
    _draw_polyline(draw, [cross_left, cross_right])


def _draw_square(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((58, 30, 198, 148), outline=INK, width=LINE_WIDTH)


def _draw_circle(draw: ImageDraw.ImageDraw) -> None:
    draw.ellipse((58, 22, 198, 162), outline=INK, width=LINE_WIDTH)


def _draw_face(draw: ImageDraw.ImageDraw) -> None:
    draw.ellipse((48, 16, 208, 168), outline=INK, width=LINE_WIDTH)
    draw.ellipse((86, 58, 108, 80), fill=INK)
    draw.ellipse((148, 58, 170, 80), fill=INK)
    draw.arc((86, 78, 170, 132), start=20, end=160, fill=INK, width=LINE_WIDTH)


def _draw_human(draw: ImageDraw.ImageDraw) -> None:
    draw.ellipse((106, 18, 150, 62), outline=INK, width=LINE_WIDTH)
    draw.line((128, 62, 128, 120), fill=INK, width=LINE_WIDTH)
    draw.line((88, 84, 168, 84), fill=INK, width=LINE_WIDTH)
    draw.line((128, 120, 92, 156), fill=INK, width=LINE_WIDTH)
    draw.line((128, 120, 164, 156), fill=INK, width=LINE_WIDTH)


def _draw_ta_field(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((54, 26, 202, 152), outline=INK, width=LINE_WIDTH)
    draw.line((128, 26, 128, 152), fill=INK, width=LINE_WIDTH)
    draw.line((54, 89, 202, 89), fill=INK, width=LINE_WIDTH)


def _save_rotated(image: Image.Image, filename: str) -> None:
    image.save(OUTPUT_DIR / filename)


def _save_text_image(filename: str, text: str, font_size: int = 84, stroke_width: int = 2) -> None:
    image, draw = _new_canvas()
    font = _load_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (WIDTH - text_width) / 2
    y = (HEIGHT - text_height) / 2 - 6
    draw.text((x, y), text, fill=INK, font=font, stroke_width=stroke_width, stroke_fill=INK)
    transformed = image.rotate(90, expand=False, fillcolor=BACKGROUND)
    _save_rotated(transformed, filename)


def _save_vector_image(filename: str, draw_fn) -> None:
    image, draw = _new_canvas()
    draw_fn(draw)
    _save_rotated(image, filename)


def _save_vector_image_rot90(filename: str, draw_fn) -> None:
    image, draw = _new_canvas()
    draw_fn(draw)
    _save_rotated(image.rotate(90, expand=False, fillcolor=BACKGROUND), filename)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _save_text_image("write_a.png", "A", font_size=132, stroke_width=3)
    _save_text_image("draw_15.png", "15", font_size=128, stroke_width=3)
    _save_text_image("draw_55.png", "55", font_size=128, stroke_width=3)

    image, draw = _new_canvas()
    _draw_square(draw)
    _save_rotated(image, "draw_square.png")

    image, draw = _new_canvas()
    _draw_circle(draw)
    _save_rotated(image, "draw_circle.png")

    _save_vector_image_rot90("draw_face.png", _draw_face)
    _save_vector_image_rot90("draw_human.png", _draw_human)

    image, draw = _new_canvas()
    _draw_ta_field(draw)
    _save_rotated(image, "draw_tian.png")


if __name__ == "__main__":
    main()
