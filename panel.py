"""Shared helper for writing a labeled grid of images to a single PNG,
used by the various diagnostic/preview scripts."""

from PIL import Image, ImageDraw, ImageFont


def make_panel(labeled_images, filename, cols, cell_size=200, pad=6, label_h=18):
    rows = -(-len(labeled_images) // cols)  # ceil division
    panel_w = cols * cell_size + (cols + 1) * pad
    panel_h = rows * (cell_size + label_h) + (rows + 1) * pad
    panel = Image.new("RGB", (panel_w, panel_h), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    for i, (label, img) in enumerate(labeled_images):
        r, c = divmod(i, cols)
        x = pad + c * (cell_size + pad)
        y = pad + r * (cell_size + label_h + pad)
        thumb = Image.fromarray(img).resize((cell_size, cell_size), resample=Image.NEAREST)
        panel.paste(thumb, (x, y))
        draw.text((x, y + cell_size + 2), label, fill="black", font=font)

    panel.save(filename)
