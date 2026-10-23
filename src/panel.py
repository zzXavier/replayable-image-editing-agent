import json
from PIL import Image, ImageDraw


def panel(folder, dest):
    modes = ("source", "direct", "fixed", "feedback")
    canvas = Image.new("RGB", (4 * 160, 198), "white")
    d = ImageDraw.Draw(canvas)
    for i, mode in enumerate(modes):
        if mode == "source":
            img = folder / "direct" / "source.png"
            status = "original"
        else:
            log = json.loads((folder / mode / "trace.json").read_text())
            status = log["status"]
            img = folder / mode / log["chosen_image"] if "chosen_image" in log else None
        if img is not None:
            with Image.open(img) as im:
                canvas.paste(im.convert("RGB").resize((160, 160)), (160 * i, 20))
        d.text((160 * i + 3, 2), mode, fill="black")
        d.text((160 * i + 3, 182), status, fill="black")
    canvas.save(dest)
