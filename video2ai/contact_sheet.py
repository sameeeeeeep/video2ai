"""Generate contact sheets (frame grids) from extracted frames."""

import os
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


@dataclass
class ContactSheet:
    index: int
    time_range: str
    path: str
    frame_indices: list[int]


def build_contact_sheets(
    frames: list,  # list[ExtractedFrame]
    output_dir: str,
    cols: int = 3,
    rows: int = 3,
    quality: int = 85,
) -> list[ContactSheet]:
    """Combine extracted frames into grid contact sheets."""
    sheets_dir = os.path.join(output_dir, "contact_sheets")
    os.makedirs(sheets_dir, exist_ok=True)

    per_sheet = cols * rows
    sheets = []

    for sheet_idx, chunk_start in enumerate(range(0, len(frames), per_sheet)):
        chunk = frames[chunk_start : chunk_start + per_sheet]
        sheet_num = sheet_idx + 1

        sheet = _build_one_sheet(chunk, sheets_dir, sheet_num, cols, rows, quality)
        if sheet:
            sheets.append(sheet)

    return sheets


def _build_one_sheet(
    frames: list,
    sheets_dir: str,
    sheet_num: int,
    cols: int,
    rows: int,
    quality: int,
) -> ContactSheet | None:
    """Build a single contact sheet from a chunk of frames."""
    if not frames:
        return None

    # Load first frame to get cell dimensions
    sample = Image.open(frames[0].path)
    cell_w, cell_h = sample.size

    # Layout params
    label_height = 28
    padding = 4
    total_w = cols * cell_w + (cols + 1) * padding
    total_h = rows * (cell_h + label_height) + (rows + 1) * padding

    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    # Try to load a reasonable font
    font = _get_font(size=16)

    frame_indices = []
    for i, frame in enumerate(frames):
        col = i % cols
        row = i // cols
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + label_height + padding)

        # Draw frame image
        img = Image.open(frame.path)
        if img.size != (cell_w, cell_h):
            img = img.resize((cell_w, cell_h), Image.LANCZOS)
        canvas.paste(img, (x, y))

        # Draw timestamp label below
        label = _format_timestamp(frame.timestamp)
        label_y = y + cell_h + 2
        draw.text((x + 4, label_y), label, fill=(200, 200, 200), font=font)

        frame_indices.append(frame.index)

    # Time range for this sheet
    t_start = _format_timestamp(frames[0].timestamp)
    t_end = _format_timestamp(frames[-1].timestamp)
    time_range = f"{t_start} - {t_end}"

    path = os.path.join(sheets_dir, f"sheet_{sheet_num:03d}.jpg")
    canvas.save(path, "JPEG", quality=quality)

    return ContactSheet(
        index=sheet_num,
        time_range=time_range,
        path=path,
        frame_indices=frame_indices,
    )


def _format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _get_font(size: int = 16):
    """Try to load a monospace font, fall back to default."""
    font_paths = [
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()
