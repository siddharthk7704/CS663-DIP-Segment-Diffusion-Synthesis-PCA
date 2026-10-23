from pathlib import Path
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

OUT_SIZE = 256
PATCH_RADIUS = 4        
EPSILON = 0.10       
MAX_CANDIDATES = 64     
BATCH = 30      
MAX_SOURCE_PATCHES = 10_000  
SEARCH_MARGIN = 40      
USE_SAMPLE_TEXTURE_FOR_INPAINT = False
SHOW_PLOTS = True
SEED = 42


def read_image(path):
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def gaussian_weights(radius):
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    sigma = max(radius / 1.5, 0.8)
    w = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return w / w.sum()


def window_sum(mask, r):
    m = mask.astype(np.int32)
    p = np.pad(m, r)
    h, w = m.shape
    out = np.zeros((h, w), dtype=np.int32)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out += p[dy:dy + h, dx:dx + w]
    return out


def dilate(mask, r=1):
    return window_sum(mask, r) > 0


def fully_known_centres(known, r):
    k = 2 * r + 1
    return np.argwhere(window_sum(known, r) == k * k).astype(np.int32)

def extract_patches(source, centres, r):
    offs = np.arange(-r, r + 1)
    iy = centres[:, 0, None, None] + offs[None, :, None]
    ix = centres[:, 1, None, None] + offs[None, None, :]
    return source[iy, ix]


class PatchDB:
    def __init__(self, source, centres, r):
        patches = extract_patches(np.asarray(source, np.float64), centres, r)
        self.n = len(patches)
        if self.n == 0:
            raise ValueError("No valid source patches available.")
        self.channels = patches.shape[3]
        self.P = patches.reshape(self.n, -1) 
        self.P2 = self.P * self.P
        c = (2 * r + 1) ** 2 // 2                     
        self.centre_vals = self.P[:, c * self.channels:(c + 1) * self.channels]

    def distances(self, target, active):
        a = np.repeat(active[:, :, None], self.channels, axis=2).ravel()
        t = target.astype(np.float64).ravel()
        at = a * t
        d = self.P2 @ a - 2.0 * (self.P @ at) + float(at @ t)
        return np.maximum(d, 0.0)

    def sample(self, target, known_mask, weights, rng):
        active = known_mask * weights
        denom = float(active.sum())
        if denom <= 1e-12:                           
            return self.centre_vals[int(rng.integers(0, self.n))]
        d = self.distances(target, active) / denom
        best = float(d.min())
        thresh = (1.0 + EPSILON) * best + 1e-3       
        cand = np.flatnonzero(d <= thresh)
        if len(cand) > MAX_CANDIDATES:             
            keep = np.argpartition(d[cand], MAX_CANDIDATES - 1)[:MAX_CANDIDATES]
            cand = cand[keep]
        return self.centre_vals[int(cand[int(rng.integers(0, len(cand)))])]

def grow_fill(out_p, known_p, missing, db, weights, r, rng, label=""):
    k = 2 * r + 1
    total = int(missing.sum())
    done = 0
    last_print = time.time()

    while missing.any():
        ys, xs = np.nonzero(missing)
        y0, y1 = max(0, ys.min() - r), min(missing.shape[0], ys.max() + r + 1)
        x0, x1 = max(0, xs.min() - r), min(missing.shape[1], xs.max() + r + 1)
        known = known_p[r:-r, r:-r]
        counts = window_sum(known[y0:y1, x0:x1], r) * missing[y0:y1, x0:x1]
        cy, cx = np.nonzero(counts)

        if len(cy) == 0:
            i = int(rng.integers(0, len(ys)))
            order_y, order_x = ys[i:i + 1], xs[i:i + 1]
        else:
            order = np.lexsort((rng.random(len(cy)), -counts[cy, cx]))[:BATCH]
            order_y, order_x = cy[order] + y0, cx[order] + x0

        for y, x in zip(order_y, order_x):
            target = out_p[y:y + k, x:x + k]         
            mask = known_p[y:y + k, x:x + k]
            value = db.sample(target, mask, weights, rng)
            out_p[y + r, x + r] = value
            known_p[y + r, x + r] = True
            missing[y, x] = False
            done += 1

        if time.time() - last_print > 5:
            print(f"  {label} {done}/{total} pixels")
            last_print = time.time()

def synthesize_texture(seed_image, output_size=OUT_SIZE, radius=PATCH_RADIUS,
                       seed=SEED):
    source = np.asarray(seed_image, dtype=np.float64)
    sh, sw, ch = source.shape
    if sh > output_size or sw > output_size:
        raise ValueError(f"Seed {sw}x{sh} is larger than output {output_size}.")

    rng = np.random.default_rng(seed)
    weights = gaussian_weights(radius)
    centres = fully_known_centres(np.ones((sh, sw), bool), radius)
    db = PatchDB(source, centres, radius)

    out_p = np.zeros((output_size + 2 * radius, output_size + 2 * radius, ch),
                     dtype=np.float32)
    known_p = np.zeros(out_p.shape[:2], dtype=bool)
    top, left = (output_size - sh) // 2, (output_size - sw) // 2
    out_p[radius + top:radius + top + sh, radius + left:radius + left + sw] = source
    known_p[radius + top:radius + top + sh, radius + left:radius + left + sw] = True

    missing = ~known_p[radius:-radius, radius:-radius]
    grow_fill(out_p, known_p, missing, db, weights, radius, rng, "synth")
    return np.clip(out_p[radius:-radius, radius:-radius], 0, 255).astype(np.uint8)

def inpaint_texture(image, missing_mask, sample_texture=None, radius=PATCH_RADIUS,
                    seed=SEED):
    original = np.asarray(image, dtype=np.float32)
    missing = np.asarray(missing_mask, dtype=bool).copy()
    if missing.shape != original.shape[:2]:
        raise ValueError("missing_mask must match the image height and width.")

    rng = np.random.default_rng(seed)
    weights = gaussian_weights(radius)

    if sample_texture is not None:
        src = np.asarray(sample_texture, dtype=np.float64)
        h, w = src.shape[:2]
        centres = fully_known_centres(np.ones((h, w), bool), radius)
    else:
        src = original.astype(np.float64)
        centres = fully_known_centres(~missing, radius)
        if len(centres) == 0:
            raise ValueError("No fully known patches; supply sample_texture.")
        ys, xs = np.nonzero(missing)
        near = ((centres[:, 0] >= ys.min() - SEARCH_MARGIN) &
                (centres[:, 0] <= ys.max() + SEARCH_MARGIN) &
                (centres[:, 1] >= xs.min() - SEARCH_MARGIN) &
                (centres[:, 1] <= xs.max() + SEARCH_MARGIN))
        if near.sum() >= 1000:      
            centres = centres[near]
    if len(centres) > MAX_SOURCE_PATCHES:
        print(f"  subsampling source patches {len(centres)} -> {MAX_SOURCE_PATCHES}")
        centres = centres[rng.choice(len(centres), MAX_SOURCE_PATCHES, replace=False)]

    db = PatchDB(src, centres, radius)

    out_p = np.pad(original, ((radius, radius), (radius, radius), (0, 0)))
    known_p = np.pad(~missing, radius)
    grow_fill(out_p, known_p, missing, db, weights, radius, rng, "inpaint")
    return np.clip(out_p[radius:-radius, radius:-radius], 0, 255).astype(np.uint8)

def make_text_mask_interactive(image, save_path="outputs/lohagad_mask.png"):
    from matplotlib.widgets import RectangleSelector

    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    rectangles = []

    def add_rectangle(eclick, erelease):
        if None in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata):
            return
        x0, x1 = sorted((int(round(eclick.xdata)), int(round(erelease.xdata))))
        y0, y1 = sorted((int(round(eclick.ydata)), int(round(erelease.ydata))))
        x0, x1 = max(0, x0), min(width - 1, x1)
        y0, y1 = max(0, y0), min(height - 1, y1)
        if x1 <= x0 or y1 <= y0:
            return
        rectangles.append((x0, y0, x1, y1))
        mask[y0:y1 + 1, x0:x1 + 1] = 255
        print(f"Added text box: x={x0}:{x1}, y={y0}:{y1}")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_title("Drag a rectangle around each text region; close the window when done.")
    ax.axis("off")
    fig._selector = RectangleSelector(
        ax, add_rectangle, useblit=True, button=[1], minspanx=2, minspany=2,
        spancoords="pixels", interactive=True,
        props={"facecolor": "red", "edgecolor": "yellow", "alpha": 0.25, "fill": True},
    )
    plt.tight_layout()
    plt.show()

    if not rectangles:
        raise RuntimeError("No text boxes were drawn. Re-run and draw a rectangle.")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(save_path)
    print(f"Saved mask to {save_path}")
    return mask > 0

def show_side_by_side(title, images, labels, save_name=None):
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    axes = np.atleast_1d(axes)
    for ax, image, label in zip(axes, images, labels):
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    if save_name:
        fig.savefig(OUT_DIR / save_name, dpi=120)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

def run_texture_samples():
    paths = sorted(glob.glob("input/textureSample*.png"))
    if not paths:
        print("No textureSample*.png files found; skipping parts (a) and (b).")
        return
    for path in paths:
        print(f"[texture synthesis] {path}")
        sample = read_image(path)
        start = time.time()
        result = synthesize_texture(sample)
        elapsed = time.time() - start
        stem = Path(path).stem
        out_path = OUT_DIR / f"{stem}_synthesized_{OUT_SIZE}.png"
        save_image(out_path, result)
        show_side_by_side(f"{Path(path).name} | {elapsed:.1f}s", [sample, result],
                          ["Seed texture", f"Synthesized {OUT_SIZE} x {OUT_SIZE}"],
                          save_name=f"{stem}_comparison.png")
        print(f"Saved {out_path} ({elapsed:.1f}s)")


def run_lohagad_inpainting():
    path = Path("input/lohagad.png")
    if not path.exists():
        print("lohagad.png not found; skipping parts (c) and (d).")
        return

    image = read_image(path)
    mask_path = OUT_DIR / "lohagad_mask.png"
    if mask_path.exists():
        mask = np.asarray(
            Image.open(mask_path).convert("L")
        ) > 0
        print(f"Loaded existing mask: {mask_path}")
    else:
        print("Draw boxes over the text and its anti-aliased edges.")
        mask = make_text_mask_interactive(image, str(mask_path))

    sample_texture = None
    if USE_SAMPLE_TEXTURE_FOR_INPAINT:
        sample_paths = sorted(glob.glob("textureSample*.png"))
        if sample_paths:
            sample_texture = read_image(sample_paths[0])

    expanded = dilate(mask, 1)

    start = time.time()
    result = inpaint_texture(image, expanded, sample_texture=sample_texture)
    elapsed = time.time() - start

    with_hole = np.where(expanded[:, :, None], 255, image).astype(np.uint8)
    save_image(OUT_DIR / "lohagad_with_hole.png", with_hole)
    save_image(OUT_DIR / "lohagad_inpainted.png", result)
    show_side_by_side(f"Lohagad inpainting | {elapsed:.1f}s", [with_hole, result],
                      ["Input with hole (white = missing)", "Inpainted result"],
                      save_name="lohagad_comparison.png")
    print(f"Saved lohagad_with_hole.png and lohagad_inpainted.png ({elapsed:.1f}s)")


if __name__ == "__main__":
    run_texture_samples()
    run_lohagad_inpainting()