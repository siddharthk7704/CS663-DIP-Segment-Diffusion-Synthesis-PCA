import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return np.asarray(image, dtype=np.float64)
'''
def calculate_weights(current, features, hs, hr):
    x, y = current[:2]
    r, g, b = current[2:]
    spatial_features = features[:, :2]
    rgb_features = features[:, 2:]
    ds = (spatial_features[:, 0]-x)**2 + (spatial_features[:, 1]-y)**2
    dr = (rgb_features[:, 0]-r)**2 + (rgb_features[:, 1]-g)**2 + (rgb_features[:, 2]-b)**2
    ws = np.exp(-ds / (2 * hs**2))
    wr = np.exp(-dr / (2 * hr**2))
    return ws * wr
'''
def feature_representation(image):
    h, w, _ = image.shape
    yy, xx = np.mgrid[0:h, 0:w]
    return np.column_stack([xx.ravel(), yy.ravel(), image.reshape(-1, 3)])

def mean_shift(image, hs, hr, n_samples=1500, merge_thr=0.6,
                      min_shift=1e-2, max_iter=50, seed=0):
    feats = feature_representation(image)
    scale = np.array([hs, hs, hr, hr, hr], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(feats), min(n_samples, len(feats)), replace=False)
    S = feats[idx] / scale                      # KDE support (subset)
    M = S.copy()                                # seeds
    for _ in range(max_iter):
        d2 = ((M[:, None, :] - S[None, :, :]) ** 2).sum(-1)
        w = np.exp(-0.5 * d2)
        new = (w @ S) / w.sum(1, keepdims=True)
        done = np.abs(new - M).max() < min_shift
        M = new
        if done: break
    # merge converged modes
    modes = []
    for m in M:
        if not any(np.linalg.norm(m - c) < merge_thr for c in modes):
            modes.append(m)
    modes = np.array(modes)
    # assign every pixel to nearest mode
    P = feats / scale
    d2 = ((P[:, None, :] - modes[None, :, :]) ** 2).sum(-1)
    labels = d2.argmin(1).reshape(image.shape[:2]) + 1
    mode_rgb = modes[:, 2:] * hr
    seg = np.clip(mode_rgb[labels - 1], 0, 255).astype(np.uint8)
    return labels, seg

def create_segment_boundaries(image, labels):

    result = np.asarray(image, dtype=np.uint8).copy()

    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundaries[1:, :] |= labels[1:, :] != labels[:-1, :]

    smooth = gaussian_filter(
        boundaries.astype(np.float32), sigma=0.6
    ) > 0.25

    from scipy.ndimage import distance_transform_edt

    distance = distance_transform_edt(~smooth)
    thick = distance <= 1.0

    result[thick] = [0, 0, 0]

    return result

def save_outputs(image, segmented_image, labels, image_path, out_dir, alpha=0.3):
    os.makedirs(out_dir, exist_ok=True)
    original_u8 = image.astype(np.uint8)
    Image.fromarray(segmented_image).save(os.path.join(out_dir, "segmented.png"))
    Image.fromarray(original_u8).save(os.path.join(out_dir, "original.png"))
    np.save(os.path.join(out_dir, "labels.npy"), labels)

    overlay = np.clip(
        alpha * segmented_image.astype(np.float32)
        + (1 - alpha) * original_u8.astype(np.float32),
        0, 255
    ).astype(np.uint8)
    Image.fromarray(overlay).save(os.path.join(out_dir, "overlay.png"))
    print(f"Saved outputs to '{out_dir}/'")
    return overlay

def main(image_path, hs, hr):
    image = load_image(image_path)

    labels, segmented_image = mean_shift(image, hs, hr)
    boundaries = create_segment_boundaries(image, labels)
    output_dir = os.path.join(
    "outputs", "part1",
    os.path.splitext(os.path.basename(image_path))[0]
)
    save_outputs(image, segmented_image, labels, image_path, output_dir)
    num_segments = len(np.unique(labels))
    Image.fromarray(boundaries).save(os.path.join(output_dir, "boundaries.png"))
    print("Number of segments:", num_segments)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image.astype(np.uint8))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(labels, cmap="tab10")
    plt.title("Label Map")
    plt.axis("off")

    plt.suptitle(
        f"Mean Shift Comparison (hs={hs}, hr={hr}, segments={num_segments})"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    plt.savefig(
        os.path.join(output_dir, "input_and_segmentation.png"),
        bbox_inches="tight",
        dpi=150
    )
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.imshow(boundaries)
    plt.title("Segment Boundaries on Input")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boundaries_display.png"),
                bbox_inches="tight", dpi=150)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python meanshift.py <image_path> hs hr")
        sys.exit(1)
    main(sys.argv[1], hs=int(sys.argv[2]), hr=int(sys.argv[3]))