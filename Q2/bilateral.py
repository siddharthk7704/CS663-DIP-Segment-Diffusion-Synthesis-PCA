import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def bilateral_filter_rgb(
    img,
    sigma_s=4.0,
    sigma_r=25.0,
    k=3,
    max_radius=15,
    block_size=256
):

    I = img.astype(np.float32)

    H, W, C = I.shape

    if C != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")

    radius = min(
        int(np.ceil(k * sigma_s)), max_radius)

    coords = np.arange(-radius, radius + 1, dtype=np.int32)

    dx, dy = np.meshgrid(coords, coords, indexing="ij")

    dx = dx.ravel()
    dy = dy.ravel()

    valid = (dx * dx + dy * dy) <= radius * radius

    dx = dx[valid]
    dy = dy[valid]

    N = len(dx)

    spatial_dist_sq = (
        dx.astype(np.float32) ** 2 +
        dy.astype(np.float32) ** 2
    )

    spatial_weights = np.exp(-spatial_dist_sq / (2.0 * sigma_s ** 2)).astype(np.float32)

    padded = np.pad(I, ((radius, radius),(radius, radius),(0, 0)), mode="edge")

    out = np.zeros_like(I)

    two_sigma_r_sq = np.float32(2.0 * sigma_r ** 2)

    for y0 in range(0, H, block_size):
        y1 = min(y0 + block_size, H)
        block_h = y1 - y0

        padded_block = padded[y0 : y1 + 2 * radius,:,:]

        neighbors = np.empty(
            (block_h, W, N, 3),
            dtype=np.float32)
        
        for n in range(N):
            x = dx[n]
            y = dy[n]

            neighbors[:, :, n, :] = padded_block[
                radius + y : radius + y + block_h,
                radius + x : radius + x + W, :
            ]

        center = I[ y0:y1, :, :][ :, :, None, :]

        diff = neighbors - center

        range_dist_sq = np.sum(diff * diff, axis=3)

        range_weights = np.exp(-range_dist_sq / two_sigma_r_sq).astype(np.float32)

        weights = (range_weights * spatial_weights[None, None, :])

        weighted_sum = np.sum(weights[:, :, :, None] * neighbors, axis=2)

        weight_sum = np.sum(weights, axis=2)

        block_output = (weighted_sum / weight_sum[:, :, None])

        out[y0:y1,:,:] = block_output

    return np.clip(np.rint(out), 0,255).astype(np.uint8)

if __name__ == "__main__":
    image_path = "sagarmaathaa.png"  
    img_pil = Image.open(image_path).convert("RGB")
    noisy_img = np.array(img_pil)
    
    denoised_img = bilateral_filter_rgb(
        noisy_img, 
        sigma_s=10.0, 
        sigma_r=30.0  
    )
    
    denoised_pil = Image.fromarray(denoised_img)
    denoised_pil.save("sagarmaathaa.png_denoised_bilateral_output.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(noisy_img.astype(np.uint8))
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(denoised_img.astype(np.uint8))
    axes[1].set_title("Bilateral Filter Denoised")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig("sagarmaathaa.png_bilateral_comparison.png", bbox_inches='tight', dpi=300)
    plt.show()
