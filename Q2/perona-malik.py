import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def perona_malik_diffusion_rgb(img, num_iter=50, lam=0.1, K=10.0):
    I = img.astype(np.float64)
    
    for _ in range(num_iter):
        padded = np.pad(I, ((1, 1), (1, 1), (0, 0)), mode='edge')
        
        nabla_N = padded[:-2, 1:-1, :] - I
        nabla_S = padded[2:, 1:-1, :] - I
        nabla_W = padded[1:-1, :-2, :] - I
        nabla_E = padded[1:-1, 2:, :] - I
        
        grad_sq_N = np.sum(nabla_N ** 2, axis=2, keepdims=True)
        grad_sq_S = np.sum(nabla_S ** 2, axis=2, keepdims=True)
        grad_sq_W = np.sum(nabla_W ** 2, axis=2, keepdims=True)
        grad_sq_E = np.sum(nabla_E ** 2, axis=2, keepdims=True)
        
        c_N = 1.0 / (1.0 + grad_sq_N / (K ** 2))
        c_S = 1.0 / (1.0 + grad_sq_S / (K ** 2))
        c_W = 1.0 / (1.0 + grad_sq_W / (K ** 2))
        c_E = 1.0 / (1.0 + grad_sq_E / (K ** 2))
        
        diffusion_term = (c_N * nabla_N) + (c_S * nabla_S) + (c_W * nabla_W) + (c_E * nabla_E)
        I = I + lam * diffusion_term
        
    return np.clip(I, 0, 255)

if __name__ == "__main__":
    image_path = "racing.png"  
    img_pil = Image.open(image_path).convert("RGB")
    noisy_img = np.array(img_pil, dtype=np.float64)
    
    denoised_img = perona_malik_diffusion_rgb(
        noisy_img, 
        num_iter=50, 
        lam=0.15,       
        K=20.0         
    )
    
    
    denoised_pil = Image.fromarray(denoised_img.astype(np.uint8))
    denoised_pil.save("racing.png_denoised_image.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(noisy_img.astype(np.uint8))
    axes[0].set_title("Input Noisy Image")
    axes[0].axis("off")
    
    axes[1].imshow(denoised_img.astype(np.uint8))
    axes[1].set_title("Denoised Image")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig("sagarmaathaa.png_comparison.png", bbox_inches='tight', dpi=300)
    plt.show()
