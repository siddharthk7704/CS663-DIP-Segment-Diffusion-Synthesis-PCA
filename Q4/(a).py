import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_att_faces(dataset_dir="ATT_Faces_Dataset", num_subjects=30, num_images_per_subject=8):
    faces = []
    
    for i in range(1, num_subjects + 1):
        subject_dir = os.path.join(dataset_dir, f"s{i}")
        for j in range(1, num_images_per_subject + 1):
            img_path = os.path.join(subject_dir, f"{j}.pgm")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("L")
                img_arr = np.array(img, dtype=np.float64).flatten()
                faces.append(img_arr)
                
    return np.array(faces)

def compute_pca_eigenfaces(data, k=10):
    mean_face = np.mean(data, axis=0)
    centered_data = data - mean_face
    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
    eigenfaces = Vt[:k, :]
    
    return mean_face, eigenfaces, S[:k], U

if __name__ == "__main__":
    print("Loading AT&T Faces training images...")
    face_matrix = load_att_faces(dataset_dir="ATT_Faces_Dataset", num_subjects=30, num_images_per_subject=8)
    print("Loaded images:", face_matrix.shape[0])
    assert face_matrix.shape[0] == 240, \
        f"Expected 240 images, got {face_matrix.shape[0]}"
    
    H, W = 112, 92 # AT&T image dimensions
    
    print("Computing Mean and Principal Modes (PCA)...")
    mean_face, eigenfaces, singular_values, U = compute_pca_eigenfaces(face_matrix, k=10)
    
    np.savez("pca_model_artifacts.npz", 
             mean_face=mean_face, 
             eigenfaces=eigenfaces, 
             singular_values=singular_values)
    print("Saved PCA artifacts to 'pca_model_artifacts.npz'.")
    
    mean_img_2d = mean_face.reshape((H, W))
    plt.figure(figsize=(4, 5))
    plt.imshow(mean_img_2d, cmap='gray')
    plt.title("Mean Face Image")
    plt.axis("off")
    plt.savefig("mean_face.png", bbox_inches='tight')
    plt.show()
    
    fig, axes = plt.subplots(10, 3, figsize=(9, 25))

    N = face_matrix.shape[0]
    k = 3.0

    for i in range(10):
        v_i = eigenfaces[i]
        lambda_i = singular_values[i]**2 / N

        img_minus = mean_face - k * np.sqrt(lambda_i) * v_i
        img_plus  = mean_face + k * np.sqrt(lambda_i) * v_i

        img_minus = np.clip(img_minus.reshape(H, W), 0, 255)
        img_plus  = np.clip(img_plus.reshape(H, W), 0, 255)

        axes[i, 0].imshow(img_minus, cmap='gray', vmin=0, vmax=255)
        axes[i, 0].set_title(f"Mode {i+1}: Mean - α")
        axes[i, 0].axis("off")

        ef = v_i.reshape(H, W)
        ef_display = (ef - ef.min()) / (ef.max() - ef.min())
        axes[i, 1].imshow(ef_display, cmap='gray')
        axes[i, 1].set_title(f"Eigenface {i+1}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img_plus, cmap='gray', vmin=0, vmax=255)
        axes[i, 2].set_title(f"Mode {i+1}: Mean + α")
        axes[i, 2].axis("off")
        
    plt.tight_layout()
    plt.savefig("eigenfaces_modes_variation_fixed.png", bbox_inches='tight', dpi=300)
    plt.show()
    print("Execution complete. Visualizations saved.")
