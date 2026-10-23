import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DATASET_DIR = "ATT_Faces_Dataset"
PCA_FILE = "pca_model_artifacts.npz"

NUM_SUBJECTS = 40
IMAGES_PER_SUBJECT = 10

TRAIN_SUBJECTS = 30
TRAIN_IMAGES_PER_SUBJECT = 8

K = 10

FACE_THRESHOLD = 5500
MATCH_THRESHOLD = 1200.0

SHOW_PLOTS = True
SAVE_RESULTS = True

def load_image(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64).flatten()


def find_image_file(subject_dir, image_number):
    for filename in os.listdir(subject_dir):
        stem, _ = os.path.splitext(filename)
        match = re.match(r"^(\d+)(?:\D|$)", stem)
        if match and int(match.group(1)) == image_number:
            return os.path.join(subject_dir, filename)
    return None


def load_all_images(dataset_dir=DATASET_DIR):
    images, subjects, image_numbers = [], [], []

    for subject in range(1, NUM_SUBJECTS + 1):
        folder = os.path.join(dataset_dir, f"s{subject}")
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing subject folder: {folder}")

        for image_number in range(1, IMAGES_PER_SUBJECT + 1):
            path = find_image_file(folder, image_number)
            if path is None:
                raise FileNotFoundError(
                    f"Cannot find image {image_number} in {folder}. "
                    f"Files found: {os.listdir(folder)}"
                )

            images.append(load_image(path))
            subjects.append(subject)
            image_numbers.append(image_number)

    return np.asarray(images), np.asarray(subjects), np.asarray(image_numbers)

def load_training_images(dataset_dir=DATASET_DIR):
    images, labels = [], []

    for subject in range(1, TRAIN_SUBJECTS + 1):
        folder = os.path.join(dataset_dir, f"s{subject}")
        for image_number in range(1, TRAIN_IMAGES_PER_SUBJECT + 1):
            path = find_image_file(folder, image_number)
            if path is None:
                raise FileNotFoundError(
                    f"Cannot find image {image_number} in {folder}"
                )
            images.append(load_image(path))
            labels.append(subject)

    return np.asarray(images), np.asarray(labels)

def build_training_gallery(train_images, train_labels, mean_face, eigenfaces):
    centered = train_images - mean_face
    gallery_weights = centered @ eigenfaces.T
    return gallery_weights, train_labels

def compute_scores(image, mean_face, eigenfaces, gallery_weights):
    phi = image - mean_face
    weights = eigenfaces @ phi

    reconstruction = weights @ eigenfaces
    residual = phi - reconstruction
    distance_from_face_space = float(np.linalg.norm(residual))

    distances = np.linalg.norm(gallery_weights - weights, axis=1)
    nearest_index = int(np.argmin(distances))
    nearest_distance = float(distances[nearest_index])

    return max(0.0, distance_from_face_space), nearest_distance, nearest_index

def compute_all_scores(images, mean_face, eigenfaces, gallery_weights):
    M1 = np.zeros(len(images), dtype=float)
    M2 = np.zeros(len(images), dtype=float)
    nearest_indices = np.zeros(len(images), dtype=int)

    for i, image in enumerate(images):
        M1[i], M2[i], nearest_indices[i] = compute_scores(
            image, mean_face, eigenfaces, gallery_weights
        )

    return M1, M2, nearest_indices

def make_decisions(M1, M2, nearest_subjects,
                   face_threshold, match_threshold):
    decisions = np.full(len(M1), -1, dtype=int)

    face_like = M1 <= face_threshold
    decisions[face_like] = 0

    recognized = face_like & (M2 <= match_threshold)
    decisions[recognized] = nearest_subjects[recognized]

    return decisions

def plot_matrix(matrix, title, colorbar_label, filename):
    plt.figure(figsize=(10, 7))
    plt.imshow(matrix, cmap="jet", aspect="auto", interpolation="nearest")
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.xlabel("Image number")
    plt.ylabel("Subject number")
    plt.xticks(np.arange(IMAGES_PER_SUBJECT), np.arange(1, IMAGES_PER_SUBJECT + 1))
    plt.yticks(np.arange(NUM_SUBJECTS), np.arange(1, NUM_SUBJECTS + 1))
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    print(f"Loading PCA model from {PCA_FILE}...")
    if not os.path.isfile(PCA_FILE):
        raise FileNotFoundError(
            f"{PCA_FILE} not found. Run Part (a) first in this folder."
        )

    artifacts = np.load(PCA_FILE)
    mean_face = artifacts["mean_face"]
    eigenfaces = artifacts["eigenfaces"]
    singular_values = artifacts["singular_values"]

    if eigenfaces.shape[0] != K:
        raise ValueError(
            f"K={K}, but the saved PCA model has {eigenfaces.shape[0]} "
            "eigenfaces. Set K to match Part (a), or regenerate its artifact."
        )

    print("Loading dataset images...")
    X, subjects, image_numbers = load_all_images()
    print("Loaded images:", len(X))

    train_mask = (
        (subjects <= TRAIN_SUBJECTS) &
        (image_numbers <= TRAIN_IMAGES_PER_SUBJECT)
    )
    test_mask = ~train_mask
    print("Training images:", int(train_mask.sum()))
    print("Test images:", int(test_mask.sum()))

    train_images, train_labels = load_training_images()
    if len(train_images) != 240:
        raise ValueError(f"Expected 240 training images, got {len(train_images)}")

    gallery_weights, gallery_subjects = build_training_gallery(
        train_images, train_labels, mean_face, eigenfaces
    )

    print("Computing M1 and M2 for all images...")
    M1_all, M2_all, nearest_indices = compute_all_scores(
        X, mean_face, eigenfaces, gallery_weights
    )
    
    nearest_subjects = gallery_subjects[nearest_indices]

    M1 = M1_all.reshape(NUM_SUBJECTS, IMAGES_PER_SUBJECT)
    M2 = M2_all.reshape(NUM_SUBJECTS, IMAGES_PER_SUBJECT)

    M3_all = make_decisions(
        M1_all, M2_all, nearest_subjects,
        FACE_THRESHOLD, MATCH_THRESHOLD
    )
    M3 = M3_all.reshape(NUM_SUBJECTS, IMAGES_PER_SUBJECT)

    true_test = np.where(subjects[test_mask] <= TRAIN_SUBJECTS,
                         subjects[test_mask], 0)
    predictions_test = M3_all[test_mask]
    accuracy = 100.0 * np.mean(predictions_test == true_test)

    print("\nManually supplied parameters")
    print("K:", K)
    print("FACE_THRESHOLD:", FACE_THRESHOLD)
    print("MATCH_THRESHOLD:", MATCH_THRESHOLD)
    print(f"Test decision accuracy: {accuracy:.2f}%")
    print("\nTest-set decision counts:")
    print("Total:", len(true_test))
    print("Correct:", int(np.sum(predictions_test == true_test)))
    print("Incorrect:", int(np.sum(predictions_test != true_test)))
    print("Output -1:", int(np.sum(predictions_test == -1)))
    print("Output 0:", int(np.sum(predictions_test == 0)))
    print("Output 1..30:", int(np.sum(predictions_test > 0)))
    
    known_test_mask = test_mask & (subjects <= TRAIN_SUBJECTS)
    unknown_test_mask = test_mask & (subjects > TRAIN_SUBJECTS)

    known_predictions = M3_all[known_test_mask]
    known_actual = subjects[known_test_mask]

    unknown_predictions = M3_all[unknown_test_mask]
    unknown_actual = np.zeros(np.sum(unknown_test_mask), dtype=int)

    known_accuracy = 100.0 * np.mean(
        known_predictions == known_actual
    )

    unknown_accuracy = 100.0 * np.mean(
        unknown_predictions == unknown_actual
    )

    print("\nSeparate Test Accuracies:")
    print(f"Known-person recognition: {known_accuracy:.2f}%")
    print(f"Unknown-person rejection: {unknown_accuracy:.2f}%")


    if SAVE_RESULTS:
        np.savez(
            "face_recognition_results.npz",
            M1=M1,
            M2=M2,
            M3=M3,
            face_threshold=FACE_THRESHOLD,
            match_threshold=MATCH_THRESHOLD,
            K=K,
            test_accuracy=accuracy
        )
        print("\nSaved matrices and parameters to face_recognition_results.npz")

    plot_matrix(M1, "M1: Euclidean Distance from Face Space",
                "Squared distance", "M1.png")
    plot_matrix(M2, "M2: Nearest Training-Projection Distance",
                "Euclidean distance", "M2.png")
    plot_matrix(M3, "M3: Final Recognition Decisions",
                "Decision output (-1, 0, subject ID)", "M3.png")
