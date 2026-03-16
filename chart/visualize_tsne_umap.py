"""
t-SNE and UMAP visualization using chart/sample.png.
- Single image: splits into patches, extracts features per patch.
- Categorizes patches by brightness: Dark (Digits) vs Light (Background/Noise).
- Runs t-SNE/UMAP to evaluate data separability.

Run from project root: python chart/visualize_tsne_umap.py
Requirements: numpy, matplotlib, opencv-python, scikit-learn, umap-learn
"""
from pathlib import Path
import numpy as np
import cv2

# Path Setup
CHART_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGE = CHART_DIR / "sample.png"
PATCH_SIZE = 16  
OUTPUT_TSNE = CHART_DIR / "tsne.png"
OUTPUT_UMAP = CHART_DIR / "umap.png"

def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def image_to_patch_vectors(gray, patch_size=PATCH_SIZE):
    h, w = gray.shape[:2]
    h2 = (h // patch_size) * patch_size
    w2 = (w // patch_size) * patch_size
    if h2 == 0 or w2 == 0:
        return np.empty((0, patch_size * patch_size), dtype=np.float32)
    
    patches = []
    for y in range(0, h2, patch_size):
        for x in range(0, w2, patch_size):
            patch = gray[y : y + patch_size, x : x + patch_size]
            # Normalize pixels to [0, 1]
            patches.append(patch.ravel().astype(np.float32) / 255.0)
    return np.array(patches)

def main():
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    if not SAMPLE_IMAGE.exists():
        print(f"Error: Put sample.png in {CHART_DIR} and run again.")
        return

    print(f"Processing: {SAMPLE_IMAGE.name}")
    gray = load_image(SAMPLE_IMAGE)
    X = image_to_patch_vectors(gray, PATCH_SIZE)

    if X.shape[0] < 2:
        print("Image too small for analysis.")
        return

    # --- COLOR LOGIC: PROXY FOR EVALUATION ---
    # Calculate mean brightness per patch. Dark < 0.5 (Digit), Light > 0.5 (Background)
    patch_means = np.mean(X, axis=1)
    # Define colors: Dark Red for digits, Light Grey for background
    colors = np.where(patch_means < 0.5, '#8B0000', '#D3D3D3')
    
    # Imports for Visualization
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("Required: pip install matplotlib scikit-learn")
        return

    # 1. RUN t-SNE
    print("Evaluating with t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0] - 1))
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 7))
    # Plot background first, then digits on top
    for c in ['#D3D3D3', '#8B0000']:
        mask = (colors == c)
        label = "Background/Noise" if c == '#D3D3D3' else "Digit Segment"
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=c, label=label, alpha=0.7, s=40, edgecolors='w', linewidth=0.5)
    
    plt.title("t-SNE: Data Separability Evaluation (Patch-based)")
    plt.xlabel("t-SNE Feature 1")
    plt.ylabel("t-SNE Feature 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(OUTPUT_TSNE, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUTPUT_TSNE}")

    # 2. RUN UMAP
    try:
        import umap
        print("Evaluating with UMAP...")
        n_neighbors = min(15, X.shape[0] - 1)
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
        X_umap = reducer.fit_transform(X)

        plt.figure(figsize=(10, 7))
        for c in ['#D3D3D3', '#8B0000']:
            mask = (colors == c)
            label = "Background/Noise" if c == '#D3D3D3' else "Digit Segment"
            plt.scatter(X_umap[mask, 0], X_umap[mask, 1], c=c, label=label, alpha=0.7, s=40, edgecolors='w', linewidth=0.5)
        
        plt.title("UMAP: Latent Space Topology (Patch-based)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(OUTPUT_UMAP, dpi=200, bbox_inches="tight")
        print(f"Saved: {OUTPUT_UMAP}")
    except ImportError:
        print("UMAP skipped. Optional: pip install umap-learn")

if __name__ == "__main__":
    main()