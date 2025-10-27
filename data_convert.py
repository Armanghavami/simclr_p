import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from PIL import Image
from tqdm import tqdm

# Directories
brain_dir = "/Users/arman/Documents/machin learning/simclr_brain /brain_dataset"
output_dir = "middle_slices_fast"
os.makedirs(output_dir, exist_ok=True)

# Target 3D volume size for downsampling
target_shape = (128, 128, 128)

def preprocess_and_extract_slice(nii_path, target_shape=target_shape):
    # Load the 3D MRI
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)

    # Optional: simple skull mask (remove background)
    data[data < 1e-3] = 0

    # Resize to target shape for faster processing
    factors = [target_shape[i]/data.shape[i] for i in range(3)]
    data_resized = zoom(data, factors, order=1)  # trilinear interpolation

    # Pick the slice with maximum brain tissue along axial axis
    z_sums = data_resized.sum(axis=(0,1))
    best_z = np.argmax(z_sums)
    slice_data = data_resized[:, :, best_z]

    # Normalize slice to 0-255
    brain_mask = slice_data > 0
    if np.any(brain_mask):
        low, high = np.percentile(slice_data[brain_mask], (1, 99))
        slice_data = np.clip(slice_data, low, high)
        slice_data = (slice_data - low) / (high - low + 1e-8)
    slice_data = (slice_data * 255).astype(np.uint8)

    return slice_data

# Process all subjects
subjects = [sub for sub in os.listdir(brain_dir) if os.path.exists(os.path.join(brain_dir, sub, "anat"))]

for sub in tqdm(subjects):
    anat_path = os.path.join(brain_dir, sub, "anat")
    nii_files = [f for f in os.listdir(anat_path) if f.endswith(".nii") or f.endswith(".nii.gz")]
    if not nii_files:
        continue

    nii_path = os.path.join(anat_path, nii_files[0])
    try:
        slice_img = preprocess_and_extract_slice(nii_path)
    except Exception as e:
        print(f"Error processing {sub}: {e}")
        continue

    # Save as PNG
    base_name = f"{sub}_{nii_files[0].replace('.nii.gz','').replace('.nii','')}"
    png_path = os.path.join(output_dir, f"{base_name}_slice.png")
    Image.fromarray(slice_img).save(png_path)
