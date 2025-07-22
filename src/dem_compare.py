# Compare two sets of tifs resulting from surface characterization pipeline

import os
import argparse
import numpy as np

def load_npz(file_path):
    with np.load(file_path) as data:
        if "arr_0" in data:
            img = data["arr_0"]
        else:
            raise ValueError(f"No array found in {file_path}")
    if img.dtype != np.float64:
        img = img.astype(np.float64)
    return img

def compare_images(pre_img, post_img):
    if pre_img.shape != post_img.shape:
        raise ValueError("Image shapes do not match")

    valid_mask = ~np.isnan(pre_img) & ~np.isnan(post_img)
    diff = np.full_like(pre_img, np.nan)
    diff[valid_mask] = post_img[valid_mask] - pre_img[valid_mask]
    return diff

def diff_viz():
      # Edit this with correct values
      fig, axes = plt.subplots(2,2,figsize=(10,10))
      m1 = axes[0][0].imshow(np.rot90(slope_angle_array[:,::-1]), cmap='inferno')
      axes[0][0].set_title("Local Normal vs +Y (Slope)")
      fig.colorbar(m1, ax=axes[0][0])

      m2 = axes[0][1].imshow(np.rot90(count_array[:,::-1]), cmap='inferno')
      axes[0][1].set_title("# of Points Per Voxel")
      fig.colorbar(m2, ax=axes[0][1])

      m3 = axes[1][0].imshow(np.rot90(sig_array[:,::-1]), cmap='inferno')
      axes[1][0].set_title("Point Error (1 Sigma) Per Voxel")
      fig.colorbar(m3, ax=axes[1][0])

      m4 = axes[1][1].imshow(np.rot90(dem_array[:,::-1]), cmap='inferno')
      axes[1][1].set_title("Digital Elevation Map")
      fig.colorbar(m4, ax=axes[1][1])

      # Add compass rose (RH: removing for the moment until I can add proper rotation and flip of data
      # draw_compass_rose(fig, (0.91, 0.94), size=0.09)

      plt.tight_layout(rect=[0.95, 0.95, 0.9, 0.9])
      plt.show()

def main(pre_dir, post_dir, output_dir):
    if not os.path.isdir(pre_dir) or not os.path.isdir(post_dir):
        raise NotADirectoryError("One or both provided paths are not valid directories")

    os.makedirs(output_dir, exist_ok=True)

    pre_files = {f for f in os.listdir(pre_dir) if f.endswith(".npz")}
    post_files = {f for f in os.listdir(post_dir) if f.endswith(".npz")}
    common_files = pre_files.intersection(post_files)

    if not common_files:
        print("No matching .npz files found in both directories.")
        return

    for fname in sorted(common_files):
        pre_path = os.path.join(pre_dir, fname)
        post_path = os.path.join(post_dir, fname)

        try:
            pre_img = load_npz(pre_path)
            post_img = load_npz(post_path)
            diff_img = compare_images(pre_img, post_img)

            base_name = fname.replace(".npz", "")
            out_path = os.path.join(output_dir, f"{base_name}_diff.npz")
            np.savez_compressed(out_path, diff_img)
            print(f"[✓] Compared: {fname} -> Saved diff to {out_path}")
        except Exception as e:
            print(f"[!] Error processing {fname}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two folders of .npz DEM arrays (post - pre), ignoring NaNs.")
    parser.add_argument("--pre", required=True, help="Path to folder of 'pre' surface .npz files")
    parser.add_argument("--post", required=True, help="Path to folder of 'post' surface .npz files")
    parser.add_argument("--output", required=True, help="Directory to save output difference .npz files")
    args = parser.parse_args()
    main(args.pre, args.post, args.output)
