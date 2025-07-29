# Compare two sets of tifs resulting from surface characterization pipeline

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_npz(file_path):
    with np.load(file_path) as data:
        if "arr_0" in data:
            img = data["arr_0"]
        else:
            raise ValueError(f"No array found in {file_path}")
    if img.dtype != np.float64:
        img = img.astype(np.float64)
    return img

def compare_images(pre_img, post_img, crop=True):
    if (not crop) and (pre_img.shape != post_img.shape):
        raise ValueError(f"Image shapes do not match: {pre_img.shape=} != {post_img.shape=}")
    else:
        # Crop the larger image to the size of the other for each dimension
        min_rows = np.min([pre_img.shape[0], post_img.shape[0]])
        min_cols = np.min([pre_img.shape[1], post_img.shape[1]])
        pre_img = pre_img.copy()[:min_rows, :min_cols]
        post_img = post_img.copy()[:min_rows, :min_cols]

    valid_mask = ~np.isnan(pre_img) & ~np.isnan(post_img)
    diff = np.full_like(pre_img, np.nan)
    diff[valid_mask] = post_img[valid_mask] - pre_img[valid_mask]
    return diff

def diff_viz(diff_img, name):
      # Edit this with correct values
#      fig, axes = plt.subplots(2,1,figsize=(10,10))
    #   m1 = axes[0][0].imshow(np.rot90(slope_angle_array[:,::-1])

#      m1 = axes[0].imshow(dem[:,::-1], cmap='inferno')
#      axes[0].set_title("Digital Elevation Map Difference")
#      fig.colorbar(m1, ax=axes[0])

#      m2 = axes[1].imshow(slope[:,::-1], cmap='inferno')
#      axes[1].set_title("Digital Elevation Map Difference")
#      fig.colorbar(m2, ax=axes[1])

      plt.figure(figsize=(4,4), dpi=250)

      # Currently optimized for DEM viz....
      ax = plt.imshow(diff_img[:,::-1], cmap='inferno', vmin=-0.01, vmax=0.01, origin='lower')
      plt.title(name)
      plt.colorbar(ax, label=r"$\Delta$ Z [m]")
      plt.ylabel("Voxel ID in Y")
      plt.xlabel("Voxel ID in X")

      # Add compass rose (RH: removing for the moment until I can add proper rotation and flip of data
      # draw_compass_rose(fig, (0.91, 0.94), size=0.09)

    #   plt.tight_layout(rect=[0.95, 0.95, 0.9, 0.9])
      plt.show()

def main(pre_dir, post_dir, output_dir):
    if not os.path.isdir(pre_dir):
        raise NotADirectoryError(f"The provided 'pre' path {pre_dir} is not a valid directory")
    if not os.path.isdir(post_dir):
        raise NotADirectoryError(f"The provided 'post' path {post_dir} is not a valid directory")

    if output_dir:
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
            diff_img = compare_images(pre_img, post_img, crop=True)

            base_name = fname.replace(".npz", "")
            diff_viz(diff_img, "Difference "+base_name.upper()+" (After - Before)")

            if output_dir:
                out_path = os.path.join(output_dir, f"{base_name}_diff.npz")
                np.savez_compressed(out_path, diff_img)

            print(f"[✓] Compared: {fname} Before and After Trial")

        except Exception as e:
            print(f"[!] Error processing {fname}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two folders of .npz DEM arrays (post - pre), ignoring NaNs.")
    parser.add_argument("--pre", required=True, help="Path to folder of 'pre' surface .npz files")
    parser.add_argument("--post", required=True, help="Path to folder of 'post' surface .npz files")
    parser.add_argument("--output", required=False, help="Directory to save output difference .npz files", default=None)
    args = parser.parse_args()
    main(args.pre, args.post, args.output)

