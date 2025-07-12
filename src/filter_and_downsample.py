import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import sys

# VOXEL_SIZE = 0.01 # 1 cm to start?
# COLORZ = True

if __name__=="__main__":
   fname = sys.argv[1]
   outpath = sys.argv[2]
   VOXEL_SIZE = float(sys.argv[3])

   # bdownsample = True if sys.argv[2].lower() in ["--downsample", "-ds"] else False
   # bsave = True if (len(sys.argv) > 3) and (sys.argv[3].lower() in ["--save"]) else False

   pc = o3d.io.read_point_cloud(fname)

   # Remove low density points
   print("Statistical oulier removal")
   pcds, ind = pc.remove_statistical_outlier(nb_neighbors=20,
                                            std_ratio=2.5)

   # Downsample point cloud using voxel downsampling to regular grid
   print(f"Downsample the point cloud with a voxel of {VOXEL_SIZE}")
   pcds = pcds.voxel_down_sample(voxel_size=VOXEL_SIZE)

   # Visualize
#   # Optionally adjust color by Z value and display a colorbar?
#   # Extract Z-values and normalize
   z_values = np.asarray(pcds.points)[:, 2]

   plt.hist(z_values.flat, bins=100)
   plt.show()

#
#   z_min, z_max = np.min(z_values), np.max(z_values)
#   normalized_z = (z_values - z_min) / (z_max - z_min)
#
#   # 3. Create a colormap and apply it
#   colormap_points = [
#       o3d.visualization.rendering.Gradient.Point(0.0, [0.0, 0.0, 1.0, 1.0]),  # Blue at min Z
#       o3d.visualization.rendering.Gradient.Point(0.5, [0.0, 1.0, 0.0, 1.0]),  # Green in middle
#       o3d.visualization.rendering.Gradient.Point(1.0, [1.0, 0.0, 0.0, 1.0])   # Red at max Z
#   ]
#   gradient = o3d.visualization.rendering.Gradient(colormap_points)
#
#   # Assign colors based on normalized Z-values
#   colors = np.array([gradient.get_color_at(val)[:3] for val in normalized_z])
#   pcds.colors = o3d.utility.Vector3dVector(colors)

   o3d.visualization.draw_geometries([pcds])

   # Save to output path
   o3d.io.write_point_cloud(outpath, pcds) # ,compressed=True) # Might want to optionally compress in the future...
