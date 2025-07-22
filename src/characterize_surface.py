# This module should perform a simple linear planar regression and output
# 	the best fit plane parameters and the RMSE to characterize the residual

# I'm sure we could do this natively in Open3D, but worst case pass points off to sklearn or even numpy

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

def draw_compass_rose(fig, pos, size=0.1):
   """Creates a compass rose on the given figure.

   Args:
       fig: The Matplotlib Figure object.
       pos: A tuple (x, y) specifying the center of the compass rose in figure coordinates.
       size: The relative size of the compass rose.
   """
   # Create a new axes for the compass rose
   ax_compass = fig.add_axes([pos[0] - size/2, pos[1] - size/2, size, size], polar=True)
   ax_compass.set_theta_direction(-1)  # Clockwise direction
   ax_compass.set_theta_zero_location("N") # North at top

   # Add directional labels
   angles = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
   labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
   ax_compass.set_xticks(angles)
   ax_compass.set_xticklabels(labels)
   ax_compass.set_yticklabels([]) # Hide y-axis labels
   ax_compass.grid(False) # Hide grid

   # Optional: Draw the "arms" of the compass rose
   ax_compass.plot([0, np.pi/2], [0, 1], color="black", lw=2)
   ax_compass.plot([np.pi/2, np.pi], [0, 1], color="black", lw=2)
   ax_compass.plot([np.pi, 3*np.pi/2], [0, 1], color="black", lw=2)
   ax_compass.plot([3*np.pi/2, 2*np.pi], [0, 1], color="black", lw=2)

   return ax_compass

def fit_plane_pca(points):
   """Fit a plane to the given points using PCA."""
   centroid = np.mean(points, axis=0)
   cov = np.cov(points.T)
   eigvals, eigvecs = np.linalg.eigh(cov)
   normal = eigvecs[:, 0]  # eigenvector with smallest eigenvalue
   return centroid, normal, cov

def voxelize_and_analyze(pcd, voxel_size):
   # Create voxel grid
   voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

   # Map voxel index to list of points
   voxel_point_map = defaultdict(list)

   print("Building voxel grid point associations...")
   for point in tqdm(np.asarray(pcd.points)):
      voxel_idx = tuple(((point[[0,2]] - voxel_grid.origin[[0,2]]) / voxel_size).astype(int))
      voxel_point_map[voxel_idx].append(point)

   voxel_data = []

   print("Computing voxel grid statistics...")
   for voxel_idx, points in tqdm(voxel_point_map.items()):
      points = np.array(points)
      if len(points) < 3:
         continue  # Not enough points to fit a plane

      centroid, normal, cov = fit_plane_pca(points)
      density = len(points) / (voxel_size ** 3)

      voxel_data.append({
         "voxel_index": voxel_idx,
         "centroid": centroid,
         "normal": normal,
         "covariance": cov,
         "var": np.trace(cov),  # "var": np.linalg.det(cov),
         "density": density,
         "num_points": len(points),
      })

   return voxel_data, np.max(list(voxel_point_map.keys()), axis=0) + 1


if __name__=="__main__":
   fname = sys.argv[1]
   MIN_Q = float(sys.argv[2])
   MAX_Q = float(sys.argv[3])
   MODE = sys.argv[4].lower()
   bag_dir = os.path.expanduser(os.path.dirname(fname))

   pcd = o3d.io.read_point_cloud(fname)

   # Do one last colormap filtering step to condense Z values to smaller range
   pts = np.asarray(pcd.points)
   mask = (pts[:,2] > np.quantile(pts[:,2], MIN_Q)) & (pts[:,2] < np.quantile(pts[:,2], MAX_Q))
#
#      filtered_pcd = o3d.geometry.PointCloud()
#      filtered_pcd.points = o3d.utility.Vector3dVector(pts[mask])
#      filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])

   # See if we can drop some points in place
   pcd = pcd.select_by_index(np.where(mask)[0])

   if MODE == "--global":
      # BEGIN GLOBAL FIT EVALUATION:

      # not sure if this is the best way. I need a way of returning errors
      # I suppose I can just compute residuals using points - plane equations
      plane_model, inliers = pcd.segment_plane(distance_threshold=0.005,
                                         ransac_n=5,
                                         num_iterations=5000)
      [A, B, C, D] = plane_model
      print(f"Plane equation: {A:.5f}x + {B:.5f}y + {C:.5f}z + {D:.5f} = 0")

      # Compute residuals and get statistics on those, such as RMSE:
      pts = np.asarray(pcd.points)
      residuals = (A * pts[:,0] + B * pts[:,1] + C * pts[:,2] + D) / np.sqrt(A**2 + B**2 + C**2)

      # Residual distribution
      plt.hist(residuals, bins=100)
      plt.show()

      # Z value distribution
      plt.hist(pts[:,1], bins=500)
      plt.show()

      RMSE = np.sqrt(np.sum(residuals**2) / len(residuals))
      print(f"Planar fit has Global RMSE = {RMSE:.8f} m")

      inlier_cloud = pcd.select_by_index(inliers)
      inlier_cloud.paint_uniform_color([1.0, 0, 0])
      outlier_cloud = pcd.select_by_index(inliers, invert=True)
      o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


   else:
      # BEGIN LOCAL PLANE FIT USING VOXEL GRID TRACE WITH 5CM VOXELS OR MASKING

      # Analyze voxel grid
      voxel_size = 0.05   # 10cm so we only get one layer of voxels
      voxel_results, voxel_grid_size = voxelize_and_analyze(pcd, voxel_size)

      # Print or use results
#      for v in voxel_results[:5]:
#         print(f"Voxel {v['voxel_index']} | Density: {v['num_points']:.2f}")
#         print(f"Plane normal: {v['normal']}")
#         print(f"Covariance matrix:\n{v['covariance']}\n")

      Y_UP = np.asarray([0,1,0]) # This will be read in from optitrack pose average over capture

      slope_angle_array = np.full(voxel_grid_size, np.nan)
      count_array = np.full_like(slope_angle_array, np.nan)
      sig_array = np.full_like(slope_angle_array, np.nan)
      dem_array = np.full_like(slope_angle_array, np.nan)

      print("Constructing visualization of voxel statistics...")
      for v in tqdm(voxel_results):
         slope_angle_array[v['voxel_index']] = np.rad2deg(np.arccos(np.abs(np.dot(v['normal'], Y_UP))))
         count_array[v['voxel_index']] = v['num_points']
         sig_array[v['voxel_index']] = np.sqrt(v['var'])
         dem_array[v['voxel_index']] = v['centroid'][1]

      print("Done.")

      print("Writing out NPZ files of all dem statistics...")

      # RH: NOTE!!! CV2 DOES NOT HANDLE WRITING NAN VALUES PROPERLY!!!! SO USE EITHER TIFFILE OR NUMPY NPZ FORMAT TO SAVE DATA
      np.savez_compressed(os.path.join(bag_dir,'slope_angle.npz'), slope_angle_array)
      np.savez_compressed(os.path.join(bag_dir,'count.npz'), count_array)
      np.savez_compressed(os.path.join(bag_dir,'sig.npz'), sig_array)
      np.savez_compressed(os.path.join(bag_dir,'dem.npz'), dem_array)

      print("Done.")

      print("Generating statistics summary (report card)...")
      print(
f"""
Report Card - Statistical Distributions Across All Voxels [mu +/- 1 sigma]:

DEM +Y                 :      {np.nanmean(dem_array):.5} +/- {np.nanstd(dem_array):.5} [meters]
Point Count            :      {np.nanmean(count_array):.5} +/- {np.nanstd(count_array):.5} [-]
Local Normal vs +Y     :      {np.nanmean(slope_angle_array):.5} +/- {np.nanstd(slope_angle_array):.5} [degrees]
Spatial Error          :      {np.nanmean(sig_array):.5} +/- {np.nanstd(sig_array):.5} [meters]

""")

      print("Done.")

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
