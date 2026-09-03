import open3d as o3d
import numpy as np
import sys

if __name__=="__main__":
    fname = sys.argv[1]
    mode = sys.argv[2].lower()

    MASK = False
    if len(sys.argv) > 3:
        MASK = True
        MIN_Q = float(sys.argv[3])
        MAX_Q = float(sys.argv[4])

    if mode in ["pointcloud", "pc"]:
        pcd = o3d.io.read_point_cloud(fname)

        if MASK:
            # Do one last colormap filtering step to condense Z values to smaller range
            pts = np.asarray(pcd.points)
            mask = (pts[:,2] > np.quantile(pts[:,2], MIN_Q)) & (pts[:,2] < np.quantile(pts[:,2], MAX_Q))

            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(pts[mask])
            filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])

            o3d.visualization.draw([filtered_pcd], show_ui=True)
        else:
            o3d.visualization.draw([pcd], show_ui=True)

    else:
        # Mesh
        mesh = o3d.io.read_triangle_mesh(fname)
        o3d.visualization.draw([mesh], show_ui=True)
