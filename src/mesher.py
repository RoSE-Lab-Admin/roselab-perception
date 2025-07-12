import numpy as np
import open3d as o3d
import sys
import matplotlib.pyplot as plt

if __name__=="__main__":
    # Read file from stdin
    MODE = sys.argv[1].lower()
    if not (MODE in ["mesh", "pointcloud"]):
        raise ValueError("[ERROR] First argument 'MODE' must be one of 'MESH' or 'POINTCLOUD'. Try again.")

    DATAFILE = sys.argv[2]
    OUTFILE = sys.argv[3]

    print(f"[INFO] Processing and visualizing '{DATAFILE}' with {MODE=}")

    # Load the point cloud
    pcd = o3d.io.read_point_cloud(DATAFILE)

#    # Remove statistical outliers first
#    cl, inds = pcd.remove_statistical_outlier(nb_neighbors=20,
#                                                    std_ratio=2.0)
#    # Downsample via voxelization to 0.01m?
#    pcd.voxel_down_sample(voxel_size=0.01)

    # THIS IS CURRENTLY CAUSING ALL SORTS OF FUCKERY!!!!!!!!!!!!
    print(f"[INFO] Estimating normals")
    pcd.estimate_normals()


    # Try setting floor normals towards center of mass of point cloud as initial guess?
    # pcd.orient_normals_towards_camera_location(np.mean(np.asarray(pcd.points), 0))
    # pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) * -1.)

    # Do we need to estimate normals first? YES!!!!
#    print(f"[INFO] Orienting normals along consistent tangent plane")
#    pcd.orient_normals_consistent_tangent_plane(10) # This shouldn't give us an issue but lambda is a keyword...

    print(f"[INFO] Detecting planar patches")
    pp = pcd.detect_planar_patches()

    if MODE == "mesh":
        print(f"[INFO] Starting mesh extraction via Poisson Surface Reconstruction")

        # Do poisson reconstruction
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5)


        print(f"[INFO] Finished mesh generation.")

        # Filter by density
        densities = np.asarray(densities)

        # DEBUG
        # Display a histogram of the mesh densities? What is this reported in in terms of units? Point per facet?
        import matplotlib.pyplot as plt
        plt.hist(densities.flat, bins=100)
        plt.title("Mesh Density Distribution")
        plt.show()

        density_colors = plt.get_cmap('plasma')((densities - densities.min()) / (densities.max() - densities.min()))
        density_colors = density_colors[:, :3]
        density_mesh = o3d.geometry.TriangleMesh()
        density_mesh.vertices = mesh.vertices
        density_mesh.triangles = mesh.triangles
        density_mesh.triangle_normals = mesh.triangle_normals
        density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
#        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        o3d.visualization.draw_geometries([density_mesh])

        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        # mesh.compute_vertex_normals()
        # mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals) * -1.)

        # Visualize the finalized textured mesh
        o3d.visualization.draw_geometries([mesh] + pp)

        # Save mesh as PLY file
        o3d.io.write_triangle_mesh(OUTFILE, mesh)

    elif MODE == "pointcloud":
        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd] + pp)
