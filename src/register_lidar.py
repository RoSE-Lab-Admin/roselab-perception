#import open3d as o3d
#import numpy as np
#
#class PointCloudSfM:
#    def __init__(self, intrinsic, voxel_size=0.01):
#        self.intrinsic = intrinsic
#        self.voxel_size = voxel_size
#        self.pose_graph = o3d.pipelines.registration.PoseGraph()
#        self.pcds = [] # Store downsampled point clouds with normals
#
#    def _prepare_pcd(self, color_path, depth_path):
#        """Creates a point cloud with normals for Point-to-Plane ICP."""
#        color = o3d.io.read_image(color_path)
#        depth = o3d.io.read_image(depth_path)
#        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#            color, depth, depth_scale=1000.0, convert_rgb_to_intensity=False
#        )
#        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
#        pcd = pcd.voxel_down_sample(self.voxel_size)
#        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#        return pcd
#
#    def add_frame(self, color_path, depth_path, known_pose=None):
#        pcd = self._prepare_pcd(color_path, depth_path)
#        curr_idx = len(self.pcds)
#        self.pcds.append(pcd)
#        
#        # Add Node: If known_pose is provided, it's a fixed anchor
#        # If None, we will initialize with Identity and let Odometry fill it
#        init_pose = known_pose if known_pose is not None else np.eye(4)
#        node = o3d.pipelines.registration.PoseGraphNode(init_pose)
#        self.pose_graph.nodes.append(node)
#
#        if curr_idx > 0:
#            self._add_odometry_edge(curr_idx - 1, curr_idx)
#
#    def _add_odometry_edge(self, s_idx, t_idx):
#        """Computes Point-to-Plane ICP between consecutive frames."""
#        source = self.pcds[t_idx]
#        target = self.pcds[s_idx]
#        
#        # Initial guess from current node state
#        initial_trans = np.linalg.inv(self.pose_graph.nodes[s_idx].pose) @ self.pose_graph.nodes[t_idx].pose
#        
#        # Point-to-Plane ICP for better structural alignment
#        reg_log = o3d.pipelines.registration.registration_icp(
#            source, target, self.voxel_size * 2, initial_trans,
#            o3d.pipelines.registration.TransformationEstimationPointToPlane()
#        )
#
#        # Update the target node pose if it was an empty guess (Identity)
#        if np.allclose(self.pose_graph.nodes[t_idx].pose, np.eye(4)):
#            self.pose_graph.nodes[t_idx].pose = self.pose_graph.nodes[s_idx].pose @ reg_log.transformation
#
#        # Information matrix gauges the "confidence" of the link
#        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
#            source, target, self.voxel_size * 2, reg_log.transformation
#        )
#        
#        self.pose_graph.edges.append(
#            o3d.pipelines.registration.PoseGraphEdge(s_idx, t_idx, reg_log.transformation, info, uncertain=False)
#        )
#
#    def optimize_and_merge(self):
#        """Optimizes poses and returns a single merged point cloud."""
#        print("Optimizing Global Pose Graph...")
#        option = o3d.pipelines.registration.GlobalOptimizationOption(
#            max_correspondence_distance=self.voxel_size * 1.5,
#            edge_pruning_threshold=0.25,
#            reference_node=0
#        )
#        o3d.pipelines.registration.global_optimization(
#            self.pose_graph,
#            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
#            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
#            option
#        )
#
#        # Merge all into one world-space cloud
#        final_pcd = o3d.geometry.PointCloud()
#        for i, pcd in enumerate(self.pcds):
#            # Use deep copy to avoid modifying original nodes
#            temp_pcd = o3d.geometry.PointCloud(pcd)
#            temp_pcd.transform(self.pose_graph.nodes[i].pose)
#            final_pcd += temp_pcd
#
#        return final_pcd.voxel_down_sample(self.voxel_size)
#
#if __name__=="__main__":
#    # Init
#
#    # Stream data from unpacking module
#
#    # For each triplet add frame
#
#    # Optimize and merge

# RH: Making a gradio app similar to Antar's but meant specifically for reconstruction of our static capture lidar data sequences

import open3d as o3d
import numpy as np
import gradio as gr
import tempfile

def apply_fusion(west_ply, center_ply, east_ply, w2c_txt, e2c_txt, c2w_txt, voxel_size):
    # 1. Load Point Clouds
    pcd_w = o3d.io.read_point_cloud(west_ply.name)
    pcd_c = o3d.io.read_point_cloud(center_ply.name)
    pcd_e = o3d.io.read_point_cloud(east_ply.name)

    # 2. Helper to load 4x4 matrices from text/numpy files
    def load_matrix(file_obj):
        if file_obj is None: return np.eye(4)
        # Handles space-separated text or numpy saved arrays
        try:
            return np.loadtxt(file_obj.name)
        except:
            return np.load(file_obj.name)

    # Load transforms
    T_west_to_center = load_matrix(w2c_txt)
    T_east_to_center = load_matrix(e2c_txt)
    T_center_to_world = load_matrix(c2w_txt)

    # 3. Initial Alignment (Move East/West to Center Frame)
    pcd_w.transform(T_west_to_center)
    pcd_e.transform(T_east_to_center)

    # 4. Refinement Pass (ICP)
    # We refine West -> Center and East -> Center independently
    def refine(source, target, v_size):
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v_size*2, max_nn=30))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v_size*2, max_nn=30))
        res = o3d.pipelines.registration.registration_icp(
            source, target, v_size * 1.5, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        return source.transform(res.transformation)

    pcd_w = refine(pcd_w, pcd_c, voxel_size)
    pcd_e = refine(pcd_e, pcd_c, voxel_size)

    # 5. Final Global Transform (Optional Center to World)
    combined = pcd_w + pcd_c + pcd_e
    combined.transform(T_center_to_world)

    # Save Result
    out_path = tempfile.mktemp(suffix=".ply")
    o3d.io.write_point_cloud(out_path, combined)
    return "Fusion Complete", out_path

# --- UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛰️ Lidar Sequence Processing Suite")
    
    with gr.Tab("1. Sequential Alignment"):
        with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
            gr.Markdown("# 🚶 Sequential LiDAR Stitcher")
            gr.Markdown("Files are processed in the **exact order** they appear in the list below.")
            
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Upload PLYs in Order", file_count="multiple")
                    with gr.Row():
                        direction_toggle = gr.Radio(["+Y", "-Y"], label="Movement Direction", value="+Y")
                        dist_input = gr.Number(label="Est. Distance (m)", value=1.0)
                    v_size = gr.Slider(0.01, 0.2, value=0.05, label="Voxel Size (ICP Precision)")
                    run_btn = gr.Button("Align Sequence", variant="primary")
                
                with gr.Column():
                    status = gr.Textbox(label="Status")
                    viewer = gr.Model3D(label="Aligned Result")

            run_btn.click(
                process_ordered_files, 
                [file_input, v_size, dist_input, direction_toggle], 
                [status, viewer]
            )

        gr.Markdown("Step 1: Align individual sensor sequences here.")

    with gr.Tab("2. Multi-Sensor Fusion"):
        gr.Markdown("### Fuse West and East sensors onto the Center Anchor")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Reconstructions (.ply)")
                pcd_w_in = gr.File(label="West Reconstruction")
                pcd_c_in = gr.File(label="Center Reconstruction (Anchor)")
                pcd_e_in = gr.File(label="East Reconstruction")
            
            with gr.Column():
                gr.Markdown("#### Calibration Matrices (4x4 txt/npy)")
                mat_w2c = gr.File(label="West-to-Center Transform")
                mat_e2c = gr.File(label="East-to-Center Transform")
                mat_c2w = gr.File(label="Optional: Center-to-World Transform")
        
        v_size_fuse = gr.Slider(0.01, 0.2, value=0.05, label="Refinement Voxel Size")
        fuse_btn = gr.Button("Perform Global Fusion", variant="primary")
        
        with gr.Row():
            fuse_status = gr.Textbox(label="Status")
            fuse_viewer = gr.Model3D(label="Global Map")

    fuse_btn.click(
        apply_fusion, 
        [pcd_w_in, pcd_c_in, pcd_e_in, mat_w2c, mat_e2c, mat_c2w, v_size_fuse],
        [fuse_status, fuse_viewer]
    )

demo.launch()
