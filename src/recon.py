from tqdm import tqdm
import open3d as o3d
import open3d.core as o3c
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from dataset.unpack import ROS2RGBDStreamer, setup_cli
import gc

def create_fresh_vbg(device):
    """Helper to initialize a strict, memory-bounded VoxelBlockGrid."""
    # A block_count of 40,000 at 1cm resolution uses roughly 3-4GB of VRAM.
    # It prevents Open3D from dynamically over-allocating past your GPU limits.
    return o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=0.01,       
        block_resolution=16,   
        block_count=40000,     # <--- Strict limit
        device=device
    )

def get_extrinsic_matrix(pose_tuple, convert_ros_to_vision=True, flip=True):
    """
    Converts a 7-DOF pose (x, y, z, qx, qy, qz, qw) to a 4x4 Extrinsic matrix.
    Extrinsics are World-to-Camera (the inverse of the Camera pose).
    """
    if pose_tuple is None:
        return np.eye(4)
        
    x, y, z, qx, qy, qz, qw = pose_tuple
    
    # 1. Build T_base_in_world
    T_base_in_world = np.eye(4)

    if flip: # RH: OUR ROVER HAS +X BACKWARDS!!!! :(
        T_base_in_world[:3, :3] = Rotation.from_euler('z', np.pi).as_matrix() * Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    else:
        T_base_in_world[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        
    T_base_in_world[:3, 3] = [x, y, z]
    
    # 2. Apply Base to Camera Optical Frame Transformation
    if convert_ros_to_vision:
        # Columns: Cam X (-Base Y), Cam Y (-Base Z), Cam Z (+Base X)
        T_cam_in_base = np.array([
            [ 0.0,  0.0,  1.0,  0.0],
            [-1.0,  0.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0,  1.0]
        ])
        
        # T_cam_in_world = T_base_in_world * T_cam_in_base
        T_cam_in_world = T_base_in_world @ T_cam_in_base
    else:
        T_cam_in_world = T_base_in_world
        
    # 3. Open3D requires the Extrinsic matrix (World-to-Camera)
    extrinsic = np.linalg.inv(T_cam_in_world)
    return extrinsic

def main():
    # Setup Intrinsics (replace with your actual camera calibrations)
    # fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5
    # intrinsics_list = [fx, fy, cx, cy]

    # Parse
    parser = setup_cli()
    args = parser.parse_args()
    
    # 1. Setup the Hardware Device (Use GPU if available)
    device = o3c.Device("cuda:0") if o3c.cuda.is_available() else o3c.Device("cpu:0")
    print(f"🚀 Initializing Open3D Tensor API on: {device}")

    # 2. Setup the Voxel Block Grid (Tensor-based TSDF)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=0.05,       # 1cm voxels
        block_resolution=16,   # 16x16x16 voxels per block
        block_count=10000,     # Initial allocated blocks
        device=device
    )

    # 3. Initialize our Streamer
    streamer = ROS2RGBDStreamer(
        bag_paths=args.bags,
        rgb_topic=args.rgb,
        depth_topic=args.depth,
        pose_topic=args.pose,
        camera_info_topic=args.info,
        intrinsics=args.intrinsics,
        max_delta=args.max_delta,
        sync_motion=args.sync_motion
    )
    streamer._analyze_and_sync()

    # Get intrinsics from streamer
    (fx, fy, cx, cy) = streamer.intrinsics
    intrinsic_tensor = o3c.Tensor(
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]], 
        dtype=o3c.float64, 
        device=o3c.Device("cpu:0") # Device must be CPU!!!
    )

    print(f"🔄 Streaming data directly to {device} TSDF Volume...")
    frame_count = 0
    skip_frame_count = 0

    # 4. The Integration Loop
    # MAX_FRAMES = 1500 # RH: Temp hack to avoid CUDA OOM
    # MAX_DEPTH = 5.0
    # SKIP_FRAMES = 300 # RH: Hack to avoid intialization period of /CubeRover_V1/pose. Going to need to make my usable data track (and maybe fill in? filter out?) gaps in ground truth...

    MAX_FRAMES = 350
    MAX_DEPTH = 1.5
    SKIP_FRAMES = 700

    for t, pose, rgb, depth in tqdm(streamer.stream(), total=len(streamer.target_stamps)):
        skip_frame_count += 1
        if skip_frame_count < SKIP_FRAMES:
            continue

        if t is None or pose is None or rgb is None or depth is None:
            print("[WARNING] Incomplete triplet!!! Skipping integration...")
            continue

        if frame_count >= MAX_FRAMES:
            break
            
        # Convert NumPy Arrays to Open3D Tensors and push to device
        depth_t = o3d.t.geometry.Image(o3c.Tensor(depth, device=device))
        
        # Open3D expects RGB, OpenCV provides BGR
        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        color_t = o3d.t.geometry.Image(o3c.Tensor(rgb_rgb, device=device))
        
        # Calculate Extrinsics
        extrinsic_matrix = get_extrinsic_matrix(pose, convert_ros_to_vision=True)
        # extrinsic_matrix = get_extrinsic_matrix(pose, convert_ros_to_vision=False, flip=False)
        extrinsic_tensor = o3c.Tensor(extrinsic_matrix, dtype=o3c.float64, device=o3c.Device("cpu:0")) # Device must be CPU!!!

        # VBG integration method (RH: TODO - Port over the RGBD->PointCloud->Aggregate+Refine pipeline as an optional step here
        try:
            # Get spatial hash
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth_t, 
                intrinsic_tensor, 
                extrinsic_tensor, 
                depth_scale=1000.0, 
                depth_max=MAX_DEPTH
            )
        except:
            print("[WARNING] No voxel hash found, skipping integration of frame...")
            continue
        
        # Integrate into the Voxel Block Grid
        vbg.integrate(
            block_coords=frustum_block_coords,
            depth=depth_t,
            color=color_t,
            intrinsic=intrinsic_tensor,
            extrinsic=extrinsic_tensor,
            depth_scale=1000.0, # Adjust if your depth is not in standard mm
            depth_max=MAX_DEPTH       # Truncate depth beyond 3 meters for cleaner meshes
        )
        
        frame_count += 1
        # print(f"Integrated Frame {frame_count}: t={t:.3f}", end='\r')

    print("\n✅ Stream complete. Extracting mesh...")

    # 5. Extract Mesh and Save
    # mesh_t = vbg.extract_triangle_mesh()
    # mesh_legacy = mesh_t.to_legacy() # Convert back to legacy for standard saving/viewing    
    # o3d.io.write_triangle_mesh("tensor_reconstruction.ply", mesh_legacy)
    # print("💾 Saved mesh to 'tensor_reconstruction.ply'.")

    pcd_t = vbg.extract_point_cloud(weight_threshold=23.0) # RH: Higher is better for filtering out noise [which we have a lot of :( ]
    geom = pcd_t.to_legacy()
    o3d.io.write_point_cloud("tensor_reconstruction.ply", geom)
    print("💾 Saved point cloud to 'tensor_reconstruction.ply'.")

    # Optional: Visualize right away
    o3d.visualization.draw_geometries([geom])

# RH: TODO - CLEAN UP AND INTEGRATE WITH ABOVE AS CLASS!!!
def dummy_chunker():

    # ... [Your streamer setup code here] ...

    vbg = create_fresh_vbg(device)
    chunk_idx = 0
    frames_per_chunk = 500  # Tune this: 500 frames is usually safe for 8GB

    print("🔄 Streaming data with Sub-Mapping enabled...")

    for frame_count, (t, pose, rgb, depth, cinfo) in enumerate(streamer.stream()):
        if pose is None or depth is None or cinfo is None:
            continue

        # ---------------------------------------------------------
        # MEMORY MANAGEMENT: Fragment the VBG if we hit the limit
        # ---------------------------------------------------------
        if frame_count > 0 and frame_count % frames_per_chunk == 0:
            print(f"\n📦 Chunk {chunk_idx} complete! Offloading to CPU/Disk...")
            
            # 1. Extract to GPU, then immediately move to CPU (legacy)
            mesh_t = vbg.extract_triangle_mesh()
            mesh_legacy = mesh_t.to_legacy()
            
            # 2. Save to disk (or append to a CPU RAM list if you prefer)
            o3d.io.write_triangle_mesh(f"reconstruction_chunk_{chunk_idx}.ply", mesh_legacy)
            
            # 3. Nuke the GPU volume to free VRAM
            del vbg
            del mesh_t
            gc.collect() # Force Python to clean up the orphaned GPU pointers
            
            # 4. Spin up a fresh volume and increment
            vbg = create_fresh_vbg(device)
            chunk_idx += 1
            print("✅ VRAM cleared. Resuming integration...")
        # ---------------------------------------------------------

        # ... [Your Intrinsics and Extrinsics matrix creation here] ...
        
        # Push images to GPU
        depth_t = o3d.t.geometry.Image(o3c.Tensor(depth, device=device))
        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        color_t = o3d.t.geometry.Image(o3c.Tensor(rgb_rgb, device=device))
        
        # 1. Compute blocks
        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth_t, intrinsic_tensor, extrinsic_tensor, 1000.0, 3.0
        )
        
        if frustum_block_coords.shape[0] == 0:
            continue

        # 2. Integrate
        vbg.integrate(
            frustum_block_coords, depth_t, color_t, 
            intrinsic_tensor, extrinsic_tensor, 1000.0, 3.0
        )
        
        print(f"Integrated Frame {frame_count}: t={t:.3f}", end='\r')

    # Flush the final chunk when the stream ends
    if vbg is not None:
        print(f"\n📦 Offloading final chunk {chunk_idx}...")
        o3d.io.write_triangle_mesh(
            f"reconstruction_chunk_{chunk_idx}.ply", 
            vbg.extract_triangle_mesh().to_legacy()
        )
        print("✅ Stream and chunking complete.")

if __name__ == "__main__":
    main()
