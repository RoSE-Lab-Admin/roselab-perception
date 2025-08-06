import open3d as o3d
import numpy as np

if __name__=="__main__":
    tf = np.array([
        [-0.03770341,  0.9992143,   0.01221658,  1.14079563],
        [-0.99914733, -0.03748941, -0.01729633, -2.50940098],
        [-0.01682475, -0.0128583,   0.99977577, -2.81812123],
        [ 0.,          0.,          0.,          1.        ]
    ])

    pc = o3d.io.read_point_cloud("out.pcd")

    # Apply transform to pc
    transformed_pc = pc.transform(tf)
    o3d.io.write_point_cloud("out_tf.pcd", transformed_pc)
