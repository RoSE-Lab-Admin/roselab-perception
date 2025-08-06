import open3d as o3d
import numpy as np
import yaml
import json
import pickle
import csv
import argparse
from pathlib import Path

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Simple script for transforming a pointcloud and writing it to disk.')
    parser.add_argument("pointcloud", help="Pointcloud to be transformed. This must be a valid path to either a PLY or PCD file.")
    # parser.add_argument("-tf", "--transform", required=True) # Optional: Will eventually be required, this is the transform provided via various input formats (npz, serialized scipy Rotation, csv, etc)
    # parser.add_argument("-o", "--output", required=False) # Optional: Specify output path and filename

    args = parser.parse_args()

    tf = np.array([
        [-0.03770341,  0.9992143,   0.01221658,  1.14079563],
        [-0.99914733, -0.03748941, -0.01729633, -2.50940098],
        [-0.01682475, -0.0128583,   0.99977577, -2.81812123],
        [ 0.,          0.,          0.,          1.        ]
    ])

    pcpath = Path(args.pointcloud).absolute()
    basepath = pcpath.parent
    basename = pcpath.stem
    ext = pcpath.suffix

    pc = o3d.io.read_point_cloud(args.pointcloud)

    # Apply transform to pc
    transformed_pc = pc.transform(tf)
    o3d.io.write_point_cloud(basepath / Path(basename + "_transformed" + ext), transformed_pc)
