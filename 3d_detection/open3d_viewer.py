#!/usr/bin/env python3
"""
Open3D Visualization Tool for 3D Detection Results

Interactive viewer for .ply files with detected objects
"""

import argparse
import sys
from pathlib import Path

try:
    import open3d as o3d
    import numpy as np
except ImportError as e:
    print(f"Error: {e}")
    print("Please install open3d: pip install open3d")
    sys.exit(1)


def visualize_ply_file(ply_path: str, window_name: str = "3D Detection Visualization"):
    """
    Visualize a PLY file using Open3D

    Args:
        ply_path: Path to .ply file
        window_name: Window title
    """
    print(f"Loading: {ply_path}")

    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)

    if not pcd.has_points():
        print(f"Error: No points found in {ply_path}")
        return

    print(f"  Points: {len(pcd.points)}")
    print(f"  Has colors: {pcd.has_colors()}")
    print(f"  Has normals: {pcd.has_normals()}")

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)

    # Add geometry
    vis.add_geometry(pcd)

    # Set render options
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.show_coordinate_frame = True

    # Set view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])

    # Run visualizer
    print("\nControls:")
    print("  - Mouse left: Rotate")
    print("  - Mouse wheel: Zoom")
    print("  - Mouse right: Pan")
    print("  - Q: Quit")
    print()

    vis.run()
    vis.destroy_window()


def visualize_directory(directory: str, pattern: str = "*.ply"):
    """
    Visualize all PLY files in a directory one by one

    Args:
        directory: Directory containing .ply files
        pattern: File pattern to match
    """
    ply_files = sorted(Path(directory).glob(pattern))

    if not ply_files:
        print(f"No PLY files found in {directory}")
        return

    print(f"Found {len(ply_files)} PLY files")
    print("Press 'Q' to move to next file")
    print()

    for i, ply_file in enumerate(ply_files):
        window_name = f"3D Detection ({i+1}/{len(ply_files)}) - {ply_file.name}"
        visualize_ply_file(str(ply_file), window_name)


def main():
    parser = argparse.ArgumentParser(description='Open3D Visualization Tool')
    parser.add_argument('input', type=str,
                       help='Path to .ply file or directory containing .ply files')
    parser.add_argument('--pattern', type=str, default='*.ply',
                       help='File pattern to match (if input is directory)')
    parser.add_argument('--examples', action='store_true',
                       help='Show usage examples')

    args = parser.parse_args()

    if args.examples:
        print("Usage Examples:")
        print("="*60)
        print()
        print("1. View single PLY file:")
        print("   python open3d_viewer.py results/waymo_pointpillars/ply/sample_000000.ply")
        print()
        print("2. View all PLY files in directory:")
        print("   python open3d_viewer.py results/waymo_pointpillars/ply/")
        print()
        print("3. View with custom pattern:")
        print("   python open3d_viewer.py results/ --pattern '**/sample_*.ply'")
        print()
        return

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Path does not exist: {args.input}")
        return

    if input_path.is_file():
        visualize_ply_file(str(input_path))
    elif input_path.is_dir():
        visualize_directory(str(input_path), args.pattern)
    else:
        print(f"Error: Invalid input path: {args.input}")


if __name__ == '__main__':
    main()
