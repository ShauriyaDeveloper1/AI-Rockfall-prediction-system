"""
Generate sample LIDAR data for testing the rockfall prediction system
"""
import numpy as np
import open3d as o3d
import os
from datetime import datetime

def generate_rock_face_point_cloud(
    width=100, height=50, depth=10, 
    num_points=50000,
    add_instabilities=True
):
    """
    Generate a synthetic rock face point cloud with geological features
    """
    # Create basic rock face structure
    x = np.random.uniform(0, width, num_points)
    y = np.random.uniform(0, height, num_points)
    
    # Create base rock face surface with some natural variation
    base_z = np.sin(x * 0.1) * 2 + np.cos(y * 0.15) * 1.5
    
    # Add roughness
    roughness = np.random.normal(0, 0.5, num_points)
    z = base_z + roughness
    
    if add_instabilities:
        # Add some unstable areas (overhangs, loose rocks)
        
        # Overhang area (negative protrusion)
        overhang_mask = (x > 30) & (x < 50) & (y > 20) & (y < 35)
        z[overhang_mask] += np.random.uniform(-3, -1, np.sum(overhang_mask))
        
        # Crack/discontinuity (sudden depth change)
        crack_mask = (x > 60) & (x < 65) & (y > 10) & (y < 40)
        z[crack_mask] -= np.random.uniform(2, 4, np.sum(crack_mask))
        
        # Loose boulder (protruding area)
        boulder_mask = (x > 75) & (x < 85) & (y > 25) & (y < 35)
        z[boulder_mask] += np.random.uniform(2, 5, np.sum(boulder_mask))
    
    # Combine into point cloud
    points = np.column_stack((x, y, z))
    
    # Create colors based on geological features
    colors = np.zeros((num_points, 3))
    
    # Base rock color (grayish)
    colors[:, 0] = 0.6 + np.random.normal(0, 0.1, num_points)  # R
    colors[:, 1] = 0.5 + np.random.normal(0, 0.1, num_points)  # G
    colors[:, 2] = 0.4 + np.random.normal(0, 0.1, num_points)  # B
    
    if add_instabilities:
        # Color unstable areas differently
        colors[overhang_mask] = [0.8, 0.4, 0.2]  # Orange for overhangs
        colors[crack_mask] = [0.3, 0.3, 0.3]     # Dark for cracks
        colors[boulder_mask] = [0.7, 0.6, 0.5]   # Lighter for loose rocks
    
    # Clamp colors to valid range
    colors = np.clip(colors, 0, 1)
    
    return points, colors

def save_point_cloud_formats(points, colors, base_filename):
    """Save point cloud in multiple formats"""
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create data directory if it doesn't exist
    os.makedirs("sample_data", exist_ok=True)
    
    # Save as PLY (common LIDAR format)
    ply_file = f"sample_data/{base_filename}.ply"
    o3d.io.write_point_cloud(ply_file, pcd)
    print(f"Saved PLY: {ply_file}")
    
    # Save as PCD (Point Cloud Data)
    pcd_file = f"sample_data/{base_filename}.pcd"
    o3d.io.write_point_cloud(pcd_file, pcd)
    print(f"Saved PCD: {pcd_file}")
    
    # Save as ASCII XYZ
    xyz_file = f"sample_data/{base_filename}.xyz"
    with open(xyz_file, 'w') as f:
        for i in range(len(points)):
            f.write(f"{points[i, 0]:.3f} {points[i, 1]:.3f} {points[i, 2]:.3f} "
                   f"{int(colors[i, 0]*255)} {int(colors[i, 1]*255)} {int(colors[i, 2]*255)}\n")
    print(f"Saved XYZ: {xyz_file}")
    
    return ply_file, pcd_file, xyz_file

def generate_sample_datasets():
    """Generate multiple sample datasets for testing"""
    
    datasets = [
        {
            "name": "stable_rock_face",
            "description": "Stable rock face with minimal instabilities",
            "add_instabilities": False,
            "num_points": 30000
        },
        {
            "name": "unstable_rock_face", 
            "description": "Rock face with multiple geological instabilities",
            "add_instabilities": True,
            "num_points": 50000
        },
        {
            "name": "high_risk_area",
            "description": "High-risk area with major discontinuities",
            "add_instabilities": True,
            "num_points": 75000,
            "width": 150,
            "height": 75
        }
    ]
    
    generated_files = []
    
    for dataset in datasets:
        print(f"\nGenerating {dataset['name']}...")
        print(f"Description: {dataset['description']}")
        
        # Generate point cloud
        points, colors = generate_rock_face_point_cloud(
            width=dataset.get('width', 100),
            height=dataset.get('height', 50),
            num_points=dataset['num_points'],
            add_instabilities=dataset['add_instabilities']
        )
        
        # Save in multiple formats
        files = save_point_cloud_formats(points, colors, dataset['name'])
        generated_files.extend(files)
        
        print(f"Generated {len(points)} points")
    
    return generated_files

if __name__ == "__main__":
    print("Generating sample LIDAR datasets for rockfall prediction system...")
    print("=" * 60)
    
    generated_files = generate_sample_datasets()
    
    print("\n" + "=" * 60)
    print("Sample data generation complete!")
    print(f"Generated {len(generated_files)} files:")
    for file in generated_files:
        file_size = os.path.getsize(file) / (1024 * 1024)  # MB
        print(f"  - {file} ({file_size:.2f} MB)")
    
    print("\nThese files can be used to test the LIDAR processing system.")
    print("Upload them through the frontend LIDAR visualization component.")