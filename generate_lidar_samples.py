"""
Generate sample LIDAR point cloud data for testing the system.
Creates a PLY file with realistic geological features.
"""

import numpy as np
import open3d as o3d
import os

def generate_sample_lidar_data(filename="sample_rockface.ply", num_points=10000):
    """Generate a realistic rock face point cloud"""
    
    # Create base terrain
    x = np.random.uniform(-50, 50, num_points)
    y = np.random.uniform(-30, 30, num_points)
    
    # Create sloped rock face with geological features
    base_slope = 0.6  # Moderate slope
    z = x * base_slope + np.random.normal(0, 2, num_points)
    
    # Add geological features
    for i in range(len(x)):
        # Add discontinuity planes
        if -10 < x[i] < 10 and -5 < y[i] < 5:
            z[i] += 5 * np.sin(x[i] * 0.2) * np.cos(y[i] * 0.3)
        
        # Add weathered areas (more noise)
        if x[i] > 20:
            z[i] += np.random.normal(0, 1.5)
        
        # Add overhang feature
        if x[i] > 30 and -10 < y[i] < 0:
            z[i] += 8 - (x[i] - 30) * 0.5
        
        # Add crack-like features
        if abs(y[i] - 10) < 1:
            z[i] -= 2
    
    # Ensure z values are reasonable
    z = np.maximum(z, 0)
    
    # Create point cloud
    points = np.column_stack([x, y, z])
    
    # Create colors based on height and features
    colors = np.zeros((num_points, 3))
    z_norm = (z - np.min(z)) / (np.max(z) - np.min(z))
    
    for i in range(num_points):
        # Base rock color (gray to brown)
        colors[i] = [0.6, 0.5, 0.4]
        
        # Weathered areas (darker)
        if x[i] > 20:
            colors[i] *= 0.7
        
        # High areas (lighter)
        if z_norm[i] > 0.8:
            colors[i] += 0.2
        
        # Crack areas (darker)
        if abs(y[i] - 10) < 1:
            colors[i] *= 0.5
    
    # Clip colors to valid range
    colors = np.clip(colors, 0, 1)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
    )
    
    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), 'sample_data', filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    success = o3d.io.write_point_cloud(output_path, pcd)
    
    if success:
        print(f"✅ Sample LIDAR data saved to: {output_path}")
        print(f"   Points: {len(points):,}")
        print(f"   Bounds: X[{np.min(x):.1f}, {np.max(x):.1f}], Y[{np.min(y):.1f}, {np.max(y):.1f}], Z[{np.min(z):.1f}, {np.max(z):.1f}]")
        return output_path
    else:
        print(f"❌ Failed to save LIDAR data to: {output_path}")
        return None

def generate_multiple_samples():
    """Generate multiple sample files with different characteristics"""
    
    samples = [
        {
            "filename": "stable_slope.ply",
            "description": "Stable rock slope",
            "points": 5000,
            "risk": "LOW"
        },
        {
            "filename": "weathered_cliff.ply", 
            "description": "Weathered cliff face",
            "points": 8000,
            "risk": "HIGH"
        },
        {
            "filename": "fractured_outcrop.ply",
            "description": "Heavily fractured outcrop",
            "points": 12000,
            "risk": "CRITICAL"
        }
    ]
    
    for sample in samples:
        print(f"\n🔧 Generating {sample['description']}...")
        
        # Modify generation parameters based on risk level
        if sample['risk'] == 'LOW':
            # Stable slope - less noise, moderate angle
            x = np.random.uniform(-30, 30, sample['points'])
            y = np.random.uniform(-20, 20, sample['points'])
            z = x * 0.3 + np.random.normal(0, 0.5, sample['points'])
            
        elif sample['risk'] == 'HIGH':
            # Weathered - more noise, steeper
            x = np.random.uniform(-40, 40, sample['points'])
            y = np.random.uniform(-25, 25, sample['points'])
            z = x * 0.8 + np.random.normal(0, 3, sample['points'])
            
            # Add weathering noise
            for i in range(len(x)):
                if np.random.random() < 0.3:  # 30% of points affected
                    z[i] += np.random.normal(0, 2)
            
        else:  # CRITICAL
            # Fractured - very noisy, overhangs, steep
            x = np.random.uniform(-45, 45, sample['points'])
            y = np.random.uniform(-30, 30, sample['points'])
            z = x * 1.2 + np.random.normal(0, 4, sample['points'])
            
            # Add fractures and overhangs
            for i in range(len(x)):
                # Fracture zones
                if abs(y[i]) < 5 or abs(y[i] - 15) < 3:
                    z[i] -= np.random.uniform(2, 5)
                
                # Overhang areas
                if x[i] > 25:
                    z[i] += np.random.uniform(5, 15)
        
        z = np.maximum(z, 0)  # Ensure positive Z
        points = np.column_stack([x, y, z])
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Simple color based on height
        colors = np.zeros((sample['points'], 3))
        z_norm = (z - np.min(z)) / (np.max(z) - np.min(z))
        
        for i in range(sample['points']):
            if sample['risk'] == 'LOW':
                colors[i] = [0.7, 0.6, 0.5]  # Light brown
            elif sample['risk'] == 'HIGH':
                colors[i] = [0.5, 0.4, 0.3]  # Dark brown
            else:
                colors[i] = [0.4, 0.3, 0.3]  # Very dark
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.estimate_normals()
        
        # Save file
        output_path = os.path.join(os.path.dirname(__file__), 'sample_data', sample['filename'])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        success = o3d.io.write_point_cloud(output_path, pcd)
        if success:
            print(f"   ✅ Saved: {sample['filename']} ({sample['points']:,} points, Risk: {sample['risk']})")
        else:
            print(f"   ❌ Failed to save: {sample['filename']}")

if __name__ == "__main__":
    print("🎯 Generating sample LIDAR data for testing...")
    
    # Generate main sample
    generate_sample_lidar_data()
    
    # Generate additional samples
    generate_multiple_samples()
    
    print("\n✅ Sample data generation complete!")
    print("📁 Files saved in 'sample_data/' directory")
    print("🚀 You can now upload these files in the LIDAR visualization interface")