"""
Main point cloud processing module for geological analysis.
Provides preprocessing, filtering, and basic analysis capabilities.
"""

import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

@dataclass 
class ProcessingParameters:
    """Parameters for point cloud processing"""
    voxel_size: float = 0.1  # Downsampling voxel size
    noise_removal_neighbors: int = 20
    noise_removal_std_ratio: float = 2.0
    surface_normal_radius: float = 0.5
    surface_normal_max_nn: int = 30
    clustering_eps: float = 0.3
    clustering_min_points: int = 10

class PointCloudProcessor:
    """Main class for processing 3D point clouds for geological analysis"""
    
    def __init__(self, params: Optional[ProcessingParameters] = None):
        self.params = params or ProcessingParameters()
        self.logger = logging.getLogger(__name__)
        
    def preprocess(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Complete preprocessing pipeline for point cloud
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Preprocessed point cloud
        """
        self.logger.info("Starting point cloud preprocessing")
        
        # Step 1: Downsample
        pcd_down = self.downsample(pcd)
        
        # Step 2: Remove noise
        pcd_clean = self.remove_noise(pcd_down)
        
        # Step 3: Estimate normals
        pcd_normals = self.estimate_normals(pcd_clean)
        
        self.logger.info(f"Preprocessing complete: {len(pcd.points)} -> {len(pcd_normals.points)} points")
        return pcd_normals
    
    def downsample(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Downsample point cloud using voxel grid
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Downsampled point cloud
        """
        if len(pcd.points) == 0:
            return pcd
        
        pcd_down = pcd.voxel_down_sample(self.params.voxel_size)
        self.logger.debug(f"Downsampled: {len(pcd.points)} -> {len(pcd_down.points)} points")
        return pcd_down
    
    def remove_noise(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Remove statistical outliers from point cloud
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Cleaned point cloud
        """
        if len(pcd.points) == 0:
            return pcd
        
        pcd_clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=self.params.noise_removal_neighbors,
            std_ratio=self.params.noise_removal_std_ratio
        )
        
        self.logger.debug(f"Noise removal: {len(pcd.points)} -> {len(pcd_clean.points)} points")
        return pcd_clean
    
    def estimate_normals(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Estimate surface normals for point cloud
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Point cloud with estimated normals
        """
        if len(pcd.points) == 0:
            return pcd
        
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.params.surface_normal_radius,
                max_nn=self.params.surface_normal_max_nn
            )
        )
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        return pcd
    
    def segment_ground(self, pcd: o3d.geometry.PointCloud, 
                      distance_threshold: float = 0.2,
                      ransac_n: int = 3,
                      num_iterations: int = 1000) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        Segment ground plane from point cloud using RANSAC
        
        Args:
            pcd: Input point cloud
            distance_threshold: Distance threshold for RANSAC
            ransac_n: Number of points to fit plane
            num_iterations: Number of RANSAC iterations
            
        Returns:
            Tuple of (ground_plane, non_ground) point clouds
        """
        if len(pcd.points) == 0:
            return pcd, pcd
        
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        ground_plane = pcd.select_by_index(inliers)
        non_ground = pcd.select_by_index(inliers, invert=True)
        
        self.logger.debug(f"Ground segmentation: {len(inliers)} ground points, "
                         f"{len(non_ground.points)} non-ground points")
        
        return ground_plane, non_ground
    
    def cluster_points(self, pcd: o3d.geometry.PointCloud) -> List[o3d.geometry.PointCloud]:
        """
        Cluster points using DBSCAN
        
        Args:
            pcd: Input point cloud
            
        Returns:
            List of clustered point clouds
        """
        if len(pcd.points) == 0:
            return []
        
        labels = np.array(pcd.cluster_dbscan(
            eps=self.params.clustering_eps,
            min_points=self.params.clustering_min_points,
            print_progress=False
        ))
        
        max_label = labels.max()
        clusters = []
        
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > self.params.clustering_min_points:
                cluster = pcd.select_by_index(cluster_indices)
                clusters.append(cluster)
        
        self.logger.debug(f"Found {len(clusters)} clusters from {len(pcd.points)} points")
        return clusters
    
    def compute_surface_roughness(self, pcd: o3d.geometry.PointCloud, 
                                 radius: float = 0.5) -> np.ndarray:
        """
        Compute surface roughness for each point
        
        Args:
            pcd: Input point cloud with normals
            radius: Search radius for local analysis
            
        Returns:
            Array of roughness values for each point
        """
        if len(pcd.points) == 0 or not pcd.has_normals():
            return np.array([])
        
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        # Build KD-tree for neighbor search
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        roughness = np.zeros(len(points))
        
        for i, point in enumerate(points):
            # Find neighbors within radius
            [k, idx, _] = kdtree.search_radius_vector_3d(point, radius)
            
            if k > 1:
                neighbor_normals = normals[idx[1:]]  # Exclude self
                center_normal = normals[i]
                
                # Compute angle differences
                dot_products = np.dot(neighbor_normals, center_normal)
                dot_products = np.clip(dot_products, -1.0, 1.0)
                angles = np.arccos(np.abs(dot_products))
                
                # Roughness as standard deviation of angles
                roughness[i] = np.std(angles)
        
        return roughness
    
    def compute_slope_angles(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Compute slope angles from surface normals
        
        Args:
            pcd: Point cloud with normals
            
        Returns:
            Array of slope angles in degrees
        """
        if not pcd.has_normals():
            return np.array([])
        
        normals = np.asarray(pcd.normals)
        
        # Compute angle between normal and vertical (0, 0, 1)
        vertical = np.array([0, 0, 1])
        dot_products = np.dot(normals, vertical)
        dot_products = np.clip(dot_products, -1.0, 1.0)
        
        # Convert to slope angles (complement of normal angle)
        slope_angles = 90.0 - np.degrees(np.arccos(np.abs(dot_products)))
        
        return slope_angles
    
    def extract_cross_sections(self, pcd: o3d.geometry.PointCloud, 
                              axis: str = 'z', 
                              num_sections: int = 10) -> List[o3d.geometry.PointCloud]:
        """
        Extract cross-sections from point cloud
        
        Args:
            pcd: Input point cloud
            axis: Axis along which to extract sections ('x', 'y', or 'z')
            num_sections: Number of cross-sections
            
        Returns:
            List of cross-section point clouds
        """
        if len(pcd.points) == 0:
            return []
        
        points = np.asarray(pcd.points)
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map.get(axis.lower(), 2)
        
        # Get axis bounds
        min_val = np.min(points[:, axis_idx])
        max_val = np.max(points[:, axis_idx])
        
        # Create sections
        section_thickness = (max_val - min_val) / num_sections
        sections = []
        
        for i in range(num_sections):
            section_min = min_val + i * section_thickness
            section_max = section_min + section_thickness
            
            # Select points in section
            mask = (points[:, axis_idx] >= section_min) & (points[:, axis_idx] <= section_max)
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                section_pcd = pcd.select_by_index(indices)
                sections.append(section_pcd)
        
        return sections
    
    def compute_point_cloud_statistics(self, pcd: o3d.geometry.PointCloud) -> Dict:
        """
        Compute comprehensive statistics for point cloud
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Dictionary of statistics
        """
        if len(pcd.points) == 0:
            return {}
        
        points = np.asarray(pcd.points)
        
        stats = {
            'num_points': len(points),
            'bounds': {
                'x_min': float(np.min(points[:, 0])),
                'x_max': float(np.max(points[:, 0])),
                'y_min': float(np.min(points[:, 1])),
                'y_max': float(np.max(points[:, 1])),
                'z_min': float(np.min(points[:, 2])),
                'z_max': float(np.max(points[:, 2]))
            },
            'centroid': {
                'x': float(np.mean(points[:, 0])),
                'y': float(np.mean(points[:, 1])),
                'z': float(np.mean(points[:, 2]))
            },
            'dimensions': {
                'width': float(np.max(points[:, 0]) - np.min(points[:, 0])),
                'length': float(np.max(points[:, 1]) - np.min(points[:, 1])),
                'height': float(np.max(points[:, 2]) - np.min(points[:, 2]))
            },
            'density': len(points) / (
                (np.max(points[:, 0]) - np.min(points[:, 0])) *
                (np.max(points[:, 1]) - np.min(points[:, 1])) *
                (np.max(points[:, 2]) - np.min(points[:, 2]))
            )
        }
        
        # Add surface area and volume if possible
        try:
            # Create mesh for surface area calculation
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
            stats['surface_area'] = mesh.get_surface_area()
            stats['volume'] = mesh.get_volume()
        except:
            stats['surface_area'] = None
            stats['volume'] = None
        
        return stats