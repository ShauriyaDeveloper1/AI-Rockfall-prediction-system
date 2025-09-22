"""
Geological feature extraction from 3D point clouds.
Specialized algorithms for identifying rockfall risk indicators.
"""

import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import scipy.spatial.distance as distance

@dataclass
class GeologicalAnalysisResults:
    """Results of geological feature analysis"""
    slope_angles: np.ndarray
    surface_roughness: np.ndarray
    discontinuity_planes: List[Dict]
    crack_features: List[Dict]
    overhang_regions: List[Dict]
    weathering_indicators: Dict
    stability_score: float
    risk_factors: Dict

class GeologicalFeatureExtractor:
    """Extract geological features relevant to rockfall prediction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_point_cloud(self, pcd: o3d.geometry.PointCloud) -> GeologicalAnalysisResults:
        """
        Comprehensive geological analysis of point cloud
        
        Args:
            pcd: Input point cloud with normals
            
        Returns:
            Complete geological analysis results
        """
        self.logger.info("Starting comprehensive geological analysis")
        
        # Ensure we have normals
        if not pcd.has_normals():
            pcd.estimate_normals()
        
        # Extract individual features
        slope_angles = self._compute_slope_angles(pcd)
        surface_roughness = self._compute_surface_roughness(pcd)
        discontinuity_planes = self._detect_discontinuity_planes(pcd)
        crack_features = self._detect_cracks(pcd)
        overhang_regions = self._detect_overhangs(pcd)
        weathering_indicators = self._analyze_weathering(pcd)
        
        # Compute overall stability score
        stability_score = self._compute_stability_score(
            slope_angles, surface_roughness, discontinuity_planes, 
            crack_features, overhang_regions, weathering_indicators
        )
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            slope_angles, surface_roughness, discontinuity_planes,
            crack_features, overhang_regions, weathering_indicators
        )
        
        results = GeologicalAnalysisResults(
            slope_angles=slope_angles,
            surface_roughness=surface_roughness,
            discontinuity_planes=discontinuity_planes,
            crack_features=crack_features,
            overhang_regions=overhang_regions,
            weathering_indicators=weathering_indicators,
            stability_score=stability_score,
            risk_factors=risk_factors
        )
        
        self.logger.info(f"Geological analysis complete. Stability score: {stability_score:.3f}")
        return results
    
    def _compute_slope_angles(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """Compute slope angles from surface normals"""
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
    
    def _compute_surface_roughness(self, pcd: o3d.geometry.PointCloud, 
                                  radius: float = 0.5) -> np.ndarray:
        """Compute surface roughness for each point"""
        if not pcd.has_normals():
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
    
    def _detect_discontinuity_planes(self, pcd: o3d.geometry.PointCloud, 
                                   min_points: int = 100,
                                   distance_threshold: float = 0.1) -> List[Dict]:
        """Detect major discontinuity planes (joints, bedding, fractures)"""
        points = np.asarray(pcd.points)
        
        if len(points) < min_points:
            return []
        
        discontinuities = []
        remaining_points = np.arange(len(points))
        
        # Iteratively find planes
        for iteration in range(5):  # Limit to 5 major planes
            if len(remaining_points) < min_points:
                break
            
            # Create temporary point cloud
            temp_pcd = pcd.select_by_index(remaining_points)
            
            # RANSAC plane fitting
            plane_model, inliers = temp_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=1000
            )
            
            if len(inliers) < min_points:
                break
            
            # Map inliers back to original indices
            original_inliers = remaining_points[inliers]
            
            # Extract plane parameters
            a, b, c, d = plane_model
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)
            
            # Compute plane properties
            plane_points = points[original_inliers]
            centroid = np.mean(plane_points, axis=0)
            
            # Compute plane area using convex hull
            try:
                hull = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    temp_pcd.select_by_index(inliers), alpha=0.1
                )
                area = hull.get_surface_area()
            except:
                area = len(inliers) * 0.01  # Rough estimate
            
            # Compute dip angle (angle with horizontal)
            dip_angle = np.degrees(np.arcsin(abs(normal[2])))
            
            # Compute strike direction
            if abs(normal[0]) > 1e-6 or abs(normal[1]) > 1e-6:
                strike_vector = np.array([-normal[1], normal[0], 0])
                strike_vector = strike_vector / np.linalg.norm(strike_vector)
                strike_angle = np.degrees(np.arctan2(strike_vector[1], strike_vector[0]))
            else:
                strike_angle = 0.0
            
            discontinuity = {
                'id': iteration,
                'normal': normal.tolist(),
                'centroid': centroid.tolist(),
                'area': float(area),
                'num_points': len(inliers),
                'dip_angle': float(dip_angle),
                'strike_angle': float(strike_angle),
                'point_indices': original_inliers.tolist()
            }
            
            discontinuities.append(discontinuity)
            
            # Remove inliers from remaining points
            remaining_points = np.setdiff1d(remaining_points, original_inliers)
        
        return discontinuities
    
    def _detect_cracks(self, pcd: o3d.geometry.PointCloud, 
                      curvature_threshold: float = 0.1) -> List[Dict]:
        """Detect potential cracks and fractures"""
        if not pcd.has_normals():
            pcd.estimate_normals()
        
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        # Compute curvature as a crack indicator
        curvatures = self._compute_curvature(pcd)
        
        # Find high curvature regions
        crack_candidates = np.where(curvatures > curvature_threshold)[0]
        
        if len(crack_candidates) == 0:
            return []
        
        # Cluster crack points
        crack_points = points[crack_candidates]
        clustering = DBSCAN(eps=0.2, min_samples=10).fit(crack_points)
        
        cracks = []
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise points
                continue
            
            cluster_mask = clustering.labels_ == cluster_id
            cluster_indices = crack_candidates[cluster_mask]
            cluster_points = points[cluster_indices]
            
            # Fit line to crack
            pca = PCA(n_components=3)
            pca.fit(cluster_points)
            
            # Primary direction (largest variance)
            direction = pca.components_[0]
            centroid = np.mean(cluster_points, axis=0)
            
            # Compute crack length
            projected = np.dot(cluster_points - centroid, direction)
            crack_length = np.max(projected) - np.min(projected)
            
            # Compute crack width (standard deviation in secondary direction)
            secondary_proj = np.dot(cluster_points - centroid, pca.components_[1])
            crack_width = np.std(secondary_proj) * 2  # 2 standard deviations
            
            crack = {
                'id': len(cracks),
                'centroid': centroid.tolist(),
                'direction': direction.tolist(),
                'length': float(crack_length),
                'width': float(crack_width),
                'num_points': len(cluster_indices),
                'avg_curvature': float(np.mean(curvatures[cluster_indices])),
                'point_indices': cluster_indices.tolist()
            }
            
            cracks.append(crack)
        
        return cracks
    
    def _detect_overhangs(self, pcd: o3d.geometry.PointCloud) -> List[Dict]:
        """Detect overhang and undercut regions"""
        points = np.asarray(pcd.points)
        
        if not pcd.has_normals():
            pcd.estimate_normals()
        
        normals = np.asarray(pcd.normals)
        
        # Find points with normals pointing upward (potential overhangs)
        # Normal z-component > 0.7 (angle < ~45 degrees from vertical)
        overhang_mask = normals[:, 2] > 0.7
        overhang_indices = np.where(overhang_mask)[0]
        
        if len(overhang_indices) == 0:
            return []
        
        # Cluster overhang points
        overhang_points = points[overhang_indices]
        clustering = DBSCAN(eps=0.5, min_samples=20).fit(overhang_points)
        
        overhangs = []
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise points
                continue
            
            cluster_mask = clustering.labels_ == cluster_id
            cluster_indices = overhang_indices[cluster_mask]
            cluster_points = points[cluster_indices]
            
            # Compute overhang properties
            min_z = np.min(cluster_points[:, 2])
            max_z = np.max(cluster_points[:, 2])
            centroid = np.mean(cluster_points, axis=0)
            
            # Estimate overhang extent
            # Project points onto horizontal plane
            xy_points = cluster_points[:, :2]
            xy_hull = o3d.geometry.PointCloud()
            xy_hull.points = o3d.utility.Vector3dVector(
                np.column_stack([xy_points, np.zeros(len(xy_points))])
            )
            
            try:
                hull_2d = xy_hull.compute_convex_hull()[0]
                overhang_area = hull_2d.get_surface_area()
            except:
                overhang_area = len(cluster_indices) * 0.01  # Rough estimate
            
            # Check for undercut (points below the overhang)
            undercut_points = []
            for point in points:
                if (point[0] >= np.min(xy_points[:, 0]) and 
                    point[0] <= np.max(xy_points[:, 0]) and
                    point[1] >= np.min(xy_points[:, 1]) and 
                    point[1] <= np.max(xy_points[:, 1]) and
                    point[2] < min_z):
                    undercut_points.append(point)
            
            overhang = {
                'id': len(overhangs),
                'centroid': centroid.tolist(),
                'area': float(overhang_area),
                'height_range': [float(min_z), float(max_z)],
                'num_points': len(cluster_indices),
                'undercut_points': len(undercut_points),
                'stability_concern': len(undercut_points) > 10,
                'point_indices': cluster_indices.tolist()
            }
            
            overhangs.append(overhang)
        
        return overhangs
    
    def _analyze_weathering(self, pcd: o3d.geometry.PointCloud) -> Dict:
        """Analyze weathering indicators from point cloud"""
        points = np.asarray(pcd.points)
        
        # Compute surface roughness statistics
        roughness = self._compute_surface_roughness(pcd)
        
        weathering_indicators = {
            'surface_roughness': {
                'mean': float(np.mean(roughness)) if len(roughness) > 0 else 0.0,
                'std': float(np.std(roughness)) if len(roughness) > 0 else 0.0,
                'max': float(np.max(roughness)) if len(roughness) > 0 else 0.0
            },
            'point_density_variation': self._compute_density_variation(pcd),
            'surface_irregularity': self._compute_surface_irregularity(pcd)
        }
        
        # Overall weathering score (0-1, higher = more weathered)
        roughness_score = min(weathering_indicators['surface_roughness']['mean'] / 0.5, 1.0)
        density_score = weathering_indicators['point_density_variation']
        irregularity_score = weathering_indicators['surface_irregularity']
        
        weathering_indicators['overall_score'] = (
            roughness_score * 0.4 + 
            density_score * 0.3 + 
            irregularity_score * 0.3
        )
        
        return weathering_indicators
    
    def _compute_curvature(self, pcd: o3d.geometry.PointCloud, 
                          radius: float = 0.3) -> np.ndarray:
        """Compute surface curvature for each point"""
        points = np.asarray(pcd.points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        curvatures = np.zeros(len(points))
        
        for i, point in enumerate(points):
            [k, idx, _] = kdtree.search_radius_vector_3d(point, radius)
            
            if k > 3:  # Need at least 4 points for curvature estimation
                neighbor_points = points[idx]
                
                # Fit plane to neighbors
                centroid = np.mean(neighbor_points, axis=0)
                centered_points = neighbor_points - centroid
                
                # SVD for plane fitting
                U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
                normal = Vt[-1]  # Smallest singular vector
                
                # Compute distances to fitted plane
                distances = np.abs(np.dot(centered_points, normal))
                
                # Curvature as RMS distance to plane
                curvatures[i] = np.sqrt(np.mean(distances**2))
        
        return curvatures
    
    def _compute_density_variation(self, pcd: o3d.geometry.PointCloud) -> float:
        """Compute point density variation as weathering indicator"""
        points = np.asarray(pcd.points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        
        densities = []
        radius = 0.5
        
        # Sample subset of points for efficiency
        sample_indices = np.random.choice(len(points), min(1000, len(points)), replace=False)
        
        for i in sample_indices:
            [k, _, _] = kdtree.search_radius_vector_3d(points[i], radius)
            # Density as points per unit volume
            volume = (4/3) * np.pi * radius**3
            density = k / volume
            densities.append(density)
        
        # Coefficient of variation as measure of density variation
        if len(densities) > 0 and np.mean(densities) > 0:
            return min(np.std(densities) / np.mean(densities), 1.0)
        else:
            return 0.0
    
    def _compute_surface_irregularity(self, pcd: o3d.geometry.PointCloud) -> float:
        """Compute surface irregularity measure"""
        points = np.asarray(pcd.points)
        
        if len(points) < 10:
            return 0.0
        
        # Compute local surface deviation
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        deviations = []
        
        # Sample subset for efficiency
        sample_indices = np.random.choice(len(points), min(500, len(points)), replace=False)
        
        for i in sample_indices:
            [k, idx, _] = kdtree.search_knn_vector_3d(points[i], 10)
            
            if k > 3:
                neighbor_points = points[idx]
                
                # Fit plane to neighbors
                centroid = np.mean(neighbor_points, axis=0)
                centered_points = neighbor_points - centroid
                
                # SVD for plane fitting
                try:
                    U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
                    normal = Vt[-1]
                    
                    # Distance from center point to fitted plane
                    deviation = abs(np.dot(points[i] - centroid, normal))
                    deviations.append(deviation)
                except:
                    continue
        
        if len(deviations) > 0:
            return min(np.mean(deviations) / 0.1, 1.0)  # Normalize to [0, 1]
        else:
            return 0.0
    
    def _compute_stability_score(self, slope_angles: np.ndarray,
                               surface_roughness: np.ndarray,
                               discontinuities: List[Dict],
                               cracks: List[Dict],
                               overhangs: List[Dict],
                               weathering: Dict) -> float:
        """Compute overall stability score (0-1, higher = more stable)"""
        
        # Slope stability factor
        if len(slope_angles) > 0:
            avg_slope = np.mean(slope_angles)
            max_slope = np.max(slope_angles)
            slope_factor = max(0, 1 - (avg_slope / 45.0))  # Decrease with steeper slopes
            critical_slope_factor = max(0, 1 - (max_slope / 60.0))
        else:
            slope_factor = 1.0
            critical_slope_factor = 1.0
        
        # Discontinuity factor
        discontinuity_factor = max(0, 1 - len(discontinuities) / 10.0)
        
        # Crack factor
        crack_factor = max(0, 1 - len(cracks) / 5.0)
        
        # Overhang factor
        overhang_factor = max(0, 1 - len(overhangs) / 3.0)
        
        # Weathering factor
        weathering_factor = max(0, 1 - weathering.get('overall_score', 0))
        
        # Combined stability score
        stability = (
            slope_factor * 0.25 +
            critical_slope_factor * 0.15 +
            discontinuity_factor * 0.20 +
            crack_factor * 0.15 +
            overhang_factor * 0.15 +
            weathering_factor * 0.10
        )
        
        return max(0.0, min(1.0, stability))
    
    def _identify_risk_factors(self, slope_angles: np.ndarray,
                             surface_roughness: np.ndarray,
                             discontinuities: List[Dict],
                             cracks: List[Dict],
                             overhangs: List[Dict],
                             weathering: Dict) -> Dict:
        """Identify specific risk factors"""
        
        risk_factors = {
            'high_slope_areas': False,
            'major_discontinuities': False,
            'active_weathering': False,
            'crack_networks': False,
            'unstable_overhangs': False,
            'poor_surface_condition': False
        }
        
        # Check for high slope areas
        if len(slope_angles) > 0:
            high_slope_percentage = np.sum(slope_angles > 45) / len(slope_angles) * 100
            risk_factors['high_slope_areas'] = high_slope_percentage > 20
        
        # Check for major discontinuities
        major_discontinuities = [d for d in discontinuities if d['area'] > 10.0]
        risk_factors['major_discontinuities'] = len(major_discontinuities) > 2
        
        # Check weathering
        risk_factors['active_weathering'] = weathering.get('overall_score', 0) > 0.6
        
        # Check crack networks
        risk_factors['crack_networks'] = len(cracks) > 3
        
        # Check unstable overhangs
        unstable_overhangs = [o for o in overhangs if o['stability_concern']]
        risk_factors['unstable_overhangs'] = len(unstable_overhangs) > 0
        
        # Check surface condition
        if len(surface_roughness) > 0:
            risk_factors['poor_surface_condition'] = np.mean(surface_roughness) > 0.3
        
        return risk_factors