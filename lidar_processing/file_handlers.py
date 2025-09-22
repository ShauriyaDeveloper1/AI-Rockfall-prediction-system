"""
File handlers for various LIDAR and point cloud formats.
Supports LAS/LAZ, PLY, PCD, and other common 3D data formats.
"""

import numpy as np
import open3d as o3d
import laspy
from plyfile import PlyData, PlyElement
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PointCloudMetadata:
    """Metadata for point cloud files"""
    filename: str
    format: str
    num_points: int
    bounds: Dict[str, Tuple[float, float]]  # min, max for x, y, z
    scan_date: Optional[datetime] = None
    scanner_info: Optional[Dict] = None
    coordinate_system: Optional[str] = None

class LIDARFileHandler:
    """Handles loading and saving of various LIDAR file formats"""
    
    SUPPORTED_FORMATS = ['.las', '.laz', '.ply', '.pcd', '.xyz', '.pts']
    
    def __init__(self):
        self.metadata = {}
    
    def load_point_cloud(self, file_path: str) -> Tuple[o3d.geometry.PointCloud, PointCloudMetadata]:
        """
        Load point cloud from various formats
        
        Args:
            file_path: Path to the point cloud file
            
        Returns:
            Tuple of (Open3D PointCloud object, metadata)
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.las', '.laz']:
            return self._load_las(file_path)
        elif file_ext == '.ply':
            return self._load_ply(file_path)
        elif file_ext == '.pcd':
            return self._load_pcd(file_path)
        elif file_ext in ['.xyz', '.pts']:
            return self._load_ascii(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _load_las(self, file_path: str) -> Tuple[o3d.geometry.PointCloud, PointCloudMetadata]:
        """Load LAS/LAZ files using laspy"""
        try:
            las_file = laspy.read(file_path)
            
            # Extract coordinates
            points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Add colors if available
            if hasattr(las_file, 'red') and hasattr(las_file, 'green') and hasattr(las_file, 'blue'):
                colors = np.vstack((las_file.red, las_file.green, las_file.blue)).transpose()
                colors = colors / 65535.0  # Normalize to [0, 1]
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Extract metadata
            bounds = {
                'x': (float(np.min(las_file.x)), float(np.max(las_file.x))),
                'y': (float(np.min(las_file.y)), float(np.max(las_file.y))),
                'z': (float(np.min(las_file.z)), float(np.max(las_file.z)))
            }
            
            metadata = PointCloudMetadata(
                filename=os.path.basename(file_path),
                format='LAS/LAZ',
                num_points=len(las_file.points),
                bounds=bounds,
                coordinate_system=getattr(las_file.header, 'global_encoding', None)
            )
            
            return pcd, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load LAS file {file_path}: {str(e)}")
    
    def _load_ply(self, file_path: str) -> Tuple[o3d.geometry.PointCloud, PointCloudMetadata]:
        """Load PLY files"""
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            
            if len(pcd.points) == 0:
                raise ValueError("PLY file contains no points")
            
            points = np.asarray(pcd.points)
            bounds = {
                'x': (float(np.min(points[:, 0])), float(np.max(points[:, 0]))),
                'y': (float(np.min(points[:, 1])), float(np.max(points[:, 1]))),
                'z': (float(np.min(points[:, 2])), float(np.max(points[:, 2])))
            }
            
            metadata = PointCloudMetadata(
                filename=os.path.basename(file_path),
                format='PLY',
                num_points=len(pcd.points),
                bounds=bounds
            )
            
            return pcd, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PLY file {file_path}: {str(e)}")
    
    def _load_pcd(self, file_path: str) -> Tuple[o3d.geometry.PointCloud, PointCloudMetadata]:
        """Load PCD files"""
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            
            if len(pcd.points) == 0:
                raise ValueError("PCD file contains no points")
            
            points = np.asarray(pcd.points)
            bounds = {
                'x': (float(np.min(points[:, 0])), float(np.max(points[:, 0]))),
                'y': (float(np.min(points[:, 1])), float(np.max(points[:, 1]))),
                'z': (float(np.min(points[:, 2])), float(np.max(points[:, 2])))
            }
            
            metadata = PointCloudMetadata(
                filename=os.path.basename(file_path),
                format='PCD',
                num_points=len(pcd.points),
                bounds=bounds
            )
            
            return pcd, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PCD file {file_path}: {str(e)}")
    
    def _load_ascii(self, file_path: str) -> Tuple[o3d.geometry.PointCloud, PointCloudMetadata]:
        """Load ASCII files (XYZ, PTS)"""
        try:
            # Try to load as space/tab separated values
            data = np.loadtxt(file_path)
            
            if data.shape[1] < 3:
                raise ValueError("ASCII file must have at least 3 columns (X, Y, Z)")
            
            points = data[:, :3]
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Add colors if available (assuming XYZRGB format)
            if data.shape[1] >= 6:
                colors = data[:, 3:6]
                if np.max(colors) > 1.0:  # Assume values are in 0-255 range
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            bounds = {
                'x': (float(np.min(points[:, 0])), float(np.max(points[:, 0]))),
                'y': (float(np.min(points[:, 1])), float(np.max(points[:, 1]))),
                'z': (float(np.min(points[:, 2])), float(np.max(points[:, 2])))
            }
            
            metadata = PointCloudMetadata(
                filename=os.path.basename(file_path),
                format='ASCII',
                num_points=len(points),
                bounds=bounds
            )
            
            return pcd, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ASCII file {file_path}: {str(e)}")
    
    def save_point_cloud(self, pcd: o3d.geometry.PointCloud, file_path: str, 
                        metadata: Optional[PointCloudMetadata] = None) -> bool:
        """
        Save point cloud to file
        
        Args:
            pcd: Open3D PointCloud object
            file_path: Output file path
            metadata: Optional metadata to save alongside
            
        Returns:
            Success status
        """
        try:
            success = o3d.io.write_point_cloud(file_path, pcd)
            
            # Save metadata as separate JSON file if provided
            if metadata and success:
                metadata_path = os.path.splitext(file_path)[0] + '_metadata.json'
                self._save_metadata(metadata, metadata_path)
            
            return success
            
        except Exception as e:
            print(f"Failed to save point cloud: {str(e)}")
            return False
    
    def _save_metadata(self, metadata: PointCloudMetadata, file_path: str):
        """Save metadata to JSON file"""
        try:
            metadata_dict = {
                'filename': metadata.filename,
                'format': metadata.format,
                'num_points': metadata.num_points,
                'bounds': metadata.bounds,
                'scan_date': metadata.scan_date.isoformat() if metadata.scan_date else None,
                'scanner_info': metadata.scanner_info,
                'coordinate_system': metadata.coordinate_system
            }
            
            with open(file_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save metadata: {str(e)}")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return self.SUPPORTED_FORMATS.copy()
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate if file can be loaded
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            True if file is valid and can be loaded
        """
        if not os.path.exists(file_path):
            return False
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            return False
        
        try:
            pcd, metadata = self.load_point_cloud(file_path)
            return len(pcd.points) > 0
        except:
            return False