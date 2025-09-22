import React, { useState, useEffect, useRef } from 'react';
import { Container, Row, Col, Card, Button, Alert, Table, Badge, Spinner } from 'react-bootstrap';
import axios from 'axios';
import { FaUpload, FaEye, FaCube, FaChartLine } from 'react-icons/fa';
import * as THREE from 'three';

// Simple 3D Point Cloud Visualization using vanilla Three.js
function PointCloudVisualization({ pointsData, analysis }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  
  useEffect(() => {
    if (!mountRef.current || !pointsData) return;
    
    // Initialize Three.js scene
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, 400 / 300, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    
    renderer.setSize(400, 300);
    renderer.setClearColor(0x222222);
    mountRef.current.appendChild(renderer.domElement);
    
    // Create point cloud
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];
    
    pointsData.forEach(point => {
      positions.push(point.x, point.y, point.z);
      
      // Color based on risk level
      const riskColor = getRiskColor(point.risk || 0);
      colors.push(riskColor.r, riskColor.g, riskColor.b);
    });
    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({ 
      size: 0.02, 
      vertexColors: true,
      sizeAttenuation: true
    });
    
    const points = new THREE.Points(geometry, material);
    scene.add(points);
    
    // Position camera
    camera.position.set(5, 5, 5);
    camera.lookAt(0, 0, 0);
    
    // Add basic lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);
    
    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      points.rotation.y += 0.005;
      renderer.render(scene, camera);
    };
    
    animate();
    
    // Store references
    sceneRef.current = scene;
    rendererRef.current = renderer;
    
    // Cleanup
    return () => {
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [pointsData]);
  
  const getRiskColor = (risk) => {
    // Color interpolation based on risk (0-1)
    if (risk < 0.3) return { r: 0, g: 1, b: 0 }; // Green (low risk)
    if (risk < 0.6) return { r: 1, g: 1, b: 0 }; // Yellow (medium risk)
    if (risk < 0.8) return { r: 1, g: 0.5, b: 0 }; // Orange (high risk)
    return { r: 1, g: 0, b: 0 }; // Red (critical risk)
  };
  
  if (!pointsData) {
    return (
      <div style={{ width: 400, height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#222' }}>
        <p style={{ color: 'white' }}>No point cloud data</p>
      </div>
    );
  }
  
  return <div ref={mountRef} style={{ width: 400, height: 300 }} />;
}

// Risk Level Badge Component
function RiskBadge({ riskLevel, probability }) {
  const getBadgeVariant = (level) => {
    switch (level) {
      case 'LOW': return 'success';
      case 'MEDIUM': return 'warning';
      case 'HIGH': return 'danger';
      case 'CRITICAL': return 'dark';
      default: return 'secondary';
    }
  };

  return (
    <Badge bg={getBadgeVariant(riskLevel)} className="me-2">
      {riskLevel} ({probability ? `${probability.toFixed(1)}%` : 'N/A'})
    </Badge>
  );
}

// Main LIDAR Visualization Component
function LIDARVisualization() {
  const [scans, setScans] = useState([]);
  const [selectedScan, setSelectedScan] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);
  const fileInputRef = useRef(null);

  // Load available LIDAR scans
  useEffect(() => {
    loadScans();
  }, []);

  const loadScans = async () => {
    try {
      const response = await axios.get('/api/lidar/scans');
      // Ensure scans is always an array
      const scansData = Array.isArray(response.data) ? response.data : 
                        (response.data?.scans && Array.isArray(response.data.scans)) ? response.data.scans : [];
      setScans(scansData);
    } catch (err) {
      setError('Failed to load LIDAR scans');
      console.error(err);
      // Set empty array on error to prevent find() errors
      setScans([]);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('scan_location', 'User Upload');
      formData.append('scanner_type', 'Unknown');
      formData.append('notes', `Uploaded: ${file.name}`);

      const response = await axios.post('/api/lidar/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
        },
      });

      if (response.data.status === 'success') {
        await loadScans();
        setSelectedScan(response.data.scan_id);
        setUploadProgress(null);
      }
    } catch (err) {
      setError(`Upload failed: ${err.response?.data?.message || err.message}`);
      setUploadProgress(null);
    } finally {
      setLoading(false);
    }
  };

  const analyzeScan = async (scanId) => {
    if (!scanId) return;

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`/api/lidar/analyze/${scanId}`);
      setAnalysis(response.data);
    } catch (err) {
      setError(`Analysis failed: ${err.response?.data?.message || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const generateSampleData = () => {
    // Generate sample point cloud for demonstration
    const samplePoints = [];
    for (let i = 0; i < 1000; i++) {
      samplePoints.push({
        x: (Math.random() - 0.5) * 10,
        y: (Math.random() - 0.5) * 10,
        z: (Math.random() - 0.5) * 10,
        risk: Math.random()
      });
    }
    return samplePoints;
  };

  const selectedScanData = Array.isArray(scans) ? scans.find(scan => scan.id === selectedScan) : null;

  return (
    <Container fluid className="py-4">
      <Row className="mb-4">
        <Col>
          <Card>
            <Card.Header className="d-flex justify-content-between align-items-center">
              <h4 className="mb-0">
                <FaCube className="me-2" />
                LIDAR 3D Point Cloud Visualization
              </h4>
              <Button
                variant="primary"
                onClick={() => fileInputRef.current?.click()}
                disabled={loading}
              >
                <FaUpload className="me-2" />
                Upload LIDAR File
              </Button>
            </Card.Header>
            <Card.Body>
              {error && (
                <Alert variant="danger" dismissible onClose={() => setError(null)}>
                  {error}
                </Alert>
              )}

              {uploadProgress !== null && (
                <Alert variant="info">
                  Upload Progress: {uploadProgress}%
                </Alert>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept=".ply,.pcd,.xyz,.las,.laz"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
              />

              <Row>
                <Col md={6}>
                  <h5>Available LIDAR Scans</h5>
                  {scans.length === 0 ? (
                    <Alert variant="info">
                      No LIDAR scans available. Upload a point cloud file to get started.
                      <br />
                      <small>Supported formats: PLY, PCD, XYZ, LAS, LAZ</small>
                    </Alert>
                  ) : (
                    <Table striped bordered hover size="sm">
                      <thead>
                        <tr>
                          <th>ID</th>
                          <th>Location</th>
                          <th>Date</th>
                          <th>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {scans.map(scan => (
                          <tr key={scan.id}>
                            <td>{scan.id}</td>
                            <td>{scan.scan_location}</td>
                            <td>{new Date(scan.upload_timestamp).toLocaleDateString()}</td>
                            <td>
                              <Button
                                size="sm"
                                variant="outline-primary"
                                onClick={() => setSelectedScan(scan.id)}
                                className="me-2"
                              >
                                <FaEye />
                              </Button>
                              <Button
                                size="sm"
                                variant="outline-success"
                                onClick={() => analyzeScan(scan.id)}
                                disabled={loading}
                              >
                                <FaChartLine />
                              </Button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  )}
                </Col>

                <Col md={6}>
                  <h5>3D Visualization</h5>
                  <Card>
                    <Card.Body className="text-center">
                      {loading ? (
                        <div className="d-flex justify-content-center align-items-center" style={{ height: 300 }}>
                          <Spinner animation="border" />
                        </div>
                      ) : selectedScanData ? (
                        <PointCloudVisualization 
                          pointsData={generateSampleData()} 
                          analysis={analysis}
                        />
                      ) : (
                        <div className="d-flex justify-content-center align-items-center" style={{ height: 300 }}>
                          <p className="text-muted">Select a LIDAR scan to visualize</p>
                        </div>
                      )}
                      
                      {selectedScanData && (
                        <div className="mt-2">
                          <small className="text-muted">
                            Scan: {selectedScanData.scan_location} | 
                            Points: {selectedScanData.num_points?.toLocaleString() || 'Unknown'}
                          </small>
                        </div>
                      )}
                    </Card.Body>
                  </Card>
                </Col>
              </Row>

              {analysis && (
                <Row className="mt-4">
                  <Col>
                    <Card>
                      <Card.Header>
                        <h5>Geological Analysis Results</h5>
                      </Card.Header>
                      <Card.Body>
                        <Row>
                          <Col md={3}>
                            <h6>Risk Assessment</h6>
                            <RiskBadge 
                              riskLevel={analysis.risk_level} 
                              probability={analysis.stability_score * 100}
                            />
                          </Col>
                          <Col md={3}>
                            <h6>Stability Score</h6>
                            <p className="mb-0">{analysis.stability_score?.toFixed(3) || 'N/A'}</p>
                          </Col>
                          <Col md={6}>
                            <h6>Features Detected</h6>
                            <p className="mb-1">
                              <strong>Discontinuity Planes:</strong> {analysis.features?.discontinuity_planes || 0}
                            </p>
                            <p className="mb-1">
                              <strong>Crack Features:</strong> {analysis.features?.crack_features || 0}
                            </p>
                            <p className="mb-0">
                              <strong>Overhang Regions:</strong> {analysis.features?.overhang_regions || 0}
                            </p>
                          </Col>
                        </Row>
                      </Card.Body>
                    </Card>
                  </Col>
                </Row>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default LIDARVisualization;