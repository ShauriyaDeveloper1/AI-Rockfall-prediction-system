import React, { useState, useEffect, useRef } from 'react';
import { Container, Row, Col, Card, Alert, Button, ButtonGroup, Form, InputGroup, Modal } from 'react-bootstrap';
import { MapContainer, TileLayer, Circle, Popup, Marker, useMapEvents } from 'react-leaflet';
import { FaMap, FaSync, FaDownload, FaMapMarkerAlt, FaSearch, FaCrosshairs } from 'react-icons/fa';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

      {/* Location Selection Controls */}
      <Row className="mb-3">
        <Col>
          <Card className="modern-card">
            <Card.Header className="modern-card-header">
              <h5 className="mb-0 text-primary">
                <FaMapMarkerAlt className="me-2" />
                Location Selection
              </h5>
            </Card.Header>
            <Card.Body>
              <Row>
                <Col md={4}>
                  <Button
                    variant={isSelectingLocation ? "success" : "outline-primary"}
                    onClick={() => setIsSelectingLocation(!isSelectingLocation)}
                    className="w-100 mb-2"
                  >
                    <FaMap className="me-2" />
                    {isSelectingLocation ? 'Click on Map' : 'Select on Map'}
                  </Button>
                </Col>
                <Col md={4}>
                  <Button
                    variant="outline-primary"
                    onClick={() => setShowLocationModal(true)}
                    className="w-100 mb-2"
                  >
                    <FaSearch className="me-2" />
                    Search Location
                  </Button>
                </Col>
                <Col md={4}>
                  <Button
                    variant="outline-primary"
                    onClick={getCurrentLocation}
                    className="w-100 mb-2"
                  >
                    <FaCrosshairs className="me-2" />
                    Current Location
                  </Button>
                </Col>
              </Row>
              {selectedLocation && (
                <Alert variant="info" className="mt-3 mb-0">
                  <strong>Selected Location:</strong> 
                  <br />
                  Latitude: {selectedLocation.lat.toFixed(6)}
                  <br />
                  Longitude: {selectedLocation.lng.toFixed(6)}
                </Alert>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>MapEvents } from 'react-leaflet';
import { FaMap, FaSync, FaDownload, FaMapMarkerAlt, FaSearch, FaCrosshairs } from 'react-icons/fa';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for default markers in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

// Map click handler component
function MapClickHandler({ onLocationSelect, isSelectingLocation }) {
  useMapEvents({
    click: (e) => {
      if (isSelectingLocation) {
        onLocationSelect(e.latlng);
      }
    },
  });
  return null;
}

const RiskMap = () => {
  const [riskZones, setRiskZones] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [mapView, setMapView] = useState('risk'); // 'risk', 'sensors', 'alerts'
  const [lastUpdate, setLastUpdate] = useState(null);
  
  // Location selection features
  const [isSelectingLocation, setIsSelectingLocation] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [showLocationModal, setShowLocationModal] = useState(false);
  const [coordinateInput, setCoordinateInput] = useState({ lat: '', lng: '' });
  const [locationSearch, setLocationSearch] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  
  const mapRef = useRef();

  // Default center coordinates (can be configured for specific mine location)
  const [mapCenter, setMapCenter] = useState([-23.5505, -46.6333]);
  const [mapZoom, setMapZoom] = useState(15);

  useEffect(() => {
    fetchRiskMapData();
    const interval = setInterval(fetchRiskMapData, 60000); // Update every minute
    return () => clearInterval(interval);
  }, []);

  const fetchRiskMapData = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/risk-map');
      setRiskZones(response.data.risk_zones);
      setLastUpdate(new Date(response.data.timestamp));
      setError(null);
    } catch (err) {
      setError('Failed to fetch risk map data');
      console.error('Risk map error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleLocationSelect = (latlng) => {
    setSelectedLocation(latlng);
    setCoordinateInput({ lat: latlng.lat.toFixed(6), lng: latlng.lng.toFixed(6) });
    setIsSelectingLocation(false);
  };

  const handleCoordinateSubmit = () => {
    const lat = parseFloat(coordinateInput.lat);
    const lng = parseFloat(coordinateInput.lng);
    
    if (isNaN(lat) || isNaN(lng)) {
      alert('Please enter valid coordinates');
      return;
    }
    
    if (lat < -90 || lat > 90) {
      alert('Latitude must be between -90 and 90');
      return;
    }
    
    if (lng < -180 || lng > 180) {
      alert('Longitude must be between -180 and 180');
      return;
    }
    
    const newLocation = { lat, lng };
    setSelectedLocation(newLocation);
    setMapCenter([lat, lng]);
    setMapZoom(16);
    setShowLocationModal(false);
  };

  const searchLocation = async () => {
    if (!locationSearch.trim()) return;
    
    setSearchLoading(true);
    try {
      // Using Nominatim API for location search
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(locationSearch)}&limit=5`
      );
      const results = await response.json();
      setSearchResults(results);
    } catch (error) {
      console.error('Location search error:', error);
      alert('Location search failed. Please try again.');
    } finally {
      setSearchLoading(false);
    }
  };

  const selectSearchResult = (result) => {
    const lat = parseFloat(result.lat);
    const lng = parseFloat(result.lon);
    const newLocation = { lat, lng };
    
    setSelectedLocation(newLocation);
    setMapCenter([lat, lng]);
    setMapZoom(16);
    setCoordinateInput({ lat: lat.toFixed(6), lng: lng.toFixed(6) });
    setSearchResults([]);
    setLocationSearch('');
    setShowLocationModal(false);
  };

  const getCurrentLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const lat = position.coords.latitude;
          const lng = position.coords.longitude;
          const newLocation = { lat, lng };
          
          setSelectedLocation(newLocation);
          setMapCenter([lat, lng]);
          setMapZoom(16);
          setCoordinateInput({ lat: lat.toFixed(6), lng: lng.toFixed(6) });
        },
        (error) => {
          alert('Unable to get current location. Please check location permissions.');
          console.error('Geolocation error:', error);
        }
      );
    } else {
      alert('Geolocation is not supported by this browser.');
    }
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'LOW': return '#28a745';
      case 'MEDIUM': return '#ffc107';
      case 'HIGH': return '#fd7e14';
      case 'CRITICAL': return '#dc3545';
      default: return '#6c757d';
    }
  };

  const getRiskRadius = (riskValue) => {
    return Math.max(20, riskValue * 100); // Minimum 20m, max 100m radius
  };

  const exportMapData = () => {
    const dataStr = JSON.stringify(riskZones, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `risk_map_${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  return (
    <Container fluid className="mt-4 fade-in">
      <Row className="mb-4">
        <Col>
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <h1 className="display-5 text-primary">
                <FaMap className="me-3" />
                Risk Map
              </h1>
              <p className="lead">Real-time visualization of rockfall risk zones</p>
            </div>
            <div>
              <ButtonGroup className="me-2">
                <Button 
                  variant={mapView === 'risk' ? 'primary' : 'outline-primary'}
                  onClick={() => setMapView('risk')}
                  className="modern-btn"
                >
                  Risk Zones
                </Button>
                <Button 
                  variant={mapView === 'sensors' ? 'primary' : 'outline-primary'}
                  onClick={() => setMapView('sensors')}
                  className="modern-btn"
                >
                  Sensors
                </Button>
              </ButtonGroup>
              <Button variant="success" onClick={fetchRiskMapData} className="me-2 modern-btn">
                <FaSync className="me-1" />
                Refresh
              </Button>
              <Button variant="info" onClick={exportMapData} className="modern-btn">
                <FaDownload className="me-1" />
                Export
              </Button>
            </div>
          </div>
        </Col>
      </Row>

      {error && (
        <Row className="mb-4">
          <Col>
            <Alert variant="danger" className="modern-alert">{error}</Alert>
          </Col>
        </Row>
      )}

      <Row>
        <Col lg={9}>
          <Card className="modern-card slide-in">
            <Card.Header className="modern-card-header">
              <div className="d-flex justify-content-between align-items-center">
                <h5 className="mb-0 text-primary">Interactive Risk Map</h5>
                {lastUpdate && (
                  <small className="text-muted">
                    Last updated: {lastUpdate.toLocaleString()}
                  </small>
                )}
              </div>
            </Card.Header>
            <Card.Body style={{ height: '600px', padding: 0 }}>
              {loading ? (
                <div className="d-flex justify-content-center align-items-center h-100">
                  <div className="spinner-border text-primary" role="status">
                    <span className="visually-hidden">Loading map...</span>
                  </div>
                </div>
              ) : (
                <MapContainer
                  center={defaultCenter}
                  zoom={defaultZoom}
                  style={{ height: '100%', width: '100%' }}
                  className="modern-map"
                >
                  <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  />
                  
                  {riskZones.map((zone, index) => (
                    <Circle
                      key={index}
                      center={[zone.lat, zone.lng]}
                      radius={getRiskRadius(zone.risk_value)}
                      pathOptions={{
                        color: getRiskColor(zone.risk_level),
                        fillColor: getRiskColor(zone.risk_level),
                        fillOpacity: 0.4,
                        weight: 2
                      }}
                    >
                      <Popup>
                        <div className="modern-popup">
                          <h6 className="text-primary">Risk Zone {index + 1}</h6>
                          <p><strong>Risk Level:</strong> <span className={`risk-${zone.risk_level.toLowerCase()}`}>{zone.risk_level}</span></p>
                          <p><strong>Risk Value:</strong> {(zone.risk_value * 100).toFixed(1)}%</p>
                          <p><strong>Coordinates:</strong> {zone.lat.toFixed(4)}, {zone.lng.toFixed(4)}</p>
                        </div>
                      </Popup>
                    </Circle>
                  ))}
                  
                  {/* Add sensor markers if in sensor view */}
                  {mapView === 'sensors' && (
                    <>
                      <Marker position={[-23.5505, -46.6333]}>
                        <Popup>
                          <div className="modern-popup">
                            <h6 className="text-primary">Displacement Sensor DS-001</h6>
                            <p><strong>Status:</strong> <span className="text-success">Active</span></p>
                            <p><strong>Last Reading:</strong> 0.8mm</p>
                          </div>
                        </Popup>
                      </Marker>
                      <Marker position={[-23.5515, -46.6343]}>
                        <Popup>
                          <div className="modern-popup">
                            <h6 className="text-primary">Strain Gauge SG-002</h6>
                            <p><strong>Status:</strong> <span className="text-success">Active</span></p>
                            <p><strong>Last Reading:</strong> 120 μstrain</p>
                          </div>
                        </Popup>
                      </Marker>
                    </>
                  )}
                </MapContainer>
              )}
            </Card.Body>
          </Card>
        </Col>

        <Col lg={3}>
          <Card className="mb-3 modern-card slide-in">
            <Card.Header className="modern-card-header">
              <h6 className="mb-0 text-primary">Risk Legend</h6>
            </Card.Header>
            <Card.Body>
              <div className="d-flex align-items-center mb-2">
                <div 
                  style={{ 
                    width: '20px', 
                    height: '20px', 
                    backgroundColor: '#28a745', 
                    marginRight: '10px',
                    borderRadius: '50%',
                    boxShadow: '0 0 10px rgba(40, 167, 69, 0.5)'
                  }}
                ></div>
                <span className="text-light">Low Risk (0-25%)</span>
              </div>
              <div className="d-flex align-items-center mb-2">
                <div 
                  style={{ 
                    width: '20px', 
                    height: '20px', 
                    backgroundColor: '#ffc107', 
                    marginRight: '10px',
                    borderRadius: '50%',
                    boxShadow: '0 0 10px rgba(255, 193, 7, 0.5)'
                  }}
                ></div>
                <span className="text-light">Medium Risk (25-50%)</span>
              </div>
              <div className="d-flex align-items-center mb-2">
                <div 
                  style={{ 
                    width: '20px', 
                    height: '20px', 
                    backgroundColor: '#fd7e14', 
                    marginRight: '10px',
                    borderRadius: '50%',
                    boxShadow: '0 0 10px rgba(253, 126, 20, 0.5)'
                  }}
                ></div>
                <span className="text-light">High Risk (50-75%)</span>
              </div>
              <div className="d-flex align-items-center">
                <div 
                  style={{ 
                    width: '20px', 
                    height: '20px', 
                    backgroundColor: '#dc3545', 
                    marginRight: '10px',
                    borderRadius: '50%',
                    boxShadow: '0 0 10px rgba(220, 53, 69, 0.5)'
                  }}
                ></div>
                <span className="text-light">Critical Risk (75-100%)</span>
              </div>
            </Card.Body>
          </Card>

          <Card className="modern-card slide-in">
            <Card.Header className="modern-card-header">
              <h6 className="mb-0 text-primary">Zone Statistics</h6>
            </Card.Header>
            <Card.Body>
              {riskZones.length > 0 ? (
                <div className="modern-stats">
                  <p><strong className="text-primary">Total Zones:</strong> <span className="text-light">{riskZones.length}</span></p>
                  <p><strong className="text-danger">High Risk Zones:</strong> <span className="text-light">{riskZones.filter(z => z.risk_level === 'HIGH' || z.risk_level === 'CRITICAL').length}</span></p>
                  <p><strong className="text-warning">Average Risk:</strong> <span className="text-light">{(riskZones.reduce((sum, z) => sum + z.risk_value, 0) / riskZones.length * 100).toFixed(1)}%</span></p>
                  <p><strong className="text-info">Max Risk:</strong> <span className="text-light">{(Math.max(...riskZones.map(z => z.risk_value)) * 100).toFixed(1)}%</span></p>
                </div>
              ) : (
                <p className="text-muted">No zone data available</p>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default RiskMap;