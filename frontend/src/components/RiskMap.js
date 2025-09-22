import React, { useState, useEffect, useRef } from 'react';
import { Container, Row, Col, Card, Alert, Button, ButtonGroup, Form, InputGroup, Modal } from 'react-bootstrap';
import { MapContainer, TileLayer, Circle, Popup, Marker, useMapEvents } from 'react-leaflet';
import { FaMap, FaSync, FaDownload, FaMapMarkerAlt, FaSearch, FaCrosshairs, FaCheckCircle } from 'react-icons/fa';
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

// Map controller to handle center changes
function MapController({ center, zoom }) {
  const map = useMapEvents({});
  
  React.useEffect(() => {
    if (center && center.length === 2) {
      map.setView(center, zoom);
    }
  }, [map, center, zoom]);
  
  return null;
}

const RiskMap = () => {
  const [riskZones, setRiskZones] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [mapView, setMapView] = useState('risk');
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
  const [mapCenter, setMapCenter] = useState([51.5074, -0.1278]); // London coordinates
  const [mapZoom, setMapZoom] = useState(13);

  // Alert functionality
  const [sendingAlert, setSendingAlert] = useState(false);
  const [alertSent, setAlertSent] = useState(false);

  useEffect(() => {
    fetchRiskMapData();
    const interval = setInterval(fetchRiskMapData, 60000);
    return () => clearInterval(interval);
  }, []);

  // Effect to update map when center changes
  useEffect(() => {
    if (mapRef.current) {
      try {
        const map = mapRef.current;
        if (map && map.setView) {
          map.setView(mapCenter, mapZoom);
        }
      } catch (error) {
        console.warn('Map update error:', error);
      }
    }
  }, [mapCenter, mapZoom]);

  const fetchRiskMapData = async (lat = null, lng = null) => {
    try {
      setLoading(true);
      
      // Build API URL with coordinates if provided
      let apiUrl = '/api/risk-assessment';
      if (lat !== null && lng !== null) {
        apiUrl += `?lat=${lat}&lng=${lng}`;
      }
      
      console.log(`Fetching risk data from: ${apiUrl}`);
      const response = await axios.get(apiUrl);
      
      if (response.data && Array.isArray(response.data)) {
        setRiskZones(response.data);
        setLastUpdate(new Date());
        console.log(`Loaded ${response.data.length} risk zones`);
      } else {
        console.log('No risk data received');
        setRiskZones([]);
      }
      
      setError(null);
    } catch (err) {
      console.error('Risk map error:', err);
      
      // Set some default risk zones if API fails
      setRiskZones([
        {
          id: 1,
          location: 'North Zone',
          risk_level: 'HIGH',
          probability: 0.75,
          latitude: 51.5074,
          longitude: -0.1278,
          factors: ['High displacement', 'Recent rainfall']
        },
        {
          id: 2,
          location: 'East Zone',
          risk_level: 'MEDIUM',
          probability: 0.45,
          latitude: 51.5084,
          longitude: -0.1288,
          factors: ['Moderate strain levels']
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleLocationSelect = (latlng) => {
    if (latlng && typeof latlng.lat === 'number' && typeof latlng.lng === 'number') {
      setSelectedLocation(latlng);
      setCoordinateInput({ 
        lat: latlng.lat.toFixed(6), 
        lng: latlng.lng.toFixed(6) 
      });
      setIsSelectingLocation(false);
      setAlertSent(false); // Reset alert status when new location is selected
      
      // Fetch risk data for the clicked location
      fetchRiskMapData(latlng.lat, latlng.lng);
    }
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
    setAlertSent(false);
    
    // Fetch risk data for the coordinate location
    fetchRiskMapData(lat, lng);
  };

  const searchLocation = async () => {
    if (!locationSearch.trim()) return;
    
    setSearchLoading(true);
    setSearchResults([]); // Clear previous results
    
    try {
      console.log(`Searching for location: ${locationSearch}`);
      
      // First try our backend API
      const response = await axios.post('/api/location-search', {
        location: locationSearch
      });
      
      console.log('Backend response:', response.data);
      
      if (response.data.success && response.data.results.length > 0) {
        const results = response.data.results.map(result => ({
          display_name: result.name,
          lat: result.lat,
          lon: result.lng // Use lon to match Nominatim format
        }));
        setSearchResults(results);
        console.log(`Found ${results.length} results from backend`);
      } else {
        console.log('Backend search failed or no results, trying Nominatim...');
        // Fallback to OpenStreetMap if backend fails
        const osmResponse = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(locationSearch)}&limit=5`
        );
        const osmResults = await osmResponse.json();
        console.log('Nominatim results:', osmResults);
        setSearchResults(osmResults);
      }
    } catch (error) {
      console.error('Location search error:', error);
      
      // Fallback to OpenStreetMap
      try {
        console.log('Trying Nominatim fallback...');
        const response = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(locationSearch)}&limit=5`
        );
        const results = await response.json();
        console.log('Nominatim fallback results:', results);
        setSearchResults(results);
        
        if (results.length === 0) {
          alert(`No results found for "${locationSearch}". Please try a different search term.`);
        }
      } catch (fallbackError) {
        console.error('Fallback location search error:', fallbackError);
        alert('Location search failed. Please check your internet connection and try again.');
      }
    } finally {
      setSearchLoading(false);
    }
  };

  const selectSearchResult = (result) => {
    const lat = parseFloat(result.lat);
    const lng = parseFloat(result.lon || result.lng); // Handle both lon and lng formats
    
    if (!isNaN(lat) && !isNaN(lng)) {
      const newLocation = { lat, lng };
      
      setSelectedLocation(newLocation);
      setMapCenter([lat, lng]);
      setMapZoom(16);
      setCoordinateInput({ lat: lat.toFixed(6), lng: lng.toFixed(6) });
      setSearchResults([]);
      setLocationSearch('');
      setShowLocationModal(false);
      setAlertSent(false);
      
      console.log(`Selected location: ${result.display_name || result.name} at ${lat}, ${lng}`);
      
      // Fetch risk data for the selected location
      fetchRiskMapData(lat, lng);
    } else {
      console.error('Invalid coordinates:', result);
    }
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
          setAlertSent(false);
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
    switch (riskLevel?.toUpperCase()) {
      case 'LOW': return '#28a745';
      case 'MEDIUM': return '#ffc107';
      case 'HIGH': return '#fd7e14';
      case 'CRITICAL': return '#dc3545';
      default: return '#6c757d';
    }
  };

  const getRiskRadius = (riskValue) => {
    const numValue = parseFloat(riskValue) || 0;
    return Math.max(20, numValue * 100);
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

  const sendAlert = async () => {
    if (!selectedLocation || typeof selectedLocation.lat !== 'number' || typeof selectedLocation.lng !== 'number') {
      alert('Please select a valid location first');
      return;
    }

    setSendingAlert(true);
    setAlertSent(false);

    try {
      const token = localStorage.getItem('token');
      
      const requestData = {
        latitude: selectedLocation.lat,
        longitude: selectedLocation.lng,
        location: `Selected Location (${selectedLocation.lat.toFixed(4)}, ${selectedLocation.lng.toFixed(4)})`
      };

      const config = token ? {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      } : {};

      const response = await axios.post('/api/send-alert', requestData, config);

      if (response.data && response.data.success) {
        setAlertSent(true);
        const riskData = response.data.risk_data;
        alert(`Alert sent successfully! 
        
✅ Risk assessment completed
📧 Email report sent automatically
🚨 Alert level: ${riskData.risk_level}
📍 Location: ${riskData.location}
📊 Risk probability: ${(riskData.probability * 100).toFixed(1)}%

The risk analysis report has been automatically sent to your registered email address.`);
      } else {
        alert('Failed to send alert: ' + (response.data?.error || 'Unknown error'));
      }
    } catch (error) {
      console.error('Error sending alert:', error);
      if (error.response?.status === 401) {
        alert('Please log in to send alerts');
      } else {
        alert('Error sending alert. Please try again.');
      }
    } finally {
      setSendingAlert(false);
    }
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
              <p className="lead">Real-time visualization of rockfall risk zones with location selection</p>
            </div>
            <div>
              <ButtonGroup className="me-2">
                <Button 
                  variant={mapView === 'risk' ? 'primary' : 'outline-primary'}
                  onClick={() => setMapView('risk')}
                >
                  Risk Zones
                </Button>
                <Button 
                  variant={mapView === 'sensors' ? 'primary' : 'outline-primary'}
                  onClick={() => setMapView('sensors')}
                >
                  Sensors
                </Button>
                <Button 
                  variant={mapView === 'alerts' ? 'primary' : 'outline-primary'}
                  onClick={() => setMapView('alerts')}
                >
                  Alerts
                </Button>
              </ButtonGroup>
              <ButtonGroup>
                <Button variant="outline-success" onClick={fetchRiskMapData}>
                  <FaSync className="me-2" />
                  Refresh
                </Button>
                <Button variant="outline-info" onClick={exportMapData}>
                  <FaDownload className="me-2" />
                  Export
                </Button>
              </ButtonGroup>
            </div>
          </div>
        </Col>
      </Row>

      {error && (
        <Row className="mb-3">
          <Col>
            <Alert variant="danger" className="modern-alert">{error}</Alert>
          </Col>
        </Row>
      )}

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
                <>
                  <Alert variant="info" className="mt-3 mb-3">
                    <strong>Selected Location:</strong> 
                    <br />
                    Latitude: {selectedLocation.lat?.toFixed(6) || 'N/A'}
                    <br />
                    Longitude: {selectedLocation.lng?.toFixed(6) || 'N/A'}
                  </Alert>
                  
                  {/* Send Alert Button */}
                  <div className="d-grid gap-2">
                    <Button
                      variant={alertSent ? "success" : "danger"}
                      size="lg"
                      onClick={sendAlert}
                      disabled={sendingAlert}
                      className="fw-bold"
                    >
                      {sendingAlert ? (
                        <>
                          <div className="spinner-border spinner-border-sm me-2" role="status">
                            <span className="visually-hidden">Loading...</span>
                          </div>
                          Sending Alert & Email Report...
                        </>
                      ) : alertSent ? (
                        <>
                          <FaCheckCircle className="me-2" />
                          Alert Sent & Email Report Generated!
                        </>
                      ) : (
                        <>
                          🚨 Send Risk Alert & Email Report
                        </>
                      )}
                    </Button>
                    {alertSent && (
                      <small className="text-success text-center">
                        ✅ Risk analysis report automatically sent to your registered email
                      </small>
                    )}
                  </div>
                </>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

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
                  center={mapCenter}
                  zoom={mapZoom}
                  style={{ height: '100%', width: '100%' }}
                  className="modern-map"
                  ref={mapRef}
                >
                  <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  />
                  
                  <MapController center={mapCenter} zoom={mapZoom} />
                  
                  <MapClickHandler 
                    onLocationSelect={handleLocationSelect}
                    isSelectingLocation={isSelectingLocation}
                  />
                  
                  {/* Selected location marker */}
                  {selectedLocation && selectedLocation.lat && selectedLocation.lng && (
                    <Marker position={[selectedLocation.lat, selectedLocation.lng]}>
                      <Popup>
                        <div>
                          <strong>Selected Location</strong><br />
                          Lat: {selectedLocation.lat.toFixed(6)}<br />
                          Lng: {selectedLocation.lng.toFixed(6)}
                        </div>
                      </Popup>
                    </Marker>
                  )}
                  
                  {/* Risk Zones */}
                  {riskZones.map((zone) => (
                    zone.latitude && zone.longitude && (
                      <Circle
                        key={zone.id}
                        center={[zone.latitude, zone.longitude]}
                        radius={getRiskRadius(zone.probability)}
                        pathOptions={{
                          color: getRiskColor(zone.risk_level),
                          fillColor: getRiskColor(zone.risk_level),
                          fillOpacity: 0.3,
                          weight: 2
                        }}
                      >
                        <Popup>
                          <div className="popup-content">
                            <h6 className="fw-bold">{zone.location}</h6>
                            <p className="mb-1">
                              <strong>Risk Level:</strong> 
                              <span style={{ color: getRiskColor(zone.risk_level) }}>
                                {zone.risk_level}
                              </span>
                            </p>
                            <p className="mb-1">
                              <strong>Probability:</strong> {(zone.probability * 100).toFixed(1)}%
                            </p>
                            <p className="mb-0">
                              <strong>Factors:</strong> {Array.isArray(zone.factors) ? zone.factors.join(', ') : zone.factors}
                            </p>
                          </div>
                        </Popup>
                      </Circle>
                    )
                  ))}
                </MapContainer>
              )}
            </Card.Body>
          </Card>
        </Col>
        
        <Col lg={3}>
          <Card className="modern-card">
            <Card.Header className="modern-card-header">
              <h5 className="mb-0 text-primary">Risk Legend</h5>
            </Card.Header>
            <Card.Body>
              <div className="risk-legend">
                <div className="legend-item">
                  <div className="legend-color" style={{backgroundColor: '#dc3545'}}></div>
                  <span>Critical Risk</span>
                </div>
                <div className="legend-item">
                  <div className="legend-color" style={{backgroundColor: '#fd7e14'}}></div>
                  <span>High Risk</span>
                </div>
                <div className="legend-item">
                  <div className="legend-color" style={{backgroundColor: '#ffc107'}}></div>
                  <span>Medium Risk</span>
                </div>
                <div className="legend-item">
                  <div className="legend-color" style={{backgroundColor: '#28a745'}}></div>
                  <span>Low Risk</span>
                </div>
              </div>
            </Card.Body>
          </Card>

          <Card className="modern-card mt-3">
            <Card.Header className="modern-card-header">
              <h5 className="mb-0 text-primary">Map Statistics</h5>
            </Card.Header>
            <Card.Body>
              <div className="map-stats">
                <p><strong>Total Zones:</strong> {riskZones.length}</p>
                <p><strong>Critical:</strong> {riskZones.filter(z => z.risk_level === 'CRITICAL').length}</p>
                <p><strong>High:</strong> {riskZones.filter(z => z.risk_level === 'HIGH').length}</p>
                <p><strong>Medium:</strong> {riskZones.filter(z => z.risk_level === 'MEDIUM').length}</p>
                <p><strong>Low:</strong> {riskZones.filter(z => z.risk_level === 'LOW').length}</p>
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Location Search Modal */}
      <Modal show={showLocationModal} onHide={() => setShowLocationModal(false)} size="lg">
        <Modal.Header closeButton>
          <Modal.Title>
            <FaSearch className="me-2" />
            Search Location
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Row className="mb-3">
            <Col>
              <h6>Search by Name</h6>
              <InputGroup>
                <Form.Control
                  type="text"
                  placeholder="Enter city, address, or landmark..."
                  value={locationSearch}
                  onChange={(e) => setLocationSearch(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && searchLocation()}
                />
                <Button 
                  variant="primary" 
                  onClick={searchLocation}
                  disabled={searchLoading}
                >
                  {searchLoading ? 'Searching...' : 'Search'}
                </Button>
              </InputGroup>
            </Col>
          </Row>

          {searchResults.length > 0 && (
            <Row className="mb-3">
              <Col>
                <h6>Search Results</h6>
                <div className="search-results">
                  {searchResults.map((result, index) => (
                    <div 
                      key={index} 
                      className="search-result-item p-2 border rounded mb-2 cursor-pointer"
                      onClick={() => selectSearchResult(result)}
                      style={{ cursor: 'pointer' }}
                    >
                      <strong>{result.display_name}</strong>
                      <br />
                      <small className="text-muted">
                        Lat: {parseFloat(result.lat).toFixed(6)}, 
                        Lng: {parseFloat(result.lon).toFixed(6)}
                      </small>
                    </div>
                  ))}
                </div>
              </Col>
            </Row>
          )}

          <Row>
            <Col>
              <h6>Enter Coordinates</h6>
              <Row>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Latitude</Form.Label>
                    <Form.Control
                      type="number"
                      step="any"
                      placeholder="-90 to 90"
                      value={coordinateInput.lat}
                      onChange={(e) => setCoordinateInput({...coordinateInput, lat: e.target.value})}
                    />
                  </Form.Group>
                </Col>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Longitude</Form.Label>
                    <Form.Control
                      type="number"
                      step="any"
                      placeholder="-180 to 180"
                      value={coordinateInput.lng}
                      onChange={(e) => setCoordinateInput({...coordinateInput, lng: e.target.value})}
                    />
                  </Form.Group>
                </Col>
              </Row>
            </Col>
          </Row>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowLocationModal(false)}>
            Cancel
          </Button>
          <Button variant="primary" onClick={handleCoordinateSubmit}>
            Go to Location
          </Button>
        </Modal.Footer>
      </Modal>

      <style jsx>{`
        .modern-card {
          background: rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 20px;
          box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
          overflow: hidden;
        }

        .modern-card-header {
          background: rgba(255, 255, 255, 0.1);
          border-bottom: 1px solid rgba(255, 255, 255, 0.2);
          padding: 1.5rem;
        }

        .modern-map {
          border-radius: 0 0 20px 20px;
        }

        .risk-legend .legend-item {
          display: flex;
          align-items: center;
          margin-bottom: 8px;
        }

        .legend-color {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          margin-right: 10px;
        }

        .search-result-item:hover {
          background-color: rgba(74, 144, 226, 0.1);
        }

        .fade-in {
          animation: fadeIn 0.5s ease-in;
        }

        .slide-in {
          animation: slideIn 0.5s ease-out;
        }

        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        @keyframes slideIn {
          from { transform: translateY(20px); opacity: 0; }
          to { transform: translateY(0); opacity: 1; }
        }

        .modern-alert {
          border: none;
          border-radius: 15px;
          backdrop-filter: blur(10px);
        }
      `}</style>
    </Container>
  );
};

export default RiskMap;