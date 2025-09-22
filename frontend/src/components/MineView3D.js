import React, { useState, useEffect } from 'react';
import './MineView3D.css';

const MineView3D = () => {
  const [siteData, setSiteData] = useState(null);
  const [selectedSensor, setSelectedSensor] = useState(null);
  const [showEscape, setShowEscape] = useState(false);
  const [loading, setLoading] = useState(true);
  const [layerControls, setLayerControls] = useState({
    radar: false,
    lidar: true,
    gnss: true,
    cameras: false,
    aiRiskHeatmap: true
  });

  useEffect(() => {
    fetchMineData();
  }, []);

  const fetchMineData = async () => {
    try {
      const response = await fetch('/api/mine-site/details?site_id=jharia-section-a');
      const data = await response.json();
      if (data.success) {
        setSiteData(data.site_data);
      }
      setLoading(false);
    } catch (error) {
      console.error('Error fetching mine data:', error);
      setLoading(false);
    }
  };

  const handleLayerToggle = (layer) => {
    setLayerControls(prev => ({
      ...prev,
      [layer]: !prev[layer]
    }));
  };

  const getSensorStatusColor = (status) => {
    switch (status) {
      case 'active': return '#28a745';
      case 'warning': return '#ffc107';
      case 'critical': return '#dc3545';
      default: return '#6c757d';
    }
  };

  const getRiskZoneColor = (riskLevel) => {
    switch (riskLevel) {
      case 'LOW': return 'rgba(40, 167, 69, 0.6)';
      case 'MEDIUM': return 'rgba(255, 193, 7, 0.6)';
      case 'HIGH': return 'rgba(253, 126, 20, 0.6)';
      case 'CRITICAL': return 'rgba(220, 53, 69, 0.6)';
      default: return 'rgba(108, 117, 125, 0.6)';
    }
  };

  if (loading) {
    return (
      <div className="mine-view-loading">
        <div className="hologram-loader">
          <div className="loader-ring"></div>
          <div className="loader-ring"></div>
          <div className="loader-ring"></div>
        </div>
        <p>Initializing 3D Mine Hologram...</p>
      </div>
    );
  }

  return (
    <div className="mine-view-3d">
      <div className="mine-view-header">
        <div className="header-left">
          <h2>🔷 3D Mine Hologram - Advanced Mapping System</h2>
          <p>Real-time holographic visualization with sensor network, escape routes, and equipment tracking</p>
        </div>
        <div className="header-right">
          <div className="emergency-escape">
            <button 
              className={`escape-btn ${showEscape ? 'active' : ''}`}
              onClick={() => setShowEscape(!showEscape)}
            >
              🚨 Emergency Escape System
            </button>
          </div>
        </div>
      </div>

      <div className="mine-view-content">
        <div className="mine-visualization">
          <div className="hologram-container">
            <div className="hologram-mine">
              {/* Mine Structure */}
              <div className="mine-pit">
                <div className="mine-levels">
                  {[...Array(6)].map((_, i) => (
                    <div key={i} className={`mine-level level-${i}`} />
                  ))}
                </div>
                
                {/* Processing Plant */}
                <div className="processing-plant">
                  <div className="plant-building"></div>
                  <div className="plant-label">PROCESSING PLANT</div>
                </div>

                {/* Risk Zones */}
                {siteData?.risk_zones?.map((zone, index) => (
                  <div 
                    key={index}
                    className="risk-zone"
                    style={{ 
                      backgroundColor: getRiskZoneColor(zone.risk_level),
                      left: `${20 + index * 25}%`,
                      top: `${30 + index * 15}%`
                    }}
                  >
                    <div className="zone-label">
                      <strong>{zone.name}</strong>
                      <div className={`risk-indicator ${zone.risk_level.toLowerCase()}`}>
                        {zone.risk_level} RISK
                      </div>
                      <small>{zone.description}</small>
                    </div>
                  </div>
                ))}

                {/* Sensor Network */}
                {siteData?.sensors?.map((sensor, index) => (
                  <div 
                    key={sensor.id}
                    className={`sensor-node ${sensor.status} ${selectedSensor?.id === sensor.id ? 'selected' : ''}`}
                    style={{
                      left: `${25 + index * 20}%`,
                      top: `${40 + (index % 2) * 20}%`,
                      borderColor: getSensorStatusColor(sensor.status)
                    }}
                    onClick={() => setSelectedSensor(sensor)}
                  >
                    <div className="sensor-pulse"></div>
                    <div className="sensor-icon">
                      {sensor.type === 'Tiltmeter' && '📐'}
                      {sensor.type === 'Piezometer' && '💧'}
                      {sensor.type === 'Vibration' && '〰️'}
                      {sensor.type === 'Crackmeter' && '📏'}
                    </div>
                    <div className="sensor-tooltip">
                      <strong>{sensor.id}</strong>
                      <div>{sensor.type}</div>
                      <div>{sensor.value} {sensor.unit}</div>
                    </div>
                  </div>
                ))}

                {/* GNSS Perimeter */}
                {layerControls.gnss && (
                  <div className="gnss-perimeter">
                    <div className="gnss-point gnss-west">GNSS<br/>Perimeter-West</div>
                  </div>
                )}

                {/* Escape Routes */}
                {showEscape && (
                  <div className="escape-routes">
                    <div className="escape-route route-1"></div>
                    <div className="escape-route route-2"></div>
                    <div className="escape-point">
                      <div className="escape-icon">🚪</div>
                      <div className="escape-label">Emergency Exit</div>
                    </div>
                  </div>
                )}
              </div>

              {/* Holographic Grid */}
              <div className="holographic-grid">
                {[...Array(10)].map((_, i) => (
                  <div key={i} className="grid-line horizontal" style={{ top: `${i * 10}%` }} />
                ))}
                {[...Array(10)].map((_, i) => (
                  <div key={i} className="grid-line vertical" style={{ left: `${i * 10}%` }} />
                ))}
              </div>
            </div>
          </div>

          {/* Layer Controls */}
          <div className="layer-controls">
            <h4>Layers:</h4>
            <div className="layer-toggles">
              {Object.entries(layerControls).map(([layer, enabled]) => (
                <label key={layer} className="layer-toggle">
                  <input
                    type="checkbox"
                    checked={enabled}
                    onChange={() => handleLayerToggle(layer)}
                  />
                  <span className="toggle-label">
                    {layer === 'aiRiskHeatmap' ? 'AI Risk Heatmap' : 
                     layer.charAt(0).toUpperCase() + layer.slice(1)}
                  </span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="mine-sidebar">
          {/* Current Site Risk */}
          <div className="current-site-risk">
            <div className="risk-badge high">Current Site Risk: HIGH</div>
            <div className="active-alarms">🚨 Active Alarms: 3</div>
          </div>

          {/* Sensor Network Status */}
          <div className="sensor-network">
            <h3>📡 Sensor Network</h3>
            <div className="sensor-list">
              {siteData?.sensors?.map((sensor) => (
                <div 
                  key={sensor.id} 
                  className={`sensor-item ${sensor.status} ${selectedSensor?.id === sensor.id ? 'selected' : ''}`}
                  onClick={() => setSelectedSensor(sensor)}
                >
                  <div className="sensor-header">
                    <span className="sensor-name">{sensor.type}</span>
                    <span className={`sensor-status ${sensor.status}`}>{sensor.status}</span>
                  </div>
                  <div className="sensor-details">
                    <div className="sensor-location">{sensor.sector}</div>
                    <div className="sensor-description">{sensor.description}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Emergency Escape Actions */}
          {showEscape && (
            <div className="emergency-actions">
              <h3>🚨 Emergency Actions</h3>
              <div className="escape-controls">
                <button className="escape-action show-routes">Show Routes</button>
                <button className="escape-action navigate">🧭 Navigate</button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Selected Sensor Details */}
      {selectedSensor && (
        <div className="sensor-details-modal">
          <div className="modal-content">
            <div className="modal-header">
              <h4>{selectedSensor.type} - {selectedSensor.id}</h4>
              <button onClick={() => setSelectedSensor(null)}>×</button>
            </div>
            <div className="modal-body">
              <div className="sensor-reading">
                <label>Current Value:</label>
                <span className={`value ${selectedSensor.status}`}>
                  {selectedSensor.value} {selectedSensor.unit}
                </span>
              </div>
              <div className="sensor-threshold">
                <label>Threshold:</label>
                <span>{selectedSensor.threshold} {selectedSensor.unit}</span>
              </div>
              <div className="sensor-location">
                <label>Location:</label>
                <span>{selectedSensor.sector}</span>
              </div>
              <div className="sensor-description">
                <label>Description:</label>
                <span>{selectedSensor.description}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MineView3D;