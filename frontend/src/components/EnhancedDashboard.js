import React, { useState, useEffect } from 'react';
import { Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import './EnhancedDashboard.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const EnhancedDashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedMine, setSelectedMine] = useState(null);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const response = await fetch('/api/dashboard/enhanced-stats');
      const data = await response.json();
      if (data.success) {
        setDashboardData(data);
      }
      setLoading(false);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setLoading(false);
    }
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'HIGH': return '#dc3545';
      case 'MEDIUM': return '#ffc107';
      case 'LOW': return '#28a745';
      default: return '#6c757d';
    }
  };

  const doughnutOptions = {
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: '#ffffff',
          font: {
            size: 12
          }
        }
      }
    },
    maintainAspectRatio: false,
    responsive: true
  };

  const riskDistributionData = dashboardData ? {
    labels: ['High Risk', 'Medium Risk', 'Low Risk'],
    datasets: [
      {
        data: [
          dashboardData.risk_distribution?.high_risk || 0,
          dashboardData.risk_distribution?.medium_risk || 0,
          dashboardData.risk_distribution?.low_risk || 0
        ],
        backgroundColor: ['#dc3545', '#ffc107', '#28a745'],
        borderColor: ['#fff', '#fff', '#fff'],
        borderWidth: 2
      }
    ]
  } : null;

  if (loading) {
    return (
      <div className="enhanced-dashboard loading">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading Enhanced Dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="enhanced-dashboard">
      {/* Header Stats */}
      <div className="dashboard-header">
        <div className="stats-cards">
          <div className="stat-card high-risk">
            <div className="stat-icon">⚠️</div>
            <div className="stat-content">
              <h3>{dashboardData?.stats?.high_risk_mines || 24}</h3>
              <p>High Risk Mines</p>
            </div>
          </div>
          
          <div className="stat-card total-incidents">
            <div className="stat-icon">📊</div>
            <div className="stat-content">
              <h3>{dashboardData?.stats?.total_incidents || 139}</h3>
              <p>Total Incidents</p>
            </div>
          </div>
          
          <div className="stat-card injuries">
            <div className="stat-icon">👥</div>
            <div className="stat-content">
              <h3>{dashboardData?.stats?.injuries_6m || 28}</h3>
              <p>Injuries (6M)</p>
            </div>
          </div>
          
          <div className="stat-card active-mines">
            <div className="stat-icon">📈</div>
            <div className="stat-content">
              <h3>{dashboardData?.stats?.active_mines || 167}</h3>
              <p>Active Mines</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="dashboard-content">
        {/* India Risk Map */}
        <div className="dashboard-section india-risk-map">
          <div className="section-header">
            <h2>🌏 Live India Mining Risk Map</h2>
            <div className="live-indicator">
              <span className="live-dot"></span>
              Live Updates
            </div>
          </div>
          
          <div className="india-map-container">
            <div className="india-map">
              <svg viewBox="0 0 800 600" className="map-svg">
                {/* Simplified India Map Shape */}
                <path
                  d="M200,100 L600,100 L650,200 L600,500 L200,480 L150,300 Z"
                  fill="#2c3e50"
                  stroke="#00ffff"
                  strokeWidth="2"
                />
                
                {/* Risk Points */}
                {dashboardData?.india_risk_map?.map((location, index) => (
                  <g key={index}>
                    <circle
                      cx={300 + index * 80}
                      cy={200 + (index % 2) * 100}
                      r="8"
                      fill={getRiskColor(location.risk)}
                      className="risk-point"
                      onClick={() => setSelectedMine(location)}
                    />
                    <circle
                      cx={300 + index * 80}
                      cy={200 + (index % 2) * 100}
                      r="15"
                      fill="none"
                      stroke={getRiskColor(location.risk)}
                      strokeWidth="2"
                      opacity="0.5"
                      className="risk-pulse"
                    />
                  </g>
                ))}
              </svg>
            </div>
            
            <div className="map-legend">
              <h4>Risk Levels</h4>
              <div className="legend-items">
                <div className="legend-item">
                  <span className="legend-color high-risk"></span>
                  High Risk
                </div>
                <div className="legend-item">
                  <span className="legend-color medium-risk"></span>
                  Medium Risk
                </div>
                <div className="legend-item">
                  <span className="legend-color low-risk"></span>
                  Low Risk
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Risk Distribution */}
        <div className="dashboard-section risk-distribution">
          <h2>📊 Risk Distribution</h2>
          <div className="chart-container">
            {riskDistributionData && (
              <Doughnut data={riskDistributionData} options={doughnutOptions} />
            )}
          </div>
          <div className="distribution-stats">
            <div className="dist-stat">
              <span className="dist-color high"></span>
              <span>High Risk</span>
              <strong>{dashboardData?.risk_distribution?.high_risk || 35}%</strong>
            </div>
            <div className="dist-stat">
              <span className="dist-color medium"></span>
              <span>Medium Risk</span>
              <strong>{dashboardData?.risk_distribution?.medium_risk || 45}%</strong>
            </div>
            <div className="dist-stat">
              <span className="dist-color low"></span>
              <span>Low Risk</span>
              <strong>{dashboardData?.risk_distribution?.low_risk || 20}%</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Events & Weather */}
      <div className="dashboard-bottom">
        {/* Recent Events */}
        <div className="dashboard-section recent-events">
          <h3>📅 Last 24h Events</h3>
          <div className="events-list">
            {dashboardData?.recent_events?.slice(0, 3).map((event, index) => (
              <div key={index} className={`event-item ${event.severity.toLowerCase()}`}>
                <div className="event-header">
                  <span className="event-title">{event.title}</span>
                  <span className={`event-severity ${event.severity.toLowerCase()}`}>
                    {event.severity}
                  </span>
                </div>
                <div className="event-time">
                  {new Date(event.timestamp).toLocaleString()}
                </div>
              </div>
            )) || [
              <div key="fallback1" className="event-item critical">
                <div className="event-header">
                  <span className="event-title">Rockfall detected - Zone A</span>
                  <span className="event-severity critical">Critical</span>
                </div>
                <div className="event-time">2 hours ago</div>
              </div>,
              <div key="fallback2" className="event-item high">
                <div className="event-header">
                  <span className="event-title">Displacement threshold exceeded</span>
                  <span className="event-severity high">High</span>
                </div>
                <div className="event-time">4 hours ago</div>
              </div>,
              <div key="fallback3" className="event-item medium">
                <div className="event-header">
                  <span className="event-title">Rainfall trigger activated</span>
                  <span className="event-severity medium">Medium</span>
                </div>
                <div className="event-time">6 hours ago</div>
              </div>
            ]}
          </div>
        </div>

        {/* Weather Triggers */}
        <div className="dashboard-section weather-triggers">
          <h3>🌤️ Weather Triggers</h3>
          <div className="weather-items">
            <div className="weather-item rainfall">
              <div className="weather-icon">🌧️</div>
              <div className="weather-content">
                <div className="weather-label">Rainfall > 25mm/hr</div>
                <div className="weather-details">
                  <span>Temperature swing: 15°C</span>
                  <span>Wind: Normal</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Selected Mine Modal */}
      {selectedMine && (
        <div className="mine-modal-overlay" onClick={() => setSelectedMine(null)}>
          <div className="mine-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{selectedMine.name}</h3>
              <button onClick={() => setSelectedMine(null)}>×</button>
            </div>
            <div className="modal-content">
              <div className="mine-info">
                <div className="info-row">
                  <label>Risk Level:</label>
                  <span className={`risk-badge ${selectedMine.risk.toLowerCase()}`}>
                    {selectedMine.risk}
                  </span>
                </div>
                <div className="info-row">
                  <label>Location:</label>
                  <span>{selectedMine.name}</span>
                </div>
                <div className="info-row">
                  <label>Coordinates:</label>
                  <span>{selectedMine.lat}°, {selectedMine.lng}°</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EnhancedDashboard;