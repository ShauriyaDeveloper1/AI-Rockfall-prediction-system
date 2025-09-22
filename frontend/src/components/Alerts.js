import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Table, Badge, Button, Alert as BootstrapAlert } from 'react-bootstrap';
import { FaBell, FaSync, FaFilter } from 'react-icons/fa';
import axios from 'axios';
import moment from 'moment';

const Alerts = () => {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('ALL');

  useEffect(() => {
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchAlerts = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/alerts');
      // Backend returns array directly, not wrapped in { alerts: [...] }
      const alertsData = Array.isArray(response.data) ? response.data : [];
      setAlerts(alertsData);
      setError(null);
    } catch (err) {
      setError('Failed to fetch alerts');
      console.error('Alerts error:', err);
      // Set empty array as fallback
      setAlerts([]);
    } finally {
      setLoading(false);
    }
  };

  const getSeverityVariant = (riskLevel) => {
    switch (riskLevel) {
      case 'LOW': return 'success';
      case 'MEDIUM': return 'warning';
      case 'HIGH': return 'danger';
      case 'CRITICAL': return 'dark';
      default: return 'secondary';
    }
  };

  const getStatusVariant = (status) => {
    switch (status) {
      case 'ACTIVE': return 'danger';
      case 'ACKNOWLEDGED': return 'warning';
      case 'RESOLVED': return 'success';
      default: return 'secondary';
    }
  };

  const filteredAlerts = (alerts || []).filter(alert => {
    if (filter === 'ALL') return true;
    if (filter === 'ACTIVE') return alert.status === 'ACTIVE';
    if (filter === 'HIGH_RISK') return alert.risk_level === 'HIGH' || alert.risk_level === 'CRITICAL';
    return true;
  });

  if (loading) {
    return (
      <Container className="mt-4">
        <div className="text-center loading-spinner">
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
        </div>
      </Container>
    );
  }

  return (
    <Container fluid className="mt-4 fade-in">
      <Row className="mb-4">
        <Col>
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <h1 className="display-5 text-primary">
                <FaBell className="me-3" />
                Alert Management
              </h1>
              <p className="lead">Monitor and manage system alerts</p>
            </div>
            <div>
              <Button 
                variant={filter === 'ALL' ? 'primary' : 'outline-primary'}
                onClick={() => setFilter('ALL')} 
                className="modern-btn"
              >
                All
              </Button>
              <Button 
                variant={filter === 'ACTIVE' ? 'warning' : 'outline-warning'}
                onClick={() => setFilter('ACTIVE')} 
                className="ms-2 modern-btn"
              >
                Active
              </Button>
              <Button 
                variant={filter === 'HIGH_RISK' ? 'danger' : 'outline-danger'}
                onClick={() => setFilter('HIGH_RISK')} 
                className="ms-2 modern-btn"
              >
                High Risk
              </Button>
              <Button variant="success" onClick={fetchAlerts} className="ms-3 modern-btn">
                <FaSync className="me-1" />
                Refresh
              </Button>
            </div>
          </div>
        </Col>
      </Row>

      {error && (
        <Row className="mb-4">
          <Col>
            <BootstrapAlert variant="danger" className="modern-alert">{error}</BootstrapAlert>
          </Col>
        </Row>
      )}

      <Row className="mb-4">
        <Col md={3}>
          <Card className="modern-card slide-in">
            <Card.Body className="text-center">
              <h4 className="text-primary display-6">{(alerts || []).length}</h4>
              <small className="text-light">Total Alerts</small>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="modern-card slide-in">
            <Card.Body className="text-center">
              <h4 className="text-danger display-6">{(alerts || []).filter(a => a.status === 'ACTIVE').length}</h4>
              <small className="text-light">Active Alerts</small>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="modern-card slide-in">
            <Card.Body className="text-center">
              <h4 className="text-warning display-6">{(alerts || []).filter(a => a.risk_level === 'HIGH' || a.risk_level === 'CRITICAL').length}</h4>
              <small className="text-light">High Risk</small>
            </Card.Body>
          </Card>
        </Col>
        <Col md={3}>
          <Card className="modern-card slide-in">
            <Card.Body className="text-center">
              <h4 className="text-success display-6">{(alerts || []).filter(a => a.status === 'RESOLVED').length}</h4>
              <small className="text-light">Resolved</small>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col>
          <Card className="modern-card slide-in">
            <Card.Header className="modern-card-header">
              <h5 className="mb-0 text-primary">
                <FaFilter className="me-2" />
                Alerts ({filteredAlerts.length})
              </h5>
            </Card.Header>
            <Card.Body className="p-0">
              {filteredAlerts.length > 0 ? (
                <Table responsive striped hover className="mb-0 modern-table">
                  <thead className="modern-table-header">
                    <tr>
                      <th className="text-primary">Timestamp</th>
                      <th className="text-primary">Type</th>
                      <th className="text-primary">Risk Level</th>
                      <th className="text-primary">Status</th>
                      <th className="text-primary">Message</th>
                      <th className="text-primary">Location</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredAlerts.map((alert) => (
                      <tr key={alert.id} className="modern-table-row">
                        <td>
                          <small className="text-light">{moment(alert.timestamp).format('YYYY-MM-DD HH:mm:ss')}</small>
                          <br />
                          <small className="text-muted">{moment(alert.timestamp).fromNow()}</small>
                        </td>
                        <td>
                          <Badge bg="info" className="modern-badge">{alert.alert_type}</Badge>
                        </td>
                        <td>
                          <Badge bg={getSeverityVariant(alert.risk_level)} className="modern-badge">
                            {alert.risk_level}
                          </Badge>
                        </td>
                        <td>
                          <Badge bg={getStatusVariant(alert.status)} className="modern-badge">
                            {alert.status}
                          </Badge>
                        </td>
                        <td className="text-light">{alert.message}</td>
                        <td>
                          {alert.location ? (
                            <small className="text-success">
                              Coordinates available
                            </small>
                          ) : (
                            <small className="text-muted">N/A</small>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              ) : (
                <div className="text-center p-4 modern-empty-state">
                  <FaBell size={48} className="text-muted mb-3 pulse" />
                  <p className="text-muted">No alerts match the current filter</p>
                </div>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default Alerts;