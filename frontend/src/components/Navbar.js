import React from 'react';
import { Navbar as BootstrapNavbar, Nav, Container, NavDropdown } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import { FaHome, FaMap, FaBell, FaChartLine, FaCog, FaCloudUploadAlt, FaCube, FaBrain, FaImage, FaFileAlt, FaUser, FaSignOutAlt } from 'react-icons/fa';

const Navbar = ({ user, onLogout }) => {
  const handleLogout = () => {
    if (window.confirm('Are you sure you want to logout?')) {
      onLogout();
    }
  };

  return (
    <BootstrapNavbar bg="dark" variant="dark" expand="lg" sticky="top">
      <Container>
        <BootstrapNavbar.Brand href="/">
          <FaHome className="me-2" />
          Rockfall Prediction System
        </BootstrapNavbar.Brand>
        
        <BootstrapNavbar.Toggle aria-controls="basic-navbar-nav" />
        <BootstrapNavbar.Collapse id="basic-navbar-nav">
          <Nav className="me-auto">
            <Nav.Link as={Link} to="/">
              <FaHome className="me-1" />
              Dashboard
            </Nav.Link>
            
            <Nav.Link as={Link} to="/risk-map">
              <FaMap className="me-1" />
              Risk Map
            </Nav.Link>
            
            <Nav.Link as={Link} to="/alerts">
              <FaBell className="me-1" />
              Alerts
            </Nav.Link>
            
            <Nav.Link as={Link} to="/forecast">
              <FaChartLine className="me-1" />
              Forecast
            </Nav.Link>
            
            <Nav.Link as={Link} to="/sensors">
              <FaCog className="me-1" />
              Sensors
            </Nav.Link>

            <Nav.Link as={Link} to="/reports">
              <FaFileAlt className="me-1" />
              Reports
            </Nav.Link>
            
            <NavDropdown title="🔬 AI Analysis" id="ai-dropdown">
              <NavDropdown.Item as={Link} to="/soil-classifier">
                <FaImage className="me-1" />
                Soil/Rock Classifier
              </NavDropdown.Item>
              <NavDropdown.Item as={Link} to="/lstm">
                <FaBrain className="me-1" />
                LSTM Predictor
              </NavDropdown.Item>
            </NavDropdown>
            
            <NavDropdown title="🗺️ Visualizations" id="viz-dropdown">
              <NavDropdown.Item as={Link} to="/mine-3d">
                <FaCube className="me-1" />
                Mine 3D View
              </NavDropdown.Item>
              <NavDropdown.Item as={Link} to="/lidar">
                <FaCube className="me-1" />
                LIDAR 3D
              </NavDropdown.Item>
            </NavDropdown>
            
            <Nav.Link as={Link} to="/data-sources">
              <FaCloudUploadAlt className="me-1" />
              Data Sources
            </Nav.Link>
          </Nav>
          
          <Nav>
            <Nav.Link className="text-success me-3">
              ● System Online
            </Nav.Link>
            
            <NavDropdown 
              title={
                <span>
                  <FaUser className="me-1" />
                  {user?.first_name} {user?.last_name}
                </span>
              } 
              id="user-dropdown"
              align="end"
            >
              <NavDropdown.Header>
                <div className="text-muted small">{user?.email}</div>
                <div className="text-muted small">{user?.company}</div>
                <div className="text-muted small">Role: {user?.role}</div>
              </NavDropdown.Header>
              <NavDropdown.Divider />
              <NavDropdown.Item onClick={handleLogout}>
                <FaSignOutAlt className="me-1" />
                Logout
              </NavDropdown.Item>
            </NavDropdown>
          </Nav>
        </BootstrapNavbar.Collapse>
      </Container>
    </BootstrapNavbar>
  );
};

export default Navbar;