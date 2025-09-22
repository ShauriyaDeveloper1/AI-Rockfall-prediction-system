import React, { useState, useRef } from 'react';
import './SoilRockClassifier.css';

const SoilRockClassifier = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [classification, setClassification] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const handleImageSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage({
          file: file,
          url: e.target.result,
          name: file.name
        });
        setClassification(null);
        setError('');
      };
      reader.readAsDataURL(file);
    } else {
      setError('Please select a valid image file (JPG, PNG, GIF, etc.)');
    }
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleImageSelect(file);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageSelect(e.dataTransfer.files[0]);
    }
  };

  const classifyImage = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError('');
    
    const formData = new FormData();
    formData.append('image', selectedImage.file);

    try {
      console.log('Sending classification request...');
      const response = await fetch('http://localhost:5001/classify', {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status);

      if (response.ok) {
        const result = await response.json();
        console.log('Classification result:', result);
        
        if (result.success) {
          setClassification(result);
          setError('');
        } else {
          setError(result.error || 'Classification failed');
          setClassification(null);
        }
      } else {
        const errorText = await response.text();
        console.error('Classification failed:', errorText);
        setError(`Classification failed: ${response.status} ${response.statusText}`);
        setClassification(null);
      }
    } catch (error) {
      console.error('Network error:', error);
      setError(`Network error: ${error.message}. Please ensure the classification server is running on port 5001.`);
      setClassification(null);
    } finally {
      setLoading(false);
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

  const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return '#28a745';
    if (confidence >= 75) return '#17a2b8';
    if (confidence >= 60) return '#ffc107';
    return '#dc3545';
  };

  return (
    <div className="soil-rock-classifier">
      <div className="classifier-header">
        <h2>🪨 AI-Powered Soil & Rock Classification System</h2>
        <p>Upload an image to identify rock/soil type with comprehensive geological analysis and Wikipedia integration</p>
      </div>

      <div className="classifier-content">
        {/* Upload Section */}
        <div className="upload-section">
          <div
            className={`upload-area ${dragActive ? 'drag-active' : ''} ${selectedImage ? 'has-image' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileInputChange}
              style={{ display: 'none' }}
            />
            
            {selectedImage ? (
              <div className="image-preview">
                <img src={selectedImage.url} alt="Selected" className="preview-image" />
                <div className="image-info">
                  <p><strong>File:</strong> {selectedImage.name}</p>
                  <button 
                    className="btn btn-primary classify-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      classifyImage();
                    }}
                    disabled={loading}
                  >
                    {loading ? (
                      <>
                        <span className="spinner-border spinner-border-sm me-2"></span>
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <i className="fas fa-microscope me-2"></i>
                        Classify Rock/Soil
                      </>
                    )}
                  </button>
                </div>
              </div>
            ) : (
              <div className="upload-placeholder">
                <i className="fas fa-cloud-upload-alt"></i>
                <h4>Upload Rock/Soil Image</h4>
                <p>Drag & drop an image here or click to browse</p>
                <small>Supports: JPG, PNG, GIF (Max 10MB)</small>
              </div>
            )}
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="alert alert-danger mt-3">
            <i className="fas fa-exclamation-triangle me-2"></i>
            {error}
          </div>
        )}

        {/* Classification Results */}
        {classification && (
          <div className="classification-results">
            <div className="results-header">
              <h3>
                <i className="fas fa-chart-line me-2"></i>
                Classification Results
              </h3>
              <div 
                className="confidence-badge"
                style={{ backgroundColor: getConfidenceColor(classification.confidence) }}
              >
                Confidence: {classification.confidence}%
              </div>
            </div>

            {/* Main Result */}
            <div className="main-result">
              <div className="result-card">
                <h4>Identified Type:</h4>
                <div className="rock-type-result">
                  <span className="rock-name">{classification.predicted_class}</span>
                  <span className={`confidence-level ${classification.confidence_level?.toLowerCase()}`}>
                    {classification.confidence_level} Confidence
                  </span>
                </div>
              </div>
            </div>

            {/* Wikipedia Information */}
            {classification.wikipedia_info && (
              <div className="wikipedia-section">
                <h4><i className="fab fa-wikipedia-w me-2"></i>Wikipedia Information</h4>
                <div className="wikipedia-content">
                  <div className="wiki-text">
                    <h5>{classification.wikipedia_info.title}</h5>
                    <p>{classification.wikipedia_info.description}</p>
                    {classification.wikipedia_info.url && (
                      <a 
                        href={classification.wikipedia_info.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="btn btn-outline-primary btn-sm"
                      >
                        <i className="fas fa-external-link-alt me-1"></i>
                        Read More on Wikipedia
                      </a>
                    )}
                  </div>
                  {classification.wikipedia_info.thumbnail && (
                    <div className="wiki-thumbnail">
                      <img 
                        src={classification.wikipedia_info.thumbnail} 
                        alt={classification.wikipedia_info.title}
                        className="thumbnail-img"
                      />
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Detailed Analysis */}
            {classification.detailed_analysis && (
              <div className="detailed-analysis">
                <h4><i className="fas fa-microscope me-2"></i>Geological Analysis</h4>
                
                {/* Geological Properties */}
                <div className="analysis-grid">
                  <div className="analysis-card">
                    <h5>🔬 Geological Properties</h5>
                    <div className="property-list">
                      <div className="property-item">
                        <span className="label">Formation Type:</span>
                        <span className="value">{classification.detailed_analysis.geological_analysis?.formation_type}</span>
                      </div>
                      <div className="property-item">
                        <span className="label">Hardness:</span>
                        <span className="value">{classification.detailed_analysis.geological_analysis?.hardness_scale}</span>
                      </div>
                      <div className="property-item">
                        <span className="label">Porosity:</span>
                        <span className="value">{classification.detailed_analysis.geological_analysis?.porosity_level}</span>
                      </div>
                      <div className="property-item">
                        <span className="label">Mining Suitability:</span>
                        <span className="value">{classification.detailed_analysis.geological_analysis?.mining_suitability}</span>
                      </div>
                    </div>
                  </div>

                  {/* Risk Assessment */}
                  <div className="analysis-card">
                    <h5>⚠️ Risk Assessment</h5>
                    <div className="risk-analysis">
                      <div className="risk-level">
                        <span className="label">Stability Level:</span>
                        <span 
                          className="risk-badge"
                          style={{ backgroundColor: getRiskColor(classification.detailed_analysis.risk_assessment?.stability_level) }}
                        >
                          {classification.detailed_analysis.risk_assessment?.stability_level}
                        </span>
                      </div>
                      <div className="safety-rating">
                        <span className="label">Safety Rating:</span>
                        <span className="value">{classification.detailed_analysis.risk_assessment?.safety_rating}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Physical Properties */}
                {classification.detailed_analysis.physical_properties && (
                  <div className="properties-section">
                    <h5>🧪 Physical Properties</h5>
                    <div className="properties-grid">
                      {classification.detailed_analysis.physical_properties.map((property, index) => (
                        <div key={index} className="property-tag">
                          {property}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Commercial Uses */}
                {classification.detailed_analysis.commercial_uses && (
                  <div className="uses-section">
                    <h5>🏗️ Commercial Applications</h5>
                    <div className="uses-grid">
                      {classification.detailed_analysis.commercial_uses.map((use, index) => (
                        <div key={index} className="use-tag">
                          {use}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Mining Analysis */}
                {classification.detailed_analysis.mining_analysis && (
                  <div className="mining-section">
                    <h5>⛏️ Mining Analysis</h5>
                    <div className="mining-grid">
                      <div className="mining-item">
                        <span className="label">Extraction Difficulty:</span>
                        <span className="value">{classification.detailed_analysis.mining_analysis.extraction_difficulty}</span>
                      </div>
                      <div className="mining-item">
                        <span className="label">Economic Value:</span>
                        <span className="value">{classification.detailed_analysis.mining_analysis.economic_value}</span>
                      </div>
                      <div className="mining-item">
                        <span className="label">Environmental Impact:</span>
                        <span className="value">{classification.detailed_analysis.mining_analysis.environmental_impact}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Analysis Metadata */}
            {classification.analysis_metadata && (
              <div className="metadata-section">
                <h5><i className="fas fa-info-circle me-2"></i>Analysis Details</h5>
                <div className="metadata-grid">
                  <div className="metadata-item">
                    <span className="label">Processing Time:</span>
                    <span className="value">{classification.analysis_metadata.processing_time}</span>
                  </div>
                  <div className="metadata-item">
                    <span className="label">Model Version:</span>
                    <span className="value">{classification.analysis_metadata.model_version}</span>
                  </div>
                  <div className="metadata-item">
                    <span className="label">Analysis Date:</span>
                    <span className="value">{new Date(classification.analysis_metadata.timestamp).toLocaleString()}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default SoilRockClassifier;