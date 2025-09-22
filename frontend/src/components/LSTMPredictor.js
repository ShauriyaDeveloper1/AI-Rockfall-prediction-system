import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Button,
  Grid,
  TextField,
  Alert,
  CircularProgress,
  Chip,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Stepper,
  Step,
  StepLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Tooltip,
  LinearProgress,
  Fab,
  Snackbar,
} from '@mui/material';
import {
  TrendingUp,
  CloudUpload,
  Analytics,
  Psychology,
  Timeline,
  Refresh,
  Info,
  Assessment,
  PlayArrow,
  Stop,
} from '@mui/icons-material';
import axios from 'axios';

const LSTMPredictor = () => {
  // State management
  const [modelStatus, setModelStatus] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [trainingFile, setTrainingFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [realTimePrediction, setRealTimePrediction] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [sequenceData, setSequenceData] = useState('');
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [activeStep, setActiveStep] = useState(0);
  const [openStatusDialog, setOpenStatusDialog] = useState(false);
  const [autoPredict, setAutoPredict] = useState(false);

  // Configuration
  const steps = ['Upload Data', 'Analytics Dashboard', 'Risk Prediction'];
  const riskColors = {
    LOW: '#4caf50',
    MEDIUM: '#ff9800',
    HIGH: '#f44336',
    CRITICAL: '#9c27b0'
  };

  // Function definitions using useCallback to prevent re-renders
  const loadModelStatus = useCallback(async () => {
    try {
      const response = await axios.get('/api/lstm/status');
      setModelStatus(response.data);
      
      // Automatically advance steps based on model status
      if (response.data.model_trained) {
        setActiveStep(2); // Move to Risk Prediction step
      } else if (trainingFile) {
        setActiveStep(1); // Stay on Analytics Dashboard step
      } else {
        setActiveStep(0); // Stay on Upload Data step
      }
    } catch (err) {
      setError('Failed to load model status');
    }
  }, [trainingFile]);

  const makeRealTimePrediction = useCallback(async () => {
    if (!modelStatus?.model_trained) return;

    try {
      const response = await axios.get('/api/lstm/predict-realtime');
      
      const newPrediction = {
        ...response.data.prediction,
        id: Date.now(),
        type: 'realtime'
      };
      
      setRealTimePrediction(newPrediction);
      setPredictions(prev => [newPrediction, ...prev.slice(0, 9)]);
    } catch (err) {
      console.error('Real-time prediction failed:', err);
    }
  }, [modelStatus]);

  // Load model status on component mount
  useEffect(() => {
    loadModelStatus();
  }, [loadModelStatus]);

  // Auto-prediction interval
  useEffect(() => {
    let interval;
    if (autoPredict && modelStatus?.model_trained) {
      interval = setInterval(() => {
        makeRealTimePrediction();
      }, 30000); // Every 30 seconds
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoPredict, modelStatus, makeRealTimePrediction]);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setTrainingFile(file);
      setActiveStep(1);
      
      // Automatically trigger training in background when file is uploaded
      setIsTraining(true);
      setError(null);
      setSuccess('File uploaded! Training will start automatically in the backend...');

      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await axios.post('/api/lstm/train', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        setTrainingHistory(response.data.training_history);
        setSuccess('Model trained successfully with combined datasets!');
        setActiveStep(2);
        await loadModelStatus();
      } catch (err) {
        setError(`Training failed: ${err.response?.data?.error || err.message}`);
      } finally {
        setIsTraining(false);
      }
    }
  };

  const retrainModel = async () => {
    setIsTraining(true);
    setError(null);

    try {
      const formData = new FormData();
      if (trainingFile) {
        formData.append('file', trainingFile);
      }

      const response = await axios.post('/api/lstm/retrain', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setTrainingHistory(response.data.training_history);
      setSuccess('Model retrained successfully!');
      await loadModelStatus();
    } catch (err) {
      setError(`Retraining failed: ${err.response?.data?.error || err.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const makeCustomPrediction = async () => {
    if (!sequenceData.trim()) {
      setError('Please enter sequence data');
      return;
    }

    setIsPredicting(true);
    setError(null);

    try {
      const sequence = JSON.parse(sequenceData);
      const response = await axios.post('/api/lstm/predict', { sequence });
      
      const newPrediction = {
        ...response.data.prediction,
        id: Date.now(),
        type: 'custom'
      };
      
      setPredictions(prev => [newPrediction, ...prev.slice(0, 9)]);
      setSuccess('Prediction completed!');
      setActiveStep(3);
    } catch (err) {
      setError(`Prediction failed: ${err.response?.data?.error || err.message}`);
    } finally {
      setIsPredicting(false);
    }
  };

  const getRiskChip = (riskLevel, confidence) => (
    <Chip
      label={`${riskLevel} (${(confidence * 100).toFixed(1)}%)`}
      sx={{
        backgroundColor: riskColors[riskLevel] || '#666',
        color: 'white',
        fontWeight: 'bold'
      }}
    />
  );

  const generateSampleSequence = () => {
    // Generate sample 10x6 sequence for demonstration
    const sequence = Array.from({ length: 10 }, () => 
      Array.from({ length: 6 }, () => 
        (Math.random() * 100).toFixed(2)
      )
    );
    setSequenceData(JSON.stringify(sequence, null, 2));
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Card sx={{ mb: 3, background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)' }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Box display="flex" alignItems="center">
              <Psychology sx={{ fontSize: 40, color: 'white', mr: 2 }} />
              <Box>
                <Typography variant="h4" sx={{ color: 'white', fontWeight: 'bold' }}>
                  LSTM Time Series Predictor
                </Typography>
                <Typography variant="subtitle1" sx={{ color: 'rgba(255,255,255,0.8)' }}>
                  Deep Learning for Rockfall Risk Assessment
                </Typography>
              </Box>
            </Box>
            <Box display="flex" gap={1}>
              <Tooltip title="Model Status">
                <IconButton onClick={() => setOpenStatusDialog(true)} sx={{ color: 'white' }}>
                  <Info />
                </IconButton>
              </Tooltip>
              <Tooltip title="Refresh Status">
                <IconButton onClick={loadModelStatus} sx={{ color: 'white' }}>
                  <Refresh />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Progress Stepper */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stepper activeStep={activeStep} alternativeLabel>
            {steps.map((label, index) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>
        </CardContent>
      </Card>

      {/* Alerts */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Data Upload & Analytics Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Data Analytics Dashboard"
              avatar={<Analytics color="primary" />}
              action={
                <Chip
                  label={modelStatus?.model_trained ? 'Model Ready' : 'Training Available'}
                  color={modelStatus?.model_trained ? 'success' : 'warning'}
                />
              }
            />
            <CardContent>
              <Box display="flex" flexDirection="column" gap={2}>
                <Box>
                  <input
                    accept=".csv"
                    style={{ display: 'none' }}
                    id="training-file"
                    type="file"
                    onChange={handleFileUpload}
                  />
                  <label htmlFor="training-file">
                    <Button
                      variant="outlined"
                      component="span"
                      startIcon={<CloudUpload />}
                      fullWidth
                    >
                      {trainingFile ? trainingFile.name : 'Upload CSV Data for Analysis'}
                    </Button>
                  </label>
                </Box>

                <Typography variant="body2" color="textSecondary">
                  Upload CSV data to enhance model training with Hugging Face dataset integration.
                  Model training happens automatically in the backend.
                </Typography>

                {trainingFile && (
                  <Alert severity="info">
                    <Typography variant="body2">
                      File ready for analysis. Combined with Hugging Face "zhaoyiww/Rockfall_Simulator" dataset.
                    </Typography>
                  </Alert>
                )}

                {modelStatus?.model_trained && (
                  <Button
                    variant="outlined"
                    onClick={retrainModel}
                    disabled={isTraining}
                    startIcon={<Refresh />}
                    fullWidth
                  >
                    Retrain with New Data
                  </Button>
                )}

                {isTraining && (
                  <Box>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                      Backend training in progress...
                    </Typography>
                    <LinearProgress />
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Prediction Section */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader
              title="Make Predictions"
              avatar={<Analytics color="primary" />}
              action={
                <Box display="flex" gap={1}>
                  <Button
                    size="small"
                    onClick={generateSampleSequence}
                    variant="outlined"
                  >
                    Sample
                  </Button>
                  <Button
                    size="small"
                    onClick={() => setAutoPredict(!autoPredict)}
                    variant={autoPredict ? "contained" : "outlined"}
                    color={autoPredict ? "success" : "primary"}
                  >
                    {autoPredict ? <Stop /> : <PlayArrow />}
                  </Button>
                </Box>
              }
            />
            <CardContent>
              <Box display="flex" flexDirection="column" gap={2}>
                <TextField
                  label="Sequence Data (JSON)"
                  multiline
                  rows={4}
                  value={sequenceData}
                  onChange={(e) => setSequenceData(e.target.value)}
                  placeholder="Enter 10x6 array: [[val1, val2, ...], ...]"
                  disabled={!modelStatus?.model_trained}
                  fullWidth
                />

                <Button
                  variant="contained"
                  onClick={makeCustomPrediction}
                  disabled={!modelStatus?.model_trained || isPredicting || !sequenceData.trim()}
                  startIcon={isPredicting ? <CircularProgress size={20} /> : <Assessment />}
                  fullWidth
                >
                  {isPredicting ? 'Predicting...' : 'Predict Risk'}
                </Button>

                <Button
                  variant="outlined"
                  onClick={makeRealTimePrediction}
                  disabled={!modelStatus?.model_trained}
                  startIcon={<Timeline />}
                  fullWidth
                >
                  Real-time Prediction
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Real-time Status */}
        {realTimePrediction && (
          <Grid item xs={12}>
            <Card sx={{ backgroundColor: riskColors[realTimePrediction.risk_level] + '10' }}>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="between">
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Latest Real-time Prediction
                    </Typography>
                    <Box display="flex" alignItems="center" gap={2}>
                      {getRiskChip(realTimePrediction.risk_level, realTimePrediction.confidence)}
                      <Typography variant="body2" color="textSecondary">
                        {new Date(realTimePrediction.timestamp).toLocaleString()}
                      </Typography>
                    </Box>
                  </Box>
                  <TrendingUp sx={{ fontSize: 40, color: riskColors[realTimePrediction.risk_level] }} />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Training History Chart */}
        {trainingHistory && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Training Progress" />
              <CardContent>
                <Box display="flex" gap={2} mb={2}>
                  <Chip label={`Accuracy: ${(trainingHistory.final_accuracy * 100).toFixed(1)}%`} color="success" />
                  <Chip label={`Loss: ${trainingHistory.final_loss.toFixed(4)}`} color="info" />
                  <Chip label={`Epochs: ${trainingHistory.epochs_trained}`} color="default" />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Prediction History */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader 
              title="Prediction History" 
              action={
                <Button size="small" onClick={() => setPredictions([])}>
                  Clear
                </Button>
              }
            />
            <CardContent>
              <TableContainer component={Paper} sx={{ maxHeight: 300 }}>
                <Table stickyHeader size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Time</TableCell>
                      <TableCell>Risk Level</TableCell>
                      <TableCell>Confidence</TableCell>
                      <TableCell>Type</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {predictions.map((pred) => (
                      <TableRow key={pred.id}>
                        <TableCell>
                          {new Date(pred.timestamp).toLocaleTimeString()}
                        </TableCell>
                        <TableCell>
                          {getRiskChip(pred.risk_level, pred.confidence)}
                        </TableCell>
                        <TableCell>{(pred.confidence * 100).toFixed(1)}%</TableCell>
                        <TableCell>
                          <Chip
                            label={pred.type}
                            size="small"
                            variant="outlined"
                            color={pred.type === 'realtime' ? 'success' : 'default'}
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Auto-predict FAB */}
      {modelStatus?.model_trained && (
        <Fab
          color={autoPredict ? "secondary" : "primary"}
          aria-label="auto-predict"
          sx={{ position: 'fixed', bottom: 16, right: 16 }}
          onClick={() => setAutoPredict(!autoPredict)}
        >
          {autoPredict ? <Stop /> : <PlayArrow />}
        </Fab>
      )}

      {/* Status Dialog */}
      <Dialog open={openStatusDialog} onClose={() => setOpenStatusDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Model Status</DialogTitle>
        <DialogContent>
          {modelStatus && (
            <TableContainer>
              <Table>
                <TableBody>
                  <TableRow>
                    <TableCell>LSTM Available</TableCell>
                    <TableCell>
                      <Chip 
                        label={modelStatus.lstm_available ? 'Yes' : 'No'} 
                        color={modelStatus.lstm_available ? 'success' : 'error'} 
                      />
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Model Loaded</TableCell>
                    <TableCell>
                      <Chip 
                        label={modelStatus.model_loaded ? 'Yes' : 'No'} 
                        color={modelStatus.model_loaded ? 'success' : 'error'} 
                      />
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Model Trained</TableCell>
                    <TableCell>
                      <Chip 
                        label={modelStatus.model_trained ? 'Yes' : 'No'} 
                        color={modelStatus.model_trained ? 'success' : 'error'} 
                      />
                    </TableCell>
                  </TableRow>
                  {modelStatus.model_config && (
                    <>
                      <TableRow>
                        <TableCell>Sequence Length</TableCell>
                        <TableCell>{modelStatus.model_config.sequence_length}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Features</TableCell>
                        <TableCell>{modelStatus.model_config.num_features}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>LSTM Units</TableCell>
                        <TableCell>{modelStatus.model_config.lstm_units}</TableCell>
                      </TableRow>
                    </>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenStatusDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Success Snackbar */}
      <Snackbar
        open={!!success}
        autoHideDuration={6000}
        onClose={() => setSuccess(null)}
        message={success}
      />
    </Box>
  );
};

export default LSTMPredictor;