import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  TextField, 
  Button, 
  Card, 
  CardContent, 
  Grid, 
  Box, 
  Slider, 
  CircularProgress, 
  Alert,
  Autocomplete,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper
} from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import HealthAndSafetyIcon from '@mui/icons-material/HealthAndSafety';
import axios from 'axios';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#4CAF50',
    },
    secondary: {
      main: '#FF5722',
    },
  },
});

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [items, setItems] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);
  const [budget, setBudget] = useState(500);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [healthScore, setHealthScore] = useState(null);
  const [totalCost, setTotalCost] = useState(0);

  useEffect(() => {
    fetchItems();
  }, []);

  const fetchItems = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/items`);
      setItems(response.data.items);
    } catch (err) {
      setError('Failed to load items. Make sure the backend server is running.');
    }
  };

  const fetchHealthScore = async (itemId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/item/${encodeURIComponent(itemId)}/health_score`);
      setHealthScore(response.data.health_score);
    } catch (err) {
      console.error('Failed to fetch health score:', err);
    }
  };

  const getRecommendations = async () => {
    if (!selectedItem) {
      setError('Please select an item first.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await axios.post(`${API_BASE_URL}/recommendations`, {
        item_id: selectedItem,
        budget: budget,
        top_n: 5
      });

      setRecommendations(response.data.recommendations);
      setTotalCost(response.data.total_cost);
    } catch (err) {
      setError('Failed to get recommendations. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleItemChange = (event, newValue) => {
    setSelectedItem(newValue);
    if (newValue) {
      fetchHealthScore(newValue);
    } else {
      setHealthScore(null);
    }
    setRecommendations([]);
  };

  const getHealthScoreColor = (score) => {
    if (score >= 2) return '#4CAF50'; // Green
    if (score >= 1) return '#FF9800'; // Orange
    return '#F44336'; // Red
  };

  const getHealthScoreLabel = (score) => {
    if (score >= 2.5) return 'Very Healthy';
    if (score >= 2) return 'Healthy';
    if (score >= 1) return 'Moderately Healthy';
    return 'Less Healthy';
  };

  return (
    <ThemeProvider theme={theme}>
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <Typography variant="h3" component="h1" gutterBottom sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 2 }}>
            <HealthAndSafetyIcon fontSize="large" />
            Health Intelligent Virtual Shopping Assistant
            <ShoppingCartIcon fontSize="large" />
          </Typography>
          <Typography variant="h6" color="text.secondary">
            Get personalized food recommendations based on health scores and your budget
          </Typography>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        <Grid container spacing={3}>
          {/* Left Panel - Controls */}
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Search & Select Item
                </Typography>
                
                <Autocomplete
                  options={items}
                  value={selectedItem}
                  onChange={handleItemChange}
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      label="Search for an item"
                      variant="outlined"
                      fullWidth
                    />
                  )}
                  sx={{ mb: 3 }}
                />

                {healthScore !== null && (
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Health Score: {healthScore.toFixed(2)} / 3.00
                    </Typography>
                    <Box 
                      sx={{ 
                        width: '100%', 
                        height: 20, 
                        backgroundColor: '#e0e0e0', 
                        borderRadius: 1,
                        overflow: 'hidden'
                      }}
                    >
                      <Box 
                        sx={{ 
                          width: `${(healthScore / 3) * 100}%`, 
                          height: '100%', 
                          backgroundColor: getHealthScoreColor(healthScore),
                          transition: 'width 0.3s ease'
                        }} 
                      />
                    </Box>
                    <Typography variant="caption" sx={{ color: getHealthScoreColor(healthScore) }}>
                      {getHealthScoreLabel(healthScore)}
                    </Typography>
                  </Box>
                )}

                <Typography variant="subtitle1" gutterBottom>
                  Budget: ${budget}
                </Typography>
                <Slider
                  value={budget}
                  onChange={(e, newValue) => setBudget(newValue)}
                  min={10}
                  max={1000}
                  step={10}
                  valueLabelDisplay="auto"
                  sx={{ mb: 3 }}
                />

                <Button
                  variant="contained"
                  fullWidth
                  onClick={getRecommendations}
                  disabled={!selectedItem || loading}
                  startIcon={loading ? <CircularProgress size={20} /> : <ShoppingCartIcon />}
                >
                  {loading ? 'Getting Recommendations...' : 'Get Recommendations'}
                </Button>
              </CardContent>
            </Card>
          </Grid>

          {/* Right Panel - Results */}
          <Grid item xs={12} md={8}>
            {recommendations.length > 0 && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Recommendations for: {selectedItem}
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    Budget: ${budget} | Total Cost: ${totalCost.toFixed(2)}
                  </Typography>

                  <TableContainer component={Paper}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Item</TableCell>
                          <TableCell>Price</TableCell>
                          <TableCell>Health Score</TableCell>
                          <TableCell>NOVA Classification</TableCell>
                          <TableCell>Relevance</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {recommendations.map((rec, index) => (
                          <TableRow key={index}>
                            <TableCell>{rec.item}</TableCell>
                            <TableCell>${rec.price.toFixed(2)}</TableCell>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                {rec.health_score.toFixed(2)}
                                <Box 
                                  sx={{ 
                                    width: 10, 
                                    height: 10, 
                                    borderRadius: '50%', 
                                    backgroundColor: getHealthScoreColor(rec.health_score) 
                                  }} 
                                />
                              </Box>
                            </TableCell>
                            <TableCell>{rec.nova_class}</TableCell>
                            <TableCell>{rec.relevance.toFixed(3)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            )}

            {!loading && recommendations.length === 0 && selectedItem && (
              <Card>
                <CardContent>
                  <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center' }}>
                    Click "Get Recommendations" to see personalized suggestions.
                  </Typography>
                </CardContent>
              </Card>
            )}
          </Grid>
        </Grid>

        <Box sx={{ mt: 4, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            NOVA Classifications: 3 = Unprocessed, 2 = Processed ingredients, 1 = Processed foods, 0 = Ultra-processed
          </Typography>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
