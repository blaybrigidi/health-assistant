import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  TextField,
  Button,
  Card,
  CardContent,
  CardActions,
  Grid,
  Chip,
  Slider,
  CircularProgress,
  Paper,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  IconButton,
  InputAdornment
} from '@mui/material';
import {
  Search as SearchIcon,
  ShoppingCart as ShoppingCartIcon,
  Favorite as FavoriteIcon,
  Star as StarIcon,
  ArrowBack as ArrowBackIcon,
  Send as SendIcon,
  SmartToy as SmartToyIcon
} from '@mui/icons-material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import axios from 'axios';

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-app-name.railway.app'  // Replace with your Railway URL
  : 'http://localhost:8000';

// Dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2563eb',
      light: '#60a5fa',
      dark: '#1d4ed8'
    },
    secondary: {
      main: '#10b981',
      light: '#34d399',
      dark: '#059669'
    },
    background: {
      default: '#000000',
      paper: '#111111'
    },
    text: {
      primary: '#ffffff',
      secondary: '#cccccc'
    }
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      color: '#ffffff'
    }
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: '12px',
          border: '1px solid #333333',
          backgroundColor: '#111111'
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          textTransform: 'none',
          fontWeight: 600
        }
      }
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: '12px',
          border: '1px solid #333333',
          backgroundColor: '#111111'
        }
      }
    }
  }
});

// Quick prompt suggestions
const quickQuestions = [
  "Healthy breakfast options under $50",
  "High protein snacks for gym",
  "Organic vegetables in budget",
  "Low sodium meal prep ideas",
  "Gluten-free alternatives",
  "Heart-healthy dinner options"
];

function Hero({ onStart }) {
  const [typedText, setTypedText] = useState('');
  const fullText = 'Get AI-powered food recommendations tailored to your health goals and budget. Discover nutritious options that fit your lifestyle perfectly.';
  
  useEffect(() => {
    let currentIndex = 0;
    const typingInterval = setInterval(() => {
      if (currentIndex <= fullText.length) {
        setTypedText(fullText.slice(0, currentIndex));
        currentIndex++;
      } else {
        clearInterval(typingInterval);
      }
    }, 50);
    
    return () => clearInterval(typingInterval);
  }, [fullText]);

  return (
    <Container maxWidth="lg" sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Box textAlign="center">
        <Box display="flex" alignItems="center" justifyContent="center" mb={4}>
          <Box
            sx={{
              width: 60,
              height: 60,
              background: 'linear-gradient(45deg, #2563eb, #10b981)',
              borderRadius: '16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mr: 2
            }}
            className="floating-element neon-glow"
          >
            <FavoriteIcon sx={{ fontSize: 32, color: 'white' }} />
          </Box>
          <Typography variant="h4" fontWeight="bold">
            HealthLink
          </Typography>
        </Box>

        <Typography variant="h1" sx={{ fontSize: { xs: '3rem', md: '4rem' }, mb: 3, lineHeight: 1.2 }}>
          Your Smart<br />Health Assistant
        </Typography>

        <Typography variant="h6" color="text.secondary" sx={{ mb: 6, maxWidth: '600px', mx: 'auto', minHeight: '3em' }}>
          {typedText}
          <span className="typing-cursor">|</span>
        </Typography>

        <Button
          onClick={onStart}
          variant="contained"
          size="large"
          sx={{
            px: 6,
            py: 2,
            fontSize: '1.2rem',
            '&:hover': {
              transform: 'translateY(-2px)'
            }
          }}
          startIcon={<SmartToyIcon />}
        >
          Start Your Journey
        </Button>

        <Box mt={8} display="flex" justifyContent="center" gap={4} flexWrap="wrap">
          <Chip 
            icon={<SmartToyIcon className="pulsing-icon" />} 
            label="AI-Powered" 
            color="primary"
            sx={{
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-4px) scale(1.05)',
                boxShadow: '0 8px 25px rgba(37, 99, 235, 0.4)'
              }
            }}
          />
          <Chip 
            icon={<FavoriteIcon className="pulsing-heart" />} 
            label="Health-Focused" 
            color="secondary"
            sx={{
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-4px) scale(1.05)',
                boxShadow: '0 8px 25px rgba(16, 185, 129, 0.4)'
              }
            }}
          />
          <Chip 
            icon={<StarIcon />} 
            label="Budget-Friendly" 
            color="primary"
            sx={{
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-4px) scale(1.05)',
                boxShadow: '0 8px 25px rgba(37, 99, 235, 0.4)'
              }
            }}
          />
        </Box>
      </Box>
    </Container>
  );
}

function LoadingScreen() {
  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        bgcolor: 'rgba(0, 0, 0, 0.9)',
        backdropFilter: 'blur(10px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999
      }}
    >
      <Box textAlign="center">
        <CircularProgress
          color="primary"
          sx={{ mb: 2 }}
          size={60}
        />
        <Typography color="text.secondary">
          AI is analyzing your preferences...
        </Typography>
      </Box>
    </Box>
  );
}

function ProductCard({ product, onAdd, index }) {
  const getHealthColor = (score) => {
    if (score >= 2.5) return 'success';
    if (score >= 2) return 'primary';
    if (score >= 1.5) return 'warning';
    if (score >= 1) return 'secondary';
    return 'error';
  };

  const getHealthLabel = (score) => {
    if (score >= 2.5) return 'Excellent';
    if (score >= 2) return 'Healthy';
    if (score >= 1.5) return 'Moderate';
    if (score >= 1) return 'Fair';
    return 'Poor';
  };

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flexGrow: 1 }}>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
          <Typography variant="h6" component="h3" sx={{ flexGrow: 1, mr: 2 }}>
            {product.item}
          </Typography>
          <Typography variant="h5" fontWeight="bold" color="primary">
            ${product.price.toFixed(2)}
          </Typography>
        </Box>

        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Box display="flex" alignItems="center" gap={1}>
            <Chip
              icon={<FavoriteIcon />}
              label={getHealthLabel(product.health_score)}
              color={getHealthColor(product.health_score)}
              size="small"
            />
            <Typography variant="caption" color="text.secondary">
              {product.health_score.toFixed(1)}/3.0
            </Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={0.5}>
            <StarIcon sx={{ fontSize: 16, color: '#10b981' }} />
            <Typography variant="caption" color="secondary">
              {product.relevance.toFixed(2)}
            </Typography>
          </Box>
        </Box>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {product.nova_class}
        </Typography>
      </CardContent>

      <CardActions sx={{ gap: 1, p: 2 }}>
        <Button variant="outlined" size="small" sx={{ flex: 1 }}>
          Details
        </Button>
        <Button
          variant="contained"
          size="small"
          onClick={() => onAdd(product)}
          startIcon={<ShoppingCartIcon />}
          sx={{ flex: 1 }}
        >
          Add
        </Button>
      </CardActions>
    </Card>
  );
}

function MessageBubble({ message, isBot }) {
  if (isBot && message.products && message.products.length > 0) {
    return (
      <Box sx={{ width: '100%' }}>
        <Paper sx={{ p: 3, mb: 3, bgcolor: 'background.paper' }}>
          <Typography>{message.text}</Typography>
        </Paper>
        <Grid container spacing={3}>
          {message.products.map((product, index) => (
            <Grid item xs={12} sm={6} lg={4} key={index}>
              <ProductCard
                product={product}
                index={index}
                onAdd={(product) => console.log('Added:', product)}
              />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isBot ? 'flex-start' : 'flex-end',
        mb: 2
      }}
    >
      <Paper
        sx={{
          p: 2,
          maxWidth: '70%',
          bgcolor: isBot ? 'background.paper' : 'primary.main',
          color: isBot ? 'text.primary' : 'white',
          borderRadius: '16px'
        }}
      >
        <Typography>{message.text}</Typography>
      </Paper>
    </Box>
  );
}

function ChatInterface({ selectedItem, budget, onBack }) {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async (messageText) => {
    if (!messageText.trim() || loading) return;

    const userMessage = { text: messageText, isBot: false };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/recommendations`, {
        item_id: selectedItem,
        budget: budget,
        top_n: 5
      });

      const botMessage = {
        text: `Here are my top recommendations for "${selectedItem}" within your $${budget} budget:`,
        isBot: true,
        products: response.data.recommendations
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        text: "I'm having trouble getting recommendations right now. Please try again.",
        isBot: true,
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper sx={{ p: 2, bgcolor: 'background.paper' }}>
        <Container maxWidth="lg">
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Box display="flex" alignItems="center" gap={2}>
              <IconButton onClick={onBack}>
                <ArrowBackIcon />
              </IconButton>
              <Box
                sx={{
                  width: 40,
                  height: 40,
                  background: 'linear-gradient(45deg, #2563eb, #10b981)',
                  borderRadius: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                <FavoriteIcon sx={{ color: 'white' }} />
              </Box>
              <Box>
                <Typography variant="h6">HealthLink Assistant</Typography>
                <Typography variant="caption" color="text.secondary">
                  AI-powered nutrition recommendations
                </Typography>
              </Box>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Budget: <span style={{ color: '#10b981', fontWeight: 'bold' }}>${budget}</span>
            </Typography>
          </Box>
        </Container>
      </Paper>

      {/* Messages */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
        <Container maxWidth="lg">
          {messages.length === 0 && (
            <Box textAlign="center" py={8}>
              <Box
                sx={{
                  width: 80,
                  height: 80,
                  background: 'linear-gradient(45deg, #2563eb, #10b981)',
                  borderRadius: '20px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  mx: 'auto',
                  mb: 3
                }}
              >
                <SmartToyIcon sx={{ fontSize: 40, color: 'white' }} />
              </Box>
              <Typography variant="h5" gutterBottom>
                Ready to help you eat healthier!
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
                Ask me anything about nutrition, or try one of these suggestions:
              </Typography>
              
              <Grid container spacing={2}>
                {quickQuestions.map((question, index) => (
                  <Grid item xs={12} md={6} lg={4} key={index}>
                    <Button
                      onClick={() => sendMessage(question)}
                      variant="outlined"
                      sx={{
                        width: '100%',
                        p: 2,
                        textAlign: 'left',
                        justifyContent: 'flex-start',
                        textTransform: 'none'
                      }}
                    >
                      {question}
                    </Button>
                  </Grid>
                ))}
              </Grid>
            </Box>
          )}

          {messages.map((message, index) => (
            <MessageBubble key={index} message={message} isBot={message.isBot} />
          ))}

          {loading && (
            <Box display="flex" justifyContent="center" my={2}>
              <Paper sx={{ px: 3, py: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                <CircularProgress size={16} color="primary" />
                <Typography variant="body2" color="text.secondary">
                  AI is thinking...
                </Typography>
              </Paper>
            </Box>
          )}
        </Container>
      </Box>

      {/* Input */}
      <Paper sx={{ p: 2, bgcolor: 'background.paper' }}>
        <Container maxWidth="lg">
          <Box display="flex" gap={2}>
            <TextField
              fullWidth
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage(inputValue)}
              placeholder="Ask me about healthy food options..."
              variant="outlined"
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: '16px'
                }
              }}
            />
            <Button
              onClick={() => sendMessage(inputValue)}
              disabled={loading || !inputValue.trim()}
              variant="contained"
              sx={{
                minWidth: 'auto',
                px: 3,
                borderRadius: '16px'
              }}
            >
              <SendIcon />
            </Button>
          </Box>
        </Container>
      </Paper>
    </Box>
  );
}

function ItemSelector({ onSelect, budget, setBudget }) {
  const [items, setItems] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredItems, setFilteredItems] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchItems();
  }, []);

  useEffect(() => {
    if (searchQuery) {
      const filtered = items.filter(item => 
        item.toLowerCase().includes(searchQuery.toLowerCase())
      );
      setFilteredItems(filtered.slice(0, 20));
    } else {
      setFilteredItems(items.slice(0, 20));
    }
  }, [searchQuery, items]);

  const fetchItems = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/items`);
      setItems(response.data.items);
      setFilteredItems(response.data.items.slice(0, 20));
    } catch (error) {
      console.error('Failed to fetch items');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <LoadingScreen />;
  }

  return (
    <Container maxWidth="md" sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Box sx={{ width: '100%' }}>
        <Box textAlign="center" mb={6}>
          <Typography variant="h3" fontWeight="bold" gutterBottom>
            What are you looking for?
          </Typography>
          <Typography variant="h6" color="text.secondary">
            Search for a food item to get personalized recommendations
          </Typography>
        </Box>

        <Paper sx={{ p: 4, borderRadius: '24px' }}>
          <TextField
            fullWidth
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search for food items..."
            variant="outlined"
            sx={{
              mb: 4,
              '& .MuiOutlinedInput-root': {
                borderRadius: '16px',
                fontSize: '1.1rem'
              }
            }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              )
            }}
          />

          <Box mb={4}>
            <Typography variant="subtitle1" fontWeight="medium" gutterBottom>
              Budget: ${budget}
            </Typography>
            <Slider
              value={budget}
              onChange={(e, newValue) => setBudget(newValue)}
              min={10}
              max={1000}
              step={10}
              marks={[
                { value: 10, label: '$10' },
                { value: 1000, label: '$1000' }
              ]}
              color="secondary"
            />
          </Box>

          {filteredItems.length > 0 && (
            <List sx={{ maxHeight: 300, overflow: 'auto', bgcolor: 'background.default', borderRadius: '12px' }}>
              {filteredItems.map((item, index) => (
                <ListItem key={index} disablePadding>
                  <ListItemButton
                    onClick={() => onSelect(item)}
                    sx={{
                      borderRadius: '8px',
                      mb: 0.5,
                      '&:hover': {
                        bgcolor: 'rgba(37, 99, 235, 0.2)',
                        transform: 'translateX(8px)'
                      },
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <ListItemText primary={item} />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          )}

          {searchQuery && filteredItems.length === 0 && (
            <Box textAlign="center" py={4}>
              <Typography color="text.secondary">
                No items found matching "{searchQuery}"
              </Typography>
            </Box>
          )}
        </Paper>
      </Box>
    </Container>
  );
}

function App() {
  const [currentView, setCurrentView] = useState('hero'); // hero, selector, chat
  const [selectedItem, setSelectedItem] = useState(null);
  const [budget, setBudget] = useState(500);

  const handleItemSelect = (item) => {
    setSelectedItem(item);
    setCurrentView('chat');
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }} className="pattern-overlay">
        {currentView === 'hero' && (
          <Hero onStart={() => setCurrentView('selector')} />
        )}
        
        {currentView === 'selector' && (
          <ItemSelector
            onSelect={handleItemSelect}
            budget={budget}
            setBudget={setBudget}
          />
        )}
        
        {currentView === 'chat' && (
          <ChatInterface
            selectedItem={selectedItem}
            budget={budget}
            onBack={() => setCurrentView('selector')}
          />
        )}
      </Box>
    </ThemeProvider>
  );
}

export default App;