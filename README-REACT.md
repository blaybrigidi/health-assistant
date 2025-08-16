# Health Intelligent Virtual Shopping Assistant

A React frontend with FastAPI backend for personalized food recommendations based on health scores and budget constraints.

## Project Structure

```
├── backend/                # FastAPI backend
│   ├── main.py            # Main API server
│   └── requirements.txt   # Python dependencies
├── frontend/              # React frontend
│   ├── src/
│   │   └── App.js        # Main React component
│   └── package.json      # Node.js dependencies
├── food_classes_edited_twice.csv    # Dataset
├── my_recommendation_model.keras     # ML model
├── start-backend.sh       # Backend startup script
└── start-frontend.sh      # Frontend startup script
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### 1. Backend Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install backend dependencies
cd backend
pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
# Install frontend dependencies
cd frontend
npm install
```

## Running the Application

### Option 1: Using startup scripts (recommended)

**Terminal 1 - Start Backend:**
```bash
./start-backend.sh
```
This will start the FastAPI server on `http://localhost:8000`

**Terminal 2 - Start Frontend:**
```bash
./start-frontend.sh
```
This will start the React app on `http://localhost:3000`

### Option 2: Manual startup

**Backend:**
```bash
cd backend
source ../venv/bin/activate
python main.py
```

**Frontend:**
```bash
cd frontend
npm start
```

## Features

- 🔍 **Smart Search**: Autocomplete search through thousands of food items
- 🎯 **Health Scoring**: NOVA classification-based health scores (0-3 scale)
- 💰 **Budget Management**: Set your budget and get recommendations within limits
- 📊 **Visual Health Indicators**: Color-coded health scores and progress bars
- 📋 **Detailed Results**: Comprehensive table with prices, health scores, and classifications
- 🎨 **Modern UI**: Material-UI dark theme with responsive design

## API Endpoints

- `GET /items` - Get all available items
- `GET /search/{query}` - Search items by query
- `GET /item/{item_id}/health_score` - Get health score for specific item
- `POST /recommendations` - Get personalized recommendations

## Health Scoring System

- **3.0**: Unprocessed or minimally processed foods (healthiest)
- **2.0**: Processed culinary ingredients
- **1.0**: Processed foods
- **0.0**: Ultra-processed foods (least healthy)

## Troubleshooting

1. **Backend not starting**: 
   - Ensure all required files are present (`food_classes_edited_twice.csv`, `my_recommendation_model.keras`)
   - Check that virtual environment is activated
   - Install missing dependencies: `pip install -r backend/requirements.txt`

2. **Frontend not loading items**:
   - Ensure backend is running on port 8000
   - Check browser console for CORS errors
   - Verify API endpoints are accessible

3. **Recommendations not working**:
   - Check that ML model file exists and is loadable
   - Ensure selected item exists in dataset
   - Check backend logs for errors

## Development

- Backend uses FastAPI with automatic API documentation at `http://localhost:8000/docs`
- Frontend uses React with Material-UI components
- CORS is configured to allow requests from `localhost:3000`

## Performance Notes

- Initial startup may take 1-2 minutes while loading the ML model and dataset
- The app uses 25,000 data points for faster performance
- API responses are optimized for real-time recommendations