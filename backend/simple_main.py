from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn
from typing import List, Optional
import os

app = FastAPI(title="Health Intelligent Virtual Shopping Assistant API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
df = None

class RecommendationRequest(BaseModel):
    item_id: str
    budget: float
    top_n: Optional[int] = 5

class RecommendationResponse(BaseModel):
    recommendations: List[dict]
    total_cost: float
    avg_relevance: float
    selected_item: str
    budget: float

class SearchResponse(BaseModel):
    items: List[str]

def load_simple_data():
    global df
    try:
        data_path = "../food_classes_edited_twice.csv"
        if not os.path.exists(data_path):
            data_path = "./food_classes_edited_twice.csv"
        
        print("Loading dataset...")
        df_temp = pd.read_csv(data_path, na_values=["<NA>", "nan", "Nill", "Nil"])
        
        # Take only first 1000 rows for fast startup
        df_temp = df_temp.head(1000)
        print(f"Loaded {len(df_temp)} records")
        
        # Simple preprocessing
        df_temp['price_new'].fillna(df_temp['price_new'].mean(), inplace=True)
        
        # Create simple health score (mock for now)
        np.random.seed(42)
        df_temp['health_score'] = np.random.uniform(0, 3, len(df_temp))
        df_temp['original_price'] = df_temp['price_new']
        
        df = df_temp
        print("Data loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def get_simple_recommendations(item_id: str, budget: float, top_n: int = 5):
    try:
        if item_id not in df['description'].values:
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Simple recommendation logic
        # Filter items within budget and exclude selected item
        available_items = df[
            (df['original_price'] <= budget) & 
            (df['description'] != item_id)
        ].copy()
        
        # Sort by health score (descending) and price (ascending)
        available_items['score'] = available_items['health_score'] / available_items['original_price']
        available_items = available_items.sort_values('score', ascending=False)
        
        recommendations = []
        total_cost = 0
        
        for _, row in available_items.head(top_n).iterrows():
            if total_cost + row['original_price'] <= budget:
                recommendations.append({
                    "item": row['description'],
                    "relevance": float(row['score']),
                    "price": float(row['original_price']),
                    "nova_class": "Processed foods",  # Mock classification
                    "health_score": float(row['health_score'])
                })
                total_cost += row['original_price']
        
        avg_relevance = sum(r['relevance'] for r in recommendations) / len(recommendations) if recommendations else 0
        return recommendations, total_cost, avg_relevance
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.on_event("startup")
async def startup_event():
    print("Starting simple backend...")
    if not load_simple_data():
        print("Failed to load data")
        return
    print("Backend ready!")

@app.get("/")
async def root():
    return {"message": "Health Intelligent Virtual Shopping Assistant API (Simple Version)"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "data_loaded": df is not None}

@app.get("/items", response_model=SearchResponse)
async def get_all_items():
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    return SearchResponse(items=df['description'].unique().tolist())

@app.get("/search/{query}", response_model=SearchResponse)
async def search_items(query: str):
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    query = query.lower()
    matching_items = [item for item in df['description'].unique() if query in item.lower()]
    return SearchResponse(items=matching_items[:50])  # Limit results

@app.get("/item/{item_id}/health_score")
async def get_health_score(item_id: str):
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    item_data = df[df['description'] == item_id]
    if item_data.empty:
        raise HTTPException(status_code=404, detail="Item not found")
    
    health_score = float(item_data['health_score'].iloc[0])
    return {"item": item_id, "health_score": health_score}

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_item_recommendations(request: RecommendationRequest):
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    recommendations, total_cost, avg_relevance = get_simple_recommendations(
        request.item_id, request.budget, request.top_n
    )
    
    return RecommendationResponse(
        recommendations=recommendations,
        total_cost=total_cost,
        avg_relevance=avg_relevance,
        selected_item=request.item_id,
        budget=request.budget
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)