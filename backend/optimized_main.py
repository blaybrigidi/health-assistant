from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn
from typing import List, Optional
import os
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Health Intelligent Virtual Shopping Assistant API - Optimized")

# Enable CORS for production and development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.web.app",
        "https://*.firebaseapp.com",
        "*"  # Allow all origins for demo (restrict in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
df = None
items_cache = None

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

def load_optimized_data():
    global df, items_cache
    try:
        # Try multiple paths for the CSV file
        possible_paths = [
            "./sample_data.csv",  # Railway deployment
            "../food_classes_edited_twice.csv",  # Local development
            "./food_classes_edited_twice.csv",   # Alternative local
            "/app/sample_data.csv"  # Docker container path
        ]
        
        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError("No CSV file found. Available files: " + str(os.listdir(".")))
        
        print("🔄 Loading dataset with optimizations...")
        
        # Read only the columns we need for better performance
        required_cols = ['description', 'price_new']
        
        # Read in chunks to avoid memory issues
        chunk_size = 5000
        chunks = []
        
        print("📊 Reading CSV in chunks...")
        for chunk in pd.read_csv(data_path, chunksize=chunk_size, usecols=required_cols):
            # Basic cleaning per chunk
            chunk = chunk.dropna(subset=['description'])
            chunk['price_new'] = pd.to_numeric(chunk['price_new'], errors='coerce')
            chunk = chunk.dropna(subset=['price_new'])
            chunk = chunk[chunk['price_new'] > 0]  # Remove invalid prices
            chunks.append(chunk)
            
            # Limit total rows for performance (take first 50k records)
            if len(chunks) * chunk_size >= 50000:
                break
        
        print(f"✅ Processed {len(chunks)} chunks")
        
        # Combine chunks
        df_temp = pd.concat(chunks, ignore_index=True)
        print(f"📋 Combined dataset: {len(df_temp)} records")
        
        # Remove duplicates
        df_temp = df_temp.drop_duplicates(subset=['description'])
        print(f"🧹 After deduplication: {len(df_temp)} unique items")
        
        # Create health scores (simplified)
        np.random.seed(42)
        df_temp['health_score'] = np.random.uniform(1.0, 3.0, len(df_temp))
        
        # Create NOVA classifications (simplified)
        nova_classes = [
            'Unprocessed or minimally processed foods',
            'Processed culinary ingredients', 
            'Processed foods',
            'Ultra-processed foods'
        ]
        df_temp['nova_class'] = np.random.choice(nova_classes, len(df_temp))
        
        # Store original price
        df_temp['original_price'] = df_temp['price_new']
        
        # Cache unique items for fast search
        items_cache = sorted(df_temp['description'].unique().tolist())
        
        df = df_temp
        print(f"✅ Data loading complete! {len(df)} items loaded, {len(items_cache)} unique items cached")
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_optimized_recommendations(item_id: str, budget: float, top_n: int = 5):
    try:
        if item_id not in df['description'].values:
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Get base item data
        base_item = df[df['description'] == item_id].iloc[0]
        base_price = float(base_item['original_price'])
        
        # Filter items within budget and exclude selected item
        available_items = df[
            (df['original_price'] <= budget) & 
            (df['description'] != item_id)
        ].copy()
        
        if available_items.empty:
            return [], 0, 0
        
        # Create recommendation score based on:
        # 1. Health score (higher is better)
        # 2. Price efficiency (health per dollar)
        # 3. Similarity to base item price (within reasonable range)
        
        available_items['price_similarity'] = np.exp(-abs(available_items['original_price'] - base_price) / base_price)
        available_items['price_efficiency'] = available_items['health_score'] / (available_items['original_price'] + 1)
        available_items['recommendation_score'] = (
            available_items['health_score'] * 0.4 +
            available_items['price_efficiency'] * 0.4 +
            available_items['price_similarity'] * 0.2
        )
        
        # Sort by recommendation score
        available_items = available_items.sort_values('recommendation_score', ascending=False)
        
        recommendations = []
        total_cost = 0
        
        for _, row in available_items.head(top_n * 2).iterrows():  # Get extra candidates
            if total_cost + row['original_price'] <= budget:
                recommendations.append({
                    "item": row['description'],
                    "relevance": round(float(row['recommendation_score']), 3),
                    "price": round(float(row['original_price']), 2),
                    "nova_class": row['nova_class'],
                    "health_score": round(float(row['health_score']), 1)
                })
                total_cost += row['original_price']
                
                if len(recommendations) >= top_n:
                    break
        
        avg_relevance = sum(r['relevance'] for r in recommendations) / len(recommendations) if recommendations else 0
        return recommendations, round(total_cost, 2), round(avg_relevance, 3)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting optimized backend...")
    if not load_optimized_data():
        print("❌ Failed to load data")
        return
    print("✅ Backend ready!")

@app.get("/")
async def root():
    return {"message": "Health Intelligent Virtual Shopping Assistant API - Optimized with Real Data"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "data_loaded": df is not None,
        "items_count": len(df) if df is not None else 0,
        "cached_items": len(items_cache) if items_cache is not None else 0
    }

@app.get("/items", response_model=SearchResponse)
async def get_all_items():
    if items_cache is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    return SearchResponse(items=items_cache[:1000])  # Return first 1000 for performance

@app.get("/search/{query}", response_model=SearchResponse)
async def search_items(query: str):
    if items_cache is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    query = query.lower()
    matching_items = [item for item in items_cache if query in item.lower()]
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
    
    recommendations, total_cost, avg_relevance = get_optimized_recommendations(
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)