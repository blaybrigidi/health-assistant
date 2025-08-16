from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import random
from typing import List, Optional

app = FastAPI(title="Health Intelligent Virtual Shopping Assistant API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data - sample food items
MOCK_ITEMS = [
    "Apple - Fresh Organic",
    "Banana - Yellow Ripe",
    "Chicken Breast - Boneless",
    "Brown Rice - Whole Grain",
    "Spinach - Fresh Leaves",
    "Salmon - Wild Caught",
    "Greek Yogurt - Plain",
    "Broccoli - Fresh Crown",
    "Sweet Potato - Organic",
    "Quinoa - Tri-Color",
    "Avocado - Hass",
    "Blueberries - Fresh",
    "Almonds - Raw Unsalted",
    "Olive Oil - Extra Virgin",
    "Tomatoes - Cherry",
    "Eggs - Free Range",
    "Oats - Steel Cut",
    "Kale - Organic",
    "Lentils - Red",
    "Dark Chocolate - 70%"
]

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

def generate_mock_recommendation(item_name: str, budget: float):
    """Generate mock recommendation with random but realistic data"""
    price = round(random.uniform(2.99, 29.99), 2)
    health_score = round(random.uniform(1.5, 3.0), 2)
    relevance = round(random.uniform(0.6, 0.95), 3)
    
    nova_classes = [
        "Unprocessed or minimally processed foods",
        "Processed culinary ingredients", 
        "Processed foods",
        "Ultra-processed foods"
    ]
    
    return {
        "item": item_name,
        "relevance": relevance,
        "price": price,
        "nova_class": random.choice(nova_classes),
        "health_score": health_score
    }

@app.get("/")
async def root():
    return {"message": "Health Intelligent Virtual Shopping Assistant API (Minimal Version)"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "data_loaded": True}

@app.get("/items", response_model=SearchResponse)
async def get_all_items():
    return SearchResponse(items=MOCK_ITEMS)

@app.get("/search/{query}", response_model=SearchResponse)
async def search_items(query: str):
    query = query.lower()
    matching_items = [item for item in MOCK_ITEMS if query in item.lower()]
    return SearchResponse(items=matching_items)

@app.get("/item/{item_id}/health_score")
async def get_health_score(item_id: str):
    if item_id not in MOCK_ITEMS:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Generate consistent health score based on item name
    random.seed(hash(item_id) % 1000)
    health_score = round(random.uniform(1.0, 3.0), 2)
    return {"item": item_id, "health_score": health_score}

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_item_recommendations(request: RecommendationRequest):
    if request.item_id not in MOCK_ITEMS:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Generate mock recommendations
    available_items = [item for item in MOCK_ITEMS if item != request.item_id]
    random.shuffle(available_items)
    
    recommendations = []
    total_cost = 0
    
    for item in available_items[:request.top_n]:
        rec = generate_mock_recommendation(item, request.budget)
        if total_cost + rec["price"] <= request.budget:
            recommendations.append(rec)
            total_cost += rec["price"]
        
        if len(recommendations) >= request.top_n:
            break
    
    avg_relevance = sum(r["relevance"] for r in recommendations) / len(recommendations) if recommendations else 0
    
    return RecommendationResponse(
        recommendations=recommendations,
        total_cost=total_cost,
        avg_relevance=avg_relevance,
        selected_item=request.item_id,
        budget=request.budget
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting minimal backend server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)