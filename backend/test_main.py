from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
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

# Mock data for testing
MOCK_ITEMS = [
    "eggs", "milk", "bread", "chicken breast", "salmon", "bananas", "apples", "spinach",
    "broccoli", "quinoa", "brown rice", "oats", "almonds", "greek yogurt", "avocado",
    "sweet potato", "olive oil", "tomatoes", "carrots", "orange juice", "lean beef",
    "tuna", "cottage cheese", "blueberries", "whole wheat pasta", "black beans",
    "bell peppers", "onions", "garlic", "lemon"
]

def get_mock_recommendations(item_id: str, budget: float, top_n: int = 5):
    import random
    
    # Mock recommendations with realistic data
    recommendations = []
    total_cost = 0
    
    for i in range(min(top_n, len(MOCK_ITEMS))):
        item = random.choice([item for item in MOCK_ITEMS if item != item_id])
        price = round(random.uniform(2.0, 25.0), 2)
        
        if total_cost + price <= budget:
            recommendations.append({
                "item": item,
                "relevance": round(random.uniform(0.5, 1.0), 2),
                "price": price,
                "nova_class": random.choice([
                    "Unprocessed or minimally processed foods",
                    "Processed culinary ingredients", 
                    "Processed foods",
                    "Ultra-processed foods"
                ]),
                "health_score": round(random.uniform(1.0, 3.0), 1)
            })
            total_cost += price
    
    avg_relevance = sum(r['relevance'] for r in recommendations) / len(recommendations) if recommendations else 0
    return recommendations, total_cost, avg_relevance

@app.get("/")
async def root():
    return {"message": "Health Intelligent Virtual Shopping Assistant API (Test Version)"}

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
    import random
    if item_id not in MOCK_ITEMS:
        raise HTTPException(status_code=404, detail="Item not found")
    
    health_score = round(random.uniform(1.0, 3.0), 1)
    return {"item": item_id, "health_score": health_score}

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_item_recommendations(request: RecommendationRequest):
    if request.item_id not in MOCK_ITEMS:
        raise HTTPException(status_code=404, detail="Item not found")
    
    recommendations, total_cost, avg_relevance = get_mock_recommendations(
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