from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import tensorflow as tf
from scipy.spatial.distance import cosine
import uvicorn
from typing import List, Optional
import os

app = FastAPI(title="Health Intelligent Virtual Shopping Assistant API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for cached data
df = None
rules = None
model = None
nova_mapping = {}

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

def load_and_preprocess_data():
    global df, nova_mapping
    try:
        data_path = "../food_classes_edited_twice.csv"
        if not os.path.exists(data_path):
            data_path = "./food_classes_edited_twice.csv"
        
        df_temp = pd.read_csv(data_path, na_values=["<NA>", "nan", "Nill", "Nil"])
        df_temp = df_temp.head(25000)
        
        # Data Preprocessing
        df_temp['uom_criteria'].ffill(inplace=True)
        df_temp['conversion'].ffill(inplace=True)
        df_temp['price_new'].fillna(df_temp['price_new'].mean(), inplace=True)
        df_temp['price_uom'].fillna(df_temp['price_uom'].mean(), inplace=True)
        df_temp.drop(['dob_new', 'age_group', 'Unnamed: 0'], axis=1, inplace=True, errors='ignore')
        
        le = LabelEncoder()
        categorical_cols = ['item_type', 'class_name', 'subclass_name', 'customer_type', 'standard_uom', 'class_name_uom']
        for col in categorical_cols:
            if col in df_temp.columns:
                df_temp[col] = le.fit_transform(df_temp[col])
        
        ohe = OneHotEncoder(sparse_output=False)
        nova_encoded = ohe.fit_transform(df_temp[['nova']])
        nova_columns = [f'nova_{i}' for i in range(nova_encoded.shape[1])]
        df_temp[nova_columns] = nova_encoded
        
        # Create NOVA mapping
        nova_mapping = {}
        for i in range(4):
            col = f'nova_{i}'
            if col in df_temp.columns:
                most_common = df_temp[df_temp[col] == 1]['nova'].mode()
                if len(most_common) > 0:
                    nova_mapping[col] = most_common.iloc[0]
        
        df_temp.drop('nova', axis=1, inplace=True)
        
        df_temp['original_price'] = df_temp['price_new']
        scaler = StandardScaler()
        numerical_cols = ['price_new', 'conversion', 'price_uom']
        df_temp[numerical_cols] = scaler.fit_transform(df_temp[numerical_cols])
        
        # Feature Engineering
        df_temp['price_per_unit'] = df_temp['price_new'] / df_temp['conversion']
        df_temp['health_score'] = df_temp['nova_0'] * 3 + df_temp['nova_1'] * 2 + df_temp['nova_2'] * 1 + df_temp['nova_3'] * 0
        df_temp['price_category'] = pd.qcut(df_temp['price_new'], q=5, labels=[1, 2, 3, 4, 5])
        df_temp['is_brand'] = df_temp['description'].str.contains('brand', case=False).astype(int)
        
        df = df_temp
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def perform_market_basket_analysis():
    global rules
    try:
        transactions = df.groupby('transaction_id')['description'].apply(list).values.tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = apriori(transaction_df, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        rules = rules.sort_values('lift', ascending=False)
        return True
    except Exception as e:
        print(f"Error in market basket analysis: {e}")
        return False

def load_model():
    global model
    try:
        model_path = "../my_recommendation_model.keras"
        if not os.path.exists(model_path):
            model_path = "./my_recommendation_model.keras"
        model = tf.keras.models.load_model(model_path)
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def get_recommendations(item_id: str, budget: float, top_n: int = 5):
    try:
        item_to_index = {item: idx for idx, item in enumerate(df['description'].unique())}
        index_to_item = {idx: item for item, idx in item_to_index.items()}
        
        if item_id not in item_to_index:
            raise HTTPException(status_code=404, detail="Item not found")
        
        item_embeddings = model.get_layer('embedding').get_weights()[0]
        
        def cosine_similarity(a, b):
            return 1 - cosine(a, b)
        
        item_idx = item_to_index[item_id]
        item_embedding = item_embeddings[item_idx]
        similarities = np.array([cosine_similarity(item_embedding, emb) for emb in item_embeddings])
        
        recommendations = []
        total_cost = 0
        considered_items = set()
        total_relevance = 0

        for item_idx in similarities.argsort()[::-1]:
            item = index_to_item[item_idx]
            if item not in considered_items and item != item_id:
                considered_items.add(item)
                item_data = df[df['description'] == item].iloc[0]
                item_price = item_data['original_price']
                if total_cost + item_price <= budget:
                    nova_class = next(col for col in ['nova_0', 'nova_1', 'nova_2', 'nova_3'] if item_data[col] == 1)
                    
                    nova_descriptions = {
                        'nova_0': 'Ultra-processed foods',
                        'nova_1': 'Processed foods',
                        'nova_2': 'Processed culinary ingredients',
                        'nova_3': 'Unprocessed or minimally processed foods'
                    }
                    
                    recommendations.append({
                        "item": item,
                        "relevance": float(similarities[item_idx]),
                        "price": float(item_price),
                        "nova_class": nova_descriptions.get(nova_class, "Unknown"),
                        "health_score": float(item_data['health_score'])
                    })
                    total_cost += item_price
                    total_relevance += similarities[item_idx]
                    if len(recommendations) == top_n:
                        break

        avg_relevance = total_relevance / len(recommendations) if recommendations else 0
        return sorted(recommendations, key=lambda x: x['relevance'], reverse=True), total_cost, avg_relevance
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.on_event("startup")
async def startup_event():
    print("Loading data and model...")
    if not load_and_preprocess_data():
        print("Failed to load data")
        return
    if not perform_market_basket_analysis():
        print("Failed to perform market basket analysis")
        return
    if not load_model():
        print("Failed to load model")
        return
    print("Startup complete!")

@app.get("/")
async def root():
    return {"message": "Health Intelligent Virtual Shopping Assistant API"}

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
    return SearchResponse(items=matching_items)

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
    if df is None or model is None:
        raise HTTPException(status_code=503, detail="Data or model not loaded")
    
    recommendations, total_cost, avg_relevance = get_recommendations(
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