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
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Health Intelligent Virtual Shopping Assistant API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for cached data
df = None
rules = None
model = None
nova_mapping = {}
item_to_index = {}
index_to_item = {}

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
        
        print("🔄 Loading dataset...")
        df_temp = pd.read_csv(data_path, na_values=["<NA>", "nan", "Nill", "Nil"])
        df_temp = df_temp.head(10000)  # Use 10k records for good performance
        print(f"✅ Loaded {len(df_temp)} records")
        
        # Data Preprocessing
        print("🔄 Preprocessing data...")
        df_temp['uom_criteria'].ffill(inplace=True)
        df_temp['conversion'].ffill(inplace=True)
        df_temp['price_new'].fillna(df_temp['price_new'].mean(), inplace=True)
        df_temp['price_uom'].fillna(df_temp['price_uom'].mean(), inplace=True)
        
        # Drop problematic columns if they exist
        cols_to_drop = ['dob_new', 'age_group', 'Unnamed: 0']
        for col in cols_to_drop:
            if col in df_temp.columns:
                df_temp.drop(col, axis=1, inplace=True)
        
        # Handle categorical encoding
        le = LabelEncoder()
        categorical_cols = ['item_type', 'class_name', 'subclass_name', 'customer_type', 'standard_uom', 'class_name_uom']
        for col in categorical_cols:
            if col in df_temp.columns:
                df_temp[col] = le.fit_transform(df_temp[col].astype(str))
        
        # Handle NOVA encoding
        if 'nova' in df_temp.columns:
            ohe = OneHotEncoder(sparse_output=False)
            nova_encoded = ohe.fit_transform(df_temp[['nova']])
            nova_columns = [f'nova_{i}' for i in range(nova_encoded.shape[1])]
            df_temp[nova_columns] = nova_encoded
            
            # Create NOVA mapping
            nova_mapping = {}
            for i in range(min(4, nova_encoded.shape[1])):
                col = f'nova_{i}'
                if col in df_temp.columns:
                    mask = df_temp[col] == 1
                    if mask.any():
                        most_common = df_temp[mask]['nova'].mode()
                        if len(most_common) > 0:
                            nova_mapping[col] = most_common.iloc[0]
            
            df_temp.drop('nova', axis=1, inplace=True)
        else:
            # Create mock NOVA columns if not present
            for i in range(4):
                df_temp[f'nova_{i}'] = np.random.randint(0, 2, len(df_temp))
        
        # Feature Engineering
        df_temp['original_price'] = df_temp['price_new']
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = ['price_new', 'conversion', 'price_uom']
        numerical_cols = [col for col in numerical_cols if col in df_temp.columns]
        if numerical_cols:
            df_temp[numerical_cols] = scaler.fit_transform(df_temp[numerical_cols])
        
        # Create derived features
        df_temp['price_per_unit'] = df_temp['price_new'] / (df_temp['conversion'] + 0.001)  # Avoid division by zero
        df_temp['health_score'] = (df_temp.get('nova_0', 0) * 3 + 
                                 df_temp.get('nova_1', 0) * 2 + 
                                 df_temp.get('nova_2', 0) * 1 + 
                                 df_temp.get('nova_3', 0) * 0)
        
        df_temp['price_category'] = pd.qcut(df_temp['price_new'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        df_temp['is_brand'] = df_temp['description'].str.contains('brand', case=False, na=False).astype(int)
        
        df = df_temp
        print("✅ Data preprocessing complete!")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def perform_market_basket_analysis():
    global rules
    try:
        if 'transaction_id' not in df.columns:
            print("⚠️ No transaction_id column, skipping market basket analysis")
            rules = pd.DataFrame()  # Empty rules
            return True
            
        print("🔄 Performing market basket analysis...")
        transactions = df.groupby('transaction_id')['description'].apply(list).values.tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(transaction_df, min_support=0.005, use_colnames=True)
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            rules = rules.sort_values('lift', ascending=False)
        else:
            rules = pd.DataFrame()
        
        print("✅ Market basket analysis complete!")
        return True
    except Exception as e:
        print(f"⚠️ Market basket analysis failed: {e}")
        rules = pd.DataFrame()  # Empty rules as fallback
        return True

def load_model():
    global model, item_to_index, index_to_item
    try:
        model_path = "../my_recommendation_model.keras"
        if not os.path.exists(model_path):
            model_path = "./my_recommendation_model.keras"
        
        print("🔄 Loading ML model...")
        model = tf.keras.models.load_model(model_path)
        
        # Create item mappings
        unique_items = df['description'].unique()
        item_to_index = {item: idx for idx, item in enumerate(unique_items)}
        index_to_item = {idx: item for item, idx in item_to_index.items()}
        
        print("✅ ML model loaded successfully!")
        return True
    except Exception as e:
        print(f"⚠️ Could not load ML model: {e}")
        print("🔄 Using fallback recommendation system...")
        model = None
        return True

def get_recommendations(item_id: str, budget: float, top_n: int = 5):
    try:
        if item_id not in df['description'].values:
            raise HTTPException(status_code=404, detail="Item not found")
        
        if model is not None and item_id in item_to_index:
            return get_ml_recommendations(item_id, budget, top_n)
        else:
            return get_fallback_recommendations(item_id, budget, top_n)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

def get_ml_recommendations(item_id: str, budget: float, top_n: int = 5):
    """Use the actual ML model for recommendations"""
    try:
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
                item_price = float(item_data['original_price'])
                
                if total_cost + item_price <= budget:
                    # Find NOVA classification
                    nova_class = 'Unknown'
                    for col in ['nova_0', 'nova_1', 'nova_2', 'nova_3']:
                        if col in item_data and item_data[col] == 1:
                            nova_descriptions = {
                                'nova_0': 'Ultra-processed foods',
                                'nova_1': 'Processed foods', 
                                'nova_2': 'Processed culinary ingredients',
                                'nova_3': 'Unprocessed or minimally processed foods'
                            }
                            nova_class = nova_descriptions.get(col, 'Unknown')
                            break
                    
                    recommendations.append({
                        "item": item,
                        "relevance": float(similarities[item_idx]),
                        "price": item_price,
                        "nova_class": nova_class,
                        "health_score": float(item_data.get('health_score', 0))
                    })
                    total_cost += item_price
                    total_relevance += similarities[item_idx]
                    
                    if len(recommendations) == top_n:
                        break

        avg_relevance = total_relevance / len(recommendations) if recommendations else 0
        return sorted(recommendations, key=lambda x: x['relevance'], reverse=True), total_cost, avg_relevance
        
    except Exception as e:
        print(f"ML recommendation failed: {e}, falling back to simple recommendations")
        return get_fallback_recommendations(item_id, budget, top_n)

def get_fallback_recommendations(item_id: str, budget: float, top_n: int = 5):
    """Fallback recommendation system without ML model"""
    try:
        # Get items within budget, excluding selected item
        available_items = df[
            (df['original_price'] <= budget) & 
            (df['description'] != item_id)
        ].copy()
        
        # Sort by health score and price efficiency
        available_items['efficiency'] = available_items['health_score'] / (available_items['original_price'] + 1)
        available_items = available_items.sort_values('efficiency', ascending=False)
        
        recommendations = []
        total_cost = 0
        
        for _, row in available_items.head(top_n * 3).iterrows():  # Get more candidates
            if total_cost + row['original_price'] <= budget:
                # Find NOVA classification
                nova_class = 'Processed foods'  # Default
                for col in ['nova_0', 'nova_1', 'nova_2', 'nova_3']:
                    if col in row and row[col] == 1:
                        nova_descriptions = {
                            'nova_0': 'Ultra-processed foods',
                            'nova_1': 'Processed foods',
                            'nova_2': 'Processed culinary ingredients', 
                            'nova_3': 'Unprocessed or minimally processed foods'
                        }
                        nova_class = nova_descriptions.get(col, 'Processed foods')
                        break
                
                recommendations.append({
                    "item": row['description'],
                    "relevance": float(row['efficiency']),
                    "price": float(row['original_price']),
                    "nova_class": nova_class,
                    "health_score": float(row.get('health_score', 0))
                })
                total_cost += row['original_price']
                
                if len(recommendations) >= top_n:
                    break
        
        avg_relevance = sum(r['relevance'] for r in recommendations) / len(recommendations) if recommendations else 0
        return recommendations, total_cost, avg_relevance
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallback recommendation failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Health Intelligent Virtual Shopping Assistant API...")
    
    if not load_and_preprocess_data():
        print("❌ Failed to load data")
        return
    
    if not perform_market_basket_analysis():
        print("⚠️ Market basket analysis had issues, continuing anyway")
    
    if not load_model():
        print("⚠️ ML model loading had issues, using fallback system")
    
    print("✅ API startup complete!")

@app.get("/")
async def root():
    return {"message": "Health Intelligent Virtual Shopping Assistant API - Real Data & Model"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "data_loaded": df is not None,
        "model_loaded": model is not None,
        "records_count": len(df) if df is not None else 0
    }

@app.get("/items", response_model=SearchResponse)
async def get_all_items():
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    items = df['description'].unique().tolist()
    return SearchResponse(items=items[:1000])  # Limit for performance

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