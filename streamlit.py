import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import tensorflow as tf
from scipy.spatial.distance import cosine
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Health Intelligent Virtual Shopping Assistant", page_icon="🍽️", layout="wide")

# Custom CSS to improve the app's appearance
st.markdown("""
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .dataframe {
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("food_classes_edited_twice.csv", na_values=["<NA>", "nan", "Nill", "Nil"])
    df = df.head(25000)  # Use more data if available
    
    # Data Preprocessing
    df['uom_criteria'].fillna(method='ffill', inplace=True)
    df['conversion'].fillna(method='ffill', inplace=True)
    df['price_new'].fillna(df['price_new'].mean(), inplace=True)
    df['price_uom'].fillna(df['price_uom'].mean(), inplace=True)
    df.drop(['dob_new', 'age_group', 'Unnamed: 0'], axis=1, inplace=True)
    
    le = LabelEncoder()
    categorical_cols = ['item_type', 'class_name', 'subclass_name', 'customer_type', 'standard_uom', 'class_name_uom']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    ohe = OneHotEncoder(sparse_output=False)
    nova_encoded = ohe.fit_transform(df[['nova']])
    nova_columns = [f'nova_{i}' for i in range(nova_encoded.shape[1])]
    df[nova_columns] = nova_encoded
    
    # Diagnostic information for NOVA classification
    nova_mapping = {}
    for i in range(4):
        most_common = df[df[f'nova_{i}'] == 1]['nova'].mode().iloc[0]
        nova_mapping[f'nova_{i}'] = most_common
    
    df.drop('nova', axis=1, inplace=True)
    
    df['original_price'] = df['price_new']
    scaler = StandardScaler()
    numerical_cols = ['price_new', 'conversion', 'price_uom']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Feature Engineering
    df['price_per_unit'] = df['price_new'] / df['conversion']
    df['health_score'] = df['nova_0'] * 3 + df['nova_1'] * 2 + df['nova_2'] * 1 + df['nova_3'] * 0
    df['price_category'] = pd.qcut(df['price_new'], q=5, labels=[1, 2, 3, 4, 5])
    df['is_brand'] = df['description'].str.contains('brand', case=False).astype(int)
    
    return df, nova_mapping

# Market Basket Analysis
@st.cache_data
def perform_market_basket_analysis(df):
    transactions = df.groupby('transaction_id')['description'].apply(list).values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(transaction_df, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules = rules.sort_values('lift', ascending=False)
    return rules

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('my_recommendation_model.keras')

# Recommendation function
def get_recommendations(item_id, budget, df, rules, model, top_n=5):
    item_to_index = {item: idx for idx, item in enumerate(df['description'].unique())}
    index_to_item = {idx: item for item, idx in item_to_index.items()}
    
    item_embeddings = model.get_layer('embedding').get_weights()[0]
    
    def cosine_similarity(a, b):
        return 1 - cosine(a, b)
    
    item_rules = rules[rules['antecedents'].apply(lambda x: item_id in x)]
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
                recommendations.append((item, similarities[item_idx], item_price, nova_class))
                total_cost += item_price
                total_relevance += similarities[item_idx]
                if len(recommendations) == top_n:
                    break

    avg_relevance = total_relevance / len(recommendations) if recommendations else 0
    return sorted(recommendations, key=lambda x: x[1], reverse=True), total_cost, avg_relevance

    for item_idx in similarities.argsort()[::-1]:
        item = index_to_item[item_idx]
        if item not in considered_items and item != item_id:
            considered_items.add(item)
            item_price = df[df['description'] == item]['original_price'].iloc[0]
            if total_cost + item_price <= budget:
                recommendations.append((item, similarities[item_idx], item_price))
                total_cost += item_price
                total_relevance += similarities[item_idx]
                if len(recommendations) == top_n:
                    break

    avg_relevance = total_relevance / len(recommendations) if recommendations else 0
    return sorted(recommendations, key=lambda x: x[1], reverse=True), total_cost, avg_relevance

# Search function
def search_items(query, items):
    query = query.lower()
    return [item for item in items if query in item.lower()]

# Streamlit app
def main():
    st.title("🍽️ Smart Food Recommender")

    df, nova_mapping = load_and_preprocess_data()
    rules = perform_market_basket_analysis(df)
    model = load_model()

    # Display NOVA mapping
    st.sidebar.subheader("NOVA Classifications")
    st.sidebar.write(f"Health Score '3': Unprocessed or minimally processed foods")
    st.sidebar.write(f"Health Score '2': Processed culinary ingredients")
    st.sidebar.write(f"Health Score '1': Processed foods")
    st.sidebar.write(f"Health Score '0': Ultra-processed foods")

    # Search bar
    search_query = st.sidebar.text_input("🔍 Search for an item")
    if search_query:
        matching_items = search_items(search_query, df['description'].unique())
        if matching_items:
            item_id = st.sidebar.selectbox("Select an item", matching_items)
        else:
            st.sidebar.write("No matching items found.")
            item_id = None
    else:
        item_id = st.sidebar.selectbox("Select an item", df['description'].unique())

    if item_id:
        # Display health score
        health_score = df[df['description'] == item_id]['health_score'].iloc[0]
        st.sidebar.write(f"Health Score: {health_score:.2f} / 3.00")
        
        # Health score gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = health_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Health Score"},
            gauge = {
                'axis': {'range': [None, 3], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 1], 'color': 'red'},
                    {'range': [1, 2], 'color': 'yellow'},
                    {'range': [2, 3], 'color': 'green'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': health_score}}))
        st.sidebar.plotly_chart(fig, use_container_width=True)

        budget = st.sidebar.slider("💰 Set your budget", min_value=10, max_value=1000, value=500, step=10)

        if st.sidebar.button("Get Recommendations"):
            with st.spinner('Generating recommendations...'):
                recommendations, total_cost, avg_relevance = get_recommendations(item_id, budget, df, rules, model)
            
            st.success('Recommendations generated!')
            
            st.subheader("🛒 Recommendations")
            st.write(f"Selected Item: **{item_id}**")
            st.write(f"Budget: **${budget:.2f}**")
            st.write(f"Total Cost: **${total_cost:.2f}**")
            
            # Display recommendations in a table
            nova_descriptions = {
                'nova_0': 'Ultra-processed foods',
                'nova_1': 'Processed foods',
                'nova_2': 'Processed culinary ingredients',
                'nova_3': 'Unprocessed or minimally processed foods'
            }
            
            rec_df = pd.DataFrame(recommendations, columns=['Item', 'Relevance', 'Price', 'NOVA'])
            rec_df['Price'] = rec_df['Price'].apply(lambda x: f"${x:.2f}")
            rec_df['NOVA'] = rec_df['NOVA'].map(nova_descriptions)
            rec_df['Relevance'] = rec_df['Relevance'].apply(lambda x: f"{x:.2f}")
            st.table(rec_df)
            
            # Visualizations
            st.subheader("📊 Visualizations")
            
            # Bar chart of recommended item prices
            fig = px.bar(rec_df, x='Item', y='Price', title='Recommended Item Prices', color='NOVA')
            st.plotly_chart(fig)
            
            # Pie chart of NOVA classification for recommended items
            nova_counts = rec_df['NOVA'].value_counts()
            fig = px.pie(values=nova_counts.values, names=nova_counts.index, title='NOVA Classification of Recommended Items')
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
