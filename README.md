# AI_final

# Health Intelligent Virtual Shopping Assistant - Recommendation System 
For our project, we built and deployed a recommendation system using a combination of machine learning techniques and a pre-trained BERT model for embedding generation. The system recommends items based on a given item ID and a budget constraint, utilizing a custom-built deep learning model.

# Table of Contents
* Overview
* Features
* Installation
* Usage
   * Downloading and Opening the Repository
   * Running the application
* Hosting
   * Local Server
* Project Structure

# Overview
The recommendation system leverages various machine learning algorithms and techniques to preprocess data, generate embeddings, and build a recommendation model. It is implemented in Python and uses libraries such as TensorFlow, Scikit-Learn, and Streamlit for the web application.  

# Features
* Data preprocessing and cleaning
* Embedding generation using BERT
* Recommendation model built with TensorFlow
* Interactive web application with Streamlit
* Error analysis and model evaluation

# Installation
1. Download the repository as a ZIP file from GitHub:
  * Go to the repository's GitHub page.
  * Click on the "Code" button.
  * Select "Download ZIP".
2. Extract the ZIP file to a desired location on your system.
3. Open a command line or terminal window and navigate to the extracted directory:
~ cd path/to/extracted/directory

# Usage
1. Download the repository as a ZIP file from GitHub:
  * Go to the repository's GitHub page.
  * Click on the "Code" button.
  * Select "Download ZIP".
2. Extract the ZIP file to a desired location on your system.
3. Open a command line or terminal window and navigate to the extracted directory:
~code: cd path/to/extracted/directory
4. Running the Application
Run the Jupyter Notebook for data preprocessing and model training:
~code: jupyter notebook runstream.ipynb
Run the Streamlit web application:
~code: streamlit run streamlit.py

# Hosting
To host the application on a local server, follow these steps:
1.Ensure you have completed the Installation steps.
2.Run the Streamlit application:
~code: streamlit run streamlit.py

# Project Structure
  * runstream.ipynb: Jupyter Notebook for data preprocessing and model training.
  * streamlit.py: Streamlit script for the web application.
  * my_recommendation_model.keras: Saved Keras model.
  * requirements.txt: List of dependencies required for the project.
