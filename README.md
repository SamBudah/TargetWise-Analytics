# TargetWise Analytics: Customer Segmentation for Kenyan Cooperatives

## Overview
TargetWise Analytics is a machine learning project designed to help Kenyan cooperative businesses (e.g., Saccos, agricultural coops) segment their customers for targeted marketing. The project uses KMeans clustering to group customers into 4 segments and a Random Forest Classifier to predict segments for new customers in real-time. It was developed in Google Colab using Python, scikit-learn, and Streamlit.

### Key Features
- **Customer Segmentation**: Groups customers into 4 clusters using KMeans clustering.
- **Real-Time Prediction**: Predicts the segment of new customers using a Streamlit app.
- **Actionable Insights**: Provides marketing recommendations for each cluster (e.g., loyalty programs for Cluster 1).
- **Accuracy**: Achieves 97.35% accuracy with the Random Forest Classifier.

## Repository Contents
- `Group_02.ipynb`: The Google Colab notebook containing the full pipeline (data preprocessing, clustering, classification, visualization).
- `app.py`: The Streamlit app script for real-time segment prediction.
- `rf_model.pkl`: The trained Random Forest model.
- `scaler.pkl`: The StandardScaler used for preprocessing.
- `documentation.md`: Comprehensive documentation of the project.
- `credit_card_data.csv` (optional): The dataset used for training (or provide a link to the dataset).

## Dataset
The project uses the Kaggle Credit Card Dataset, which contains 1319 records with features like age, income, expenditure, and more. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets) or use your own dataset with similar features.

## Setup Instructions
### Prerequisites
- Google Colab account
- Python libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, `pyngrok`
- ngrok authtoken (sign up at [ngrok.com](https://ngrok.com))

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourGitHubUsername/TargetWise-Analytics.git
   cd TargetWise-Analytics