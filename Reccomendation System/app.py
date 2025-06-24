from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def displayReccomendedProducts():
    if request.method == 'POST':
        # Accept JSON or form data for username
        if request.is_json:
            user_input = request.get_json().get('username')
        else:
            user_input = request.form.get('username')
        if not user_input:
            return "Username is required.", 400

        user_based_prediction_df = pd.read_csv("user_based_prediction_df.csv")
        df_clean = pd.read_csv("df_clean.csv")
        #print(user_based_prediction_df.head())

        if user_input not in user_based_prediction_df.columns:
            return f"User '{user_input}' not found.", 404
        print(f"User input received: {user_input}")
        df_indexed = user_based_prediction_df.set_index('id')
        top_n = df_indexed[user_input].sort_values(ascending=False).head(20)
        print("Top 20 products for user:", top_n)
        recommended_products_df = df_clean[df_clean['id'].isin(top_n.index)].copy()
        print(recommended_products_df.head())
        lr_model = joblib.load('lr_smote_pipeline.pkl')
        recommended_products_df['predicted_sentiment'] = lr_model.predict(recommended_products_df['reviews_text'])
        #print("Predicted sentiments:", recommended_products_df['predicted_sentiment'].value_counts())
        sentiment_map = {0: 'Negative', 1: 'Positive'}
        recommended_products_df['predicted_sentiment_label'] = recommended_products_df['predicted_sentiment'].map(sentiment_map)
        positive_recommendations = recommended_products_df[recommended_products_df['predicted_sentiment_label'] == 'Positive']
        positive_percentage = (len(positive_recommendations) / len(recommended_products_df)) * 100
        top_positive_reviews = positive_recommendations.groupby('id').first().reset_index()

        result_html = f"<h2>Top 5 recommended products for user {user_input} with positive sentiment:</h2>"
        result_html += f"<p>Percentage of positive reviews: {positive_percentage:.2f}%</p>"
        result_html += "<table border='1'><tr><th>Product Name</th><th>Review</th></tr>"
        for _, row in top_positive_reviews[['name', 'reviews_text']].head(5).iterrows():
            result_html += f"<tr><td>{row['name']}</td><td>{row['reviews_text']}</td></tr>"
        result_html += "</table>"
        return result_html
    else:
        return render_template("index.html")

@app.route("/submit")
def submit():
    return "Hello from submit page"

if __name__ == '__main__':
    app.run()
