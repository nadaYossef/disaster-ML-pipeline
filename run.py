import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine, inspect
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens

# Load data
engine = create_engine('sqlite:///../DisasterResponse.db')
inspector = inspect(engine)

# Get the first table name dynamically
table_names = inspector.get_table_names()
if table_names:
    # Fetch the data from the first table
    df = pd.read_sql_table(table_names[0], engine)
else:
    df = pd.DataFrame()  # Handle case where there are no tables

# Load model
model = joblib.load("classifier.pkl")

# Helper function for word cloud generation
def generate_wordcloud(text_column):
    text = " ".join(text_column)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Save the image to a string
    img = BytesIO()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img, format='png')
    img.seek(0)

    img_b64 = base64.b64encode(img.getvalue()).decode('utf8')
    return img_b64

# Helper function for question 2 - most frequent words
def most_frequent_words(text_column, n=10):
    tokens = " ".join(text_column).split()
    word_counts = Counter(tokens)
    common_words = word_counts.most_common(n)

    words = [word for word, count in common_words]
    counts = [count for word, count in common_words]

    return words, counts

@app.route('/')
@app.route('/index')
def index():
    if df.empty:
        return render_template('error.html', message="DataFrame is empty. Cannot display visualizations.")

    # Q2: What is the distribution of the `related` column?
    related_counts = df['related'].value_counts()
    related_labels = ['Related', 'Not Related']

    # Q3: Distribution of offers
    offer_counts = df['offer'].value_counts()
    offer_labels = ['No Offer', 'Offer']

    # Create visualizations
    graphs = [
        # Pie chart for related messages
        {
            'data': [
                Pie(
                    labels=related_labels,
                    values=related_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Related Messages'
            }
        },
        # Pie chart for offers
        {
            'data': [
                Pie(
                    labels=offer_labels,
                    values=offer_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Offers'
            }
        }
    ]

    # Encode Plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Generate word cloud for Q2: most frequent words
    img_b64 = generate_wordcloud(df['message'].astype(str))

    return render_template('master.html', ids=ids, graphJSON=graphJSON, wordcloud_img=img_b64)

@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '')

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the go.html page
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()
