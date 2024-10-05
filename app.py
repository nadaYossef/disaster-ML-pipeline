from flask import Flask, render_template, request
import joblib
import pandas as pd
import sqlite3
import plotly.express as px

app = Flask(__name__)

# Load the model
model = joblib.load("classifier.pkl")


def load_data_from_db():
    """
    Connect to the SQLite database, retrieve the table name, and load the data
    from the table into a Pandas DataFrame.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the database.
    """
    conn = sqlite3.connect('DisasterResponse.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_name = cursor.fetchone()[0]  # Fetch the first table name
    df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
    conn.close()
    return df


# Load the data from the database once at the start
df = load_data_from_db()


@app.route('/')
def home():
    """
    Render the homepage of the web application, which contains the form for users
    to enter a message for disaster category prediction.

    Returns:
        str: Rendered HTML template for the homepage.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle the prediction of disaster categories based on user input. The user submits
    a message, and the model predicts which disaster-related categories the message falls into.

    Returns:
        str: Rendered HTML template with the prediction results (categories) and original message.
    """
    message = request.form['message']  # Get the user input message
    prediction = model.predict([message])[0]  # Predict the categories
    categories = dict(zip(df.columns[4:], prediction))  # Create a dictionary of predicted categories
    return render_template('results.html', categories=categories, message=message)


@app.route('/visualizations')
def visualizations():
    """
    Generate various visualizations based on the data in the SQLite database.
    The visualizations include:
    1. Distribution of message genres.
    2. Top 10 most frequent disaster categories.
    3. Histogram showing the number of messages containing disaster categories.
    4. Correlation heatmap of the different disaster categories.
    5. Histogram of message lengths.

    Returns:
        str: Rendered HTML template with embedded visualizations.
    """
    # Visualization 1: Distribution of genres
    genre_count = df['genre'].value_counts()
    genre_fig = px.bar(genre_count, x=genre_count.index, y=genre_count.values, title="Message Genres")

    # Visualization 2: Top 10 disaster categories
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)[:10]
    category_fig = px.pie(values=category_counts.values, names=category_counts.index, title="Top 10 Most Frequent Disaster Categories")
    
    # Visualization 3: Messages per category
    categories = df.iloc[:, 4:]
    category_message_counts = (categories != 0).sum(axis=1).value_counts()
    messages_per_category_fig = px.histogram(category_message_counts, x=category_message_counts.index, 
                                             y=category_message_counts.values, title="Messages per Category (Histogram)")
    
    # Visualization 4: Heatmap of category correlations
    correlation_matrix = df.iloc[:, 4:].corr()
    heatmap_fig = px.imshow(correlation_matrix, title="Heatmap of Category Correlations", aspect="auto", color_continuous_scale='Viridis')

    # Visualization 5: Length of messages (histogram)
    df['message_length'] = df['message'].apply(len)
    message_length_fig = px.histogram(df, x='message_length', nbins=50, title='Distribution of Message Lengths')

    # Render the HTML template with the visualizations
    return render_template('visualizations.html', 
                           genre_fig=genre_fig.to_html(),
                           category_fig=category_fig.to_html(),
                           messages_per_category_fig=messages_per_category_fig.to_html(),
                           heatmap_fig=heatmap_fig.to_html(),
                           message_length_fig=message_length_fig.to_html())


if __name__ == '__main__':
    app.run(debug=True)
