# Disaster Response ML Pipeline

![image](https://github.com/user-attachments/assets/20cffed8-1461-4792-b9f3-a54e507f92dd)

---

## Motivation
The primary goal of this project is to build a user-friendly web application that predicts the category of disaster-related messages. Using supervised learning techniques, the app classifies messages into 36 different categories, providing a visual representation of the classification results through pie charts.

## Requirements
You can find all the necessary packages listed in the `requirements.txt` file. The project utilizes the following libraries:
- **Pandas**: For data manipulation and analysis.
- **Plotly**: For visualization, specifically for creating pie charts on the main page of the web app.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For machine learning algorithms.
- **XGBoost**: For classification tasks.
- **SQLAlchemy**: For database operations.
- **NLTK**: For natural language processing tasks.
- **Flask**: For building the web application.
- **Joblib**: For saving and loading the trained model.
- **Pillow**: For image processing (if needed).

## Project Structure
### Files
- **`requirements.txt`**: Contains a list of Python packages required to run the project.
- **Data Files**:
  - **`data/categories.csv`**: Contains the `id` and `categories` for the messages.
  - **`data/messages.csv`**: Contains the `id`, `message`, `original`, and `genre` of each message.
- **Python Files**:
  - **`process_data.py`**: Handles data processing and database creation by merging `messages.csv` and `categories.csv` and cleaning the data.
  - **`train_classifier.py`**: Responsible for multiclass classification of the 36 categories from the database created by `process_data.py`, evaluates results, and saves the trained model.
  - **`run.py`**: Uses Flask and Plotly to run the web application, connecting all components in the terminal.
- **Pickle Files**:
  - **`cleaned_data.pkl`**: Contains the cleaned data from `process_data.py`.
  - **`classifier.pkl`**: Stores the trained model from `train_classifier.py`.
- **Database**:
  - **`DisasterResponse.db`**: The SQLite database created by `process_data.py`.
- **HTML Files**:
  - **`templates/master.html`**: Base template for the web app.
  - **`templates/go.html`**: Displays the results of the disaster response message classification app, extending `master.html` and including blocks for the title, message display, and classification results.

## Techniques and Algorithms Used
### Data Imbalance Handling
The project addresses data imbalance by employing techniques such as oversampling the minority classes, using class weights, and selecting appropriate evaluation metrics (e.g., F1-score, precision, and recall) to ensure that the classifier performs well across all classes. You can find the specific code used for handling data imbalance in the [process_data.py file](https://github.com/nadaYossef/disaster-ML-pipeline/process_data.py).

### Training the Classifier
The classifier is trained using XGBoost, which is chosen for its effectiveness in handling large datasets and its ability to work well with imbalanced data. The training process involves hyperparameter tuning and cross-validation to optimize performance. More details are available in the [train_classifier.py file](https://github.com/nadaYossef/disaster-ML-pipeline/train_classifier.py).

### Running the Web App
In `run.py`, Flask is used to create the web application, and Plotly generates the visualizations. The web app connects to the database and serves the classification results dynamically. For more details, see the [run.py file](https://github.com/nadaYossef/disaster-ML-pipeline/run.py).

### HTML Templates
The templates are structured to provide a seamless user experience. The `go.html` file showcases the classification results along with the corresponding pie charts, allowing users to visualize the distribution of predicted categories. Check out the [templates folder](https://github.com/nadaYossef/disaster-ML-pipeline/templates) for more information.

## How to Run
To run the application locally, follow these steps in your terminal (modify the paths according to your local setup):
```bash
$ pip install -r requirements.txt
$ python process_data.py data/messages.csv data/categories.csv DisasterResponse.db
$ python train_classifier.py DisasterResponse.db classifier.pkl
$ python run.py
```
The web app will launch, allowing you to input messages and see the corresponding classifications along with two pie charts displaying the results.

## Acknowledgments
This project utilizes a dataset provided by the Data Scientist Advanced Nanodegree from Udacity in collaboration with IBM. It serves as a project for the Data Engineering section of the course.
