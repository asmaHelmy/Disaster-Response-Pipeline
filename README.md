# Disaster-Response-Pipeline
A machine learning pipeline to categorize emergency messages based on the needs communicated by the sender.
This project was done as a part of Udacity's Data Scientist Nanodegree.

# Project phases:

1. ETL (Extracting, transforming and loading the data), then preprocessing it before modeling.

2. Modeling the data to classify the disaster messages through ML and NLP pipelines.

3. Building a simple web app to visualize the results.

## Required packages and libraries:

- python3
- pandas
- sqlalchemy
- sklearn
- nltk
- re
- pickle
- flask
- plotly

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/




