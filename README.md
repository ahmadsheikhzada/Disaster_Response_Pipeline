# Disaster_Response_Pipeline
In this project, using the real messages that were sent during disaster events, I have created a machine learning pipeline to categorize these events so that one can send the messages to an appropriate disaster relief agency. This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also dispaly visualization of the data. The project involves, data wrangling, NLP, machine learning classiafication pipeline and developing web app using flask and plotly.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
