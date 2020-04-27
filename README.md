# Disaster_Response
This is repository of an web app where an emergency worker can input a new message and get classification results in several categories

To Use this - 
1. Clone this locally
2. Go to workspace folder - type command 'rm -rf venv'
3. Now create a new virtual environment - 'python -m venv venv'
4. Activate virtual environment (Windows) - venv\Scripts\activate
5. To install all required files use command  - 'pip install -r requirements.txt'
6. If you just want to run app go inside app folder - 'cd app'
7. Run python run Script - 'python run.py'
8. Else delete DisasterResponse.db file from data and classifier.pkl file from models.
9. Follow the insrtructions below.

### Instructions:
1. Run the following commands in the project's workspace dircetory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/

