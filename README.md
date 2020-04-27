# Disaster_Response
This is repository of an web app where an emergency worker can input a new message and get classification results in several categories. This is 
one of the most useful projects I have ever made. In this we app when someone who is in need or knows someone will enter the message, then this app will 
tell us which type of help does he/she needs and guide us which authorities to contact. 

This app contains go.html and master.html in templates folder in app. These are used for displaying index page and predicted classes message page.
App folder also contains run.py file used for running the app and responsible for preprocessing of messsage.

Data Folder contains our database from which our processed data will be loaded for training, disaster_categories.csv and disaster_messages.csv for loading
raw data for processing, and training. process_data.py file processes the data and load it in udacity_etl table in the DisasterResponse.db databse.

Models folder contain classifier.pkl file for loading trained model for making predictions ,and train_classifier.py for training and saving the model 
,and printing evaluation results.

To Use this - 
1. Clone this locally
2. Now create a new virtual environment - 'python -m venv venv'
3. Activate virtual environment (Windows) - venv\Scripts\activate
4. To install all required files use command  - 'pip install -r requirements.txt'
5. If you just want to run app go inside app folder - 'cd app'
6. Run python run Script - 'python run.py'
7. Else delete DisasterResponse.db file from data and classifier.pkl file from models.
8. Follow the insrtructions below.

### Instructions:
1. Run the following commands in the project's workspace dircetory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/

