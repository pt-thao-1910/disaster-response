# Disaster Response Pipeline Project

### Overview:
This is a project which aims to classify messages during disaster times into 36 category, for example "food" or "medical-help". 
Using this classification, I hope that we can reach out and provide better help to people when natural disasters occur.

### File Structure
- Root directory
    - app
        - templates
            - go.html
            - master.html
        - run.py
    - data
        - DisasterResponse.db
        - disaster_categories.csv
        - disaster_messages.csv
        - process_data.py
    - models
        - classifier.pkl
        - train.classifier.py
    - README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Improvements:
As of now, the data is very imbalanced in some message categories, for example "offer" (0.4%) or "shops" (0.4%). This imbalance makes it almost impossible for the model to predict accurately whether one message is belonged to these categories. This problem may be alleviated by using oversampling, which means to create a transformed version of our data to ensure a more balanced class distribution. One way to conduct "oversampling" is to use SMOTE, however, as there is a version conflict between imbalance-learn and scikit-learn, we chose to skip this step when building the model.
