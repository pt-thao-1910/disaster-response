import sys
import pandas as pd
import string
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(database_filepath):
    '''
    Load data from the database filepath.
    + Input: database_filepath
    + Output: 
        1. X: message
        2. Y: values of target features
        3. categories: names of target features 
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("select * from messages_and_categories", engine)
    X = df["message"]
    # child_alone only has 1 unique value -> no need to predict this target variable
    Y = df.drop(columns=["id", "message", "original", "genre", "child_alone"])
    categories = Y.columns
    return X, Y, categories

def tokenize(text):
    '''
    Clean and tokenize the raw text.
    + Input: text
    + Output: words - list of words tokenized from raw text
    '''
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    words = [w for w in words if w not in stopwords.words('english') ]
    return words

def build_model():
    '''
    Build pipeline to tokenize, transform and train text data.
    Best model was chosen after doing GridSearchCV on max_features and ngram_range of CounterVectorizer.
    '''
    vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()
    clf = MultiOutputClassifier(LogisticRegression(random_state=0))
    
    pipeline = Pipeline([
        ("vect", vect),
        ("tfidf", tfidf),
        ("clf", clf)
    ])
    
    param_grid = {
    'vect__max_features': [3000, 5000],
    'vect__ngram_range': [(1,1), (1,2)]
    }

    # Instantiate GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=3)
    return grid_search

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate trained model.
    + Input:
        1. model: trained model
        2. X_test: values of testing explanatory features (messages)
        3. Y_test: values of testing target features
        4. category_names: names of target features
    + Output: None
    '''
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_pred[:, i], Y_test.iloc[:, i]))

def save_model(model, model_filepath):
    '''
    Save model to a pkl file.
    '''
    f = open(model_filepath, 'wb')
    pickle.dump(obj=model, file=f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
