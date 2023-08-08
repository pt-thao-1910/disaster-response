import json
import plotly
import pandas as pd
import numpy as np
import string

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from collections import Counter


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_and_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Clean text
    def clean_text(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    # Top 5 keywords in messages
    df["cleaned_message"] = df["message"].map(clean_text)
    w_list = " ".join(df["cleaned_message"]).split(" ")
    w_list = list(filter(lambda a: a != "", w_list))
    
    top40_kw = [Counter(w_list).most_common(i+1)[i][0] for i in range(40)]
    top40_kw_cnt = [Counter(w_list).most_common(i+1)[i][1] for i in range(40)]
    
    top5_kw = [w for w in top40_kw if w not in stopwords.words('english')][:5]
    top5_kw_cnt = [Counter(w_list)[w] for w in top5_kw]

    # Top 5 message categories 
    category_names = df.columns[4:-1]
    category_message_cnt = {}
    for cat in category_names:
        category_message_cnt[cat] = df[cat].sum()
       
    top5_category = [Counter(category_message_cnt).most_common(i+1)[i][0] for i in range(5)]
    top5_category_cnt = [Counter(category_message_cnt).most_common(i+1)[i][1] for i in range(5)]
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=top5_kw,
                    y=top5_kw_cnt
                )
            ],

            'layout': {
                'title': 'Top 5 popular keywords among disaster messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Keyword"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top5_category,
                    y=top5_category_cnt
                )
            ],

            'layout': {
                'title': 'Top 5 categories among disaster messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    
    # set "Child Alone"=0 (since the original dataset has no row with "Child Alone"=1)
    classification_labels = np.insert(classification_labels, 9, 0) 
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
