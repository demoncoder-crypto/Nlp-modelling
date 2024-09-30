# app.py

from flask import Flask, render_template, request
from transformers import pipeline
from bertopic import BERTopic
from gensim import corpora
from gensim.models import LdaModel
import nltk
import spacy
from nltk.corpus import stopwords
import re

# Initialize Flask app
app = Flask(__name__)

# Initialize summarization pipeline
summarizer = pipeline("summarization")

# Initialize SpaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize BERTopic model
topic_model = BERTopic()

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Function to preprocess text for LDA
def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize and remove stopwords
    tokens = [token.lower() for token in nltk.word_tokenize(text) if token.lower() not in stop_words]
    return tokens

# Route for home page
@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    lda_topics = []
    bert_topics = []
    if request.method == 'POST':
        input_text = request.form['input_text']
        
        if input_text.strip() == "":
            summary = "Please enter some text to summarize and analyze."
            return render_template('index.html', summary=summary, lda_topics=lda_topics, bert_topics=bert_topics)
        
        # Text Summarization
        try:
            summary_result = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
            summary = summary_result[0]['summary_text']
        except Exception as e:
            summary = f"Error in summarization: {str(e)}"
        
        # Topic Modeling with LDA
        try:
            tokens = preprocess_text(input_text)
            dictionary = corpora.Dictionary([tokens])
            corpus = [dictionary.doc2bow(tokens)]
            lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
            lda_topics = lda_model.print_topics(num_words=5)
        except Exception as e:
            lda_topics = [f"Error in LDA: {str(e)}"]
        
        # Topic Modeling with BERTopic
        try:
            bert_topics, probs = topic_model.fit_transform([input_text])
            bert_topics = topic_model.get_topic_info()
        except Exception as e:
            bert_topics = [f"Error in BERTopic: {str(e)}"]
        
    return render_template('index.html', summary=summary, lda_topics=lda_topics, bert_topics=bert_topics)

if __name__ == '__main__':
    app.run(debug=True)
