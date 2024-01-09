import joblib 
import gradio as gr
import warnings
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd 
import regex as re
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordenet")
data  = pd.read_csv("Dataa.csv",encoding="utf-8")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
import spacy
import en_core_web_sm
nlp = spacy.load("en_core_web_sm")

warnings.filterwarnings("ignore")

def clean_utterance(utterance):
    clean_utterance = re.sub(r'[^a-zA-z0-9\s]','',utterance)
    return clean_utterance

def preprocess_utterance(utterance):
    tokens = word_tokenize(utterance.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if
    token.isalpha() and token not in stop_words]
    return' '.join(filtered_tokens)


def Predicttalkk(utterance):
    try:
        st_model = "smallTalkk.pkl"
        st_vector = "SmallTalkkvector.pkl"
        loaded_model = joblib.load(st_model)
        small_talk_vectorizer = joblib.load(st_vector)
        new_utterances = [utterance]
        new_utterance_tfidf = small_talk_vectorizer.transform(new_utterances)
        prediction_probabilities =  loaded_model.predict_proba(new_utterance_tfidf)
        prediction = loaded_model.predict(new_utterance_tfidf)
        max_confidence = -1 
        predicted_class = None 
        
        for i in range(len(prediction)):
            for j,class_prob in enumerate(prediction_probabilities[i]):
                class_name = loaded_model.classes_[j]
                confidence = class_prob * 100
                if confidence > max_confidence:
                    max_confidence = confidence
                    predicted_class = class_name 
                    
        print("small talk output:",{"response":predicted_class,"confidence":max_confidence})
        return{"response":predicted_class,"confidence":max_confidence}
    except:
        print("error with Predicttalkk")
        return "error with predicttalk"

def answer(utterance):
    prediction = Predicttalkk(utterance)
    if prediction['response'] == "It's Narendra Modi" and prediction['confidence'] > 75:
        doc = nlp(utterance)
        country = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        try:
            if country[0]== "India":
                return prediction["response"]
            else:
                msg = "My knowledge is only limited to India"
                return msg
        except:
            msg = "My knowledge is only limited to India"
            return msg      
              
    elif prediction['confidence'] > 75:
        return prediction['response']
    else:
        msg = "Sorry I don't have answer to the question..!! :("
        return msg
  
# while True:
#     print(answer(input("user: ")))

iface = gr.Interface(fn=answer, inputs="text", outputs="text")
iface.launch(debug=True)
