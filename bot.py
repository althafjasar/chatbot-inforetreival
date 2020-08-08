from flask import Flask, render_template, request
from chatterbot import ChatBot
import pickle
from chatterbot.trainers import ChatterBotCorpusTrainer
from functions import inforetrival
import pandas as pd


app = Flask(__name__)
#create chatbot
englishBot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
trainer = ChatterBotCorpusTrainer(englishBot)
trainer.train("chatterbot.corpus.english") #train the chatter bot for english
trainer.train("chatbotdata/greetings.yml")
with open("C:\\Users\\Jasar Althaf\\Desktop\\work\\bot\\_tfidf_features.pkl",'rb') as fv:
        feature_vector = pickle.load(fv)
with open('C:\\Users\\Jasar Althaf\\Desktop\\work\\bot\\_rf_model.pkl', 'rb') as m:
        model = pickle.load(m)
#define app routes
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/get")
#function for the bot response
def get_bot_response():
    userText =request.args.get('msg')
    df=pd.read_csv('C:/Users/Jasar Althaf/Desktop/work/bot/localhost/itfaq.csv', encoding= 'unicode_escape')
    text=[userText]
    #return userText
    inp= feature_vector.transform(text)
    cat=model.predict(inp)
    if cat==0:
         return str(englishBot.get_response(userText))
    else:
       val1 = inforetrival(df,userText)
       return val1
    
if __name__ == "__main__":
     app.run(debug = True)


