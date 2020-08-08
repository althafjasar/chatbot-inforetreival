# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 08:59:08 2020

@author: Jasar Althaf
"""
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

def response(msg):
        english_bot = ChatBot("Chatterbot",storage_adapter="chatterbot.storage.SQLStorageAdapter")
        trainer = ChatterBotCorpusTrainer(english_bot)
        trainer.train("chatterbot.corpus.english")
        trainer.train("chatbotdata/greetings.yml")
        return english_bot.get_response(msg)
