# Bibliotecas de pré-processamento de dados de texto
import nltk

import json
import pickle
import numpy as np
import random

# Biblioteca load_model
import tensorflow
from data_preprocessing import get_stem_words

#Carregue o modelo
model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Carregue os arquivos de dados
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))

ignore_words = ['?', '!',',','.', "'s", "'m"]
