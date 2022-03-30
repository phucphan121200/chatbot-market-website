import random
import json
import speech_recognition as sr
import pyttsx3
import torch
from gtts import gTTS
import os
from playsound import playsound
import datetime
import time

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

#Voice Speech
def speak(text):
    tts = gTTS(text=text, lang='vi', slow=False)
    date_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    filename = "voice"+date_string+".mp3"
    tts.save(filename)
    playsound(filename) 
    os.remove(filename)

#main
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "Emperor"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    # print("You:" )
    sentence = input("You: ")
    if sentence == "quit":
        break
    #Answer 
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).to(device)
    X = X.reshape(1, X.shape[0])
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    # engine = pyttsx3.init()
    # voices = engine.getProperty('voices') 
    # engine.setProperty('voice', voices[1].id) #Sử dụng âm thanh nữ
    if prob.item() > 0.85:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                answer = random.choice(intent['responses'])
                print(f"{bot_name}: {answer}")
                speak(answer)
    else:
        print(f"{bot_name}: I do not understand...")
        speak("I do not understand...")