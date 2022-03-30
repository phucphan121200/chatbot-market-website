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

#Listening Voice
def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:

        print("Tôi: ", end='')
        audio = r.listen(source, timeout=2)
        audio = r.record(source, duration=5)
        try:
            text = r.recognize_google(audio, language="vi-VN")
            print(text)
            return text
        except:
            print("...")
            return 0

def stop():
    speak("Tạm biệt")
    
def get_text():
    for i in range(3):
        text = get_audio()
        if text:
            return text.lower()
        elif i < 2:
            speak("Emperor không nghe rõ. Bạn nói lại được không!")
    time.sleep(3)
    stop()
    return 0

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
    # bot-ear: Listen human voice
    robotear = sr.Recognizer()
    with sr.Microphone() as mic:
        print("Toi dang nghe day")
        speak("Tôi đang nghe đây")
        audio = robotear.listen(mic, timeout=2, phrase_time_limit = 5)
        # audio = robotear.record(mic, duration=500)
    try:
        sentence = robotear.recognize_google(audio, language="vi-VN")
        print("Me: " + sentence)
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
            break
        break
    except:
        sentence = ""
    
    # print(you)
    # sentence = input("You: ")
    if sentence == "quit":
        break