import speech_recognition

robotear = speech_recognition.Recognizer()
with speech_recognition.Microphone() as mic:
    print("Toi dang nghe day")
    audio = robotear.listen(mic)
try:
    you = robotear.recognize_google(audio)
except:
    you = ""

print(you)