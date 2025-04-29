import whisper
import google.generativeai as genai
import os 
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()


def generate_content(text):
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(text)
    return response.text

def stt(audio):
    if audio is None:
        return ""
    
    if not os.path.exists(audio):
        print(f"Audio file {audio} not found")
        return ""

    try:
        # audio = whisper.load_audio(audio)
        model = whisper.load_model("base")
        result = model.transcribe(audio,language="en")
        with open('transcript.txt', 'w') as f:
            f.write(result["text"])
        return result["text"] 
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

def tts(text):
    model = pipeline(task="text-to-speech", model="suno/bark")
    return (np.array(model(text)['audio'])*32767).astype(np.int16).T


def main():  
    result = stt("jfk.mp3")
    print(result)
    
    response = generate_content(result)
    print(response)

    tts(response)

if __name__ == "__main__":
    main()
