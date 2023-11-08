from flask_cors import CORS
from flask import Flask, request, jsonify
import torch
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import pyaudio
import numpy as np
import wave
import os
import librosa
from werkzeug.utils import secure_filename
import uuid
from pathlib import Path

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'your_secret_key'
BASE_DIR = Path(__file__).resolve().parent

model_path = 'model.pth'
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
p = pyaudio.PyAudio()
stream = None


def detect_emotion_from_audio(audio):
    
    speech, _ = librosa.load(audio, sr=16000, mono=True)

    input_values = feature_extractor(speech, sampling_rate=16000, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**input_values)

    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()

    labels = {0: 'neutral', 1: 'happy', 2: 'angry', 3: 'sad', 4: 'fear'}
    detected_emotion = labels.get(predicted_label, 'Unknown')

    return detected_emotion


@app.route('/process-single-audio', methods=['POST'])
def process_single_audio():
    emotion_results = []
    error = None
    print("Started Processing Single Audio")
    if request.method == 'POST' and request.files.get("audio"):
        uploaded_audio = request.files["audio"]
        audio_name = secure_filename(uploaded_audio.filename)
        audio_bytes = uploaded_audio.read()
        
        try:
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
            
            samples_per_segment = (1 * pyaudio.PyAudio().get_sample_size(pyaudio.paInt16)*8000)
            
                
            for i in range(0, len(audio), samples_per_segment):
                segment = audio[i:i+samples_per_segment]
                
                if not os.path.exists(os.path.join('audio')):
                    os.makedirs(os.path.join('audio'))
                        
                temp_audio_path = os.path.join(BASE_DIR, "audio", f'temp_segment_{i}.wav')
                
            
                with wave.open(temp_audio_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                    wf.setframerate(16000)
                    wf.writeframes(segment.tobytes())
                    
                emotion_segment = detect_emotion_from_audio(temp_audio_path)
                emotion_results.append(emotion_segment)
                
                os.remove(temp_audio_path)
            
        except Exception as e:
            print(f"An error occurred while processing audio: {str(e)}")
            error = ("Invalid audio uploaded., Please upload a wav file")
            
    return jsonify({'error': error, 'emotion_results': emotion_results, 'audio_name': audio_name})


@app.route('/detect-emotion-live/model-1')   
def detect_emotion_live():
    detected_emotion = "Unknown"
    global stream
    audio_data = np.frombuffer(stream.read(32000), dtype=np.float32)
    image_data = {
        "src": "/neutral.png",
        "alt": "Neutral Icon Image"
    }
    try:
        unique_name = f"temp_{uuid.uuid4()}.txt"
        audio_path = os.path.join(BASE_DIR, "audio", unique_name)
        with wave.open(audio_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(audio_data.tobytes())

        detected_emotion = detect_emotion_from_audio(audio_path)
        image_data={
            "src": f"/detected_{detected_emotion}.png",
            "alt": f"{detected_emotion} Icon Image"
        }
        
        os.remove(audio_path)
        
    except Exception as e:
        print(f"An error occurred while processing audio: {str(e)}")
        
    return jsonify({'emotionImage': image_data,  'detected_emotion': detected_emotion})
                
@app.route('/start-recording')
def start_recording():
    print("listening")
    global stream
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=32000)     
        
    return jsonify({'text': 'Started Listening'})

@app.route('/stop-recording')
def stop_recording():
    print("Not listening")
    global stream
    if stream:
        stream.stop_stream()
        stream = None
        print("Recording stopped")
    else:
        print("No active audio stream to stop")        
        
    return jsonify({'text': 'Stopped Listening'})


if __name__ == '__main__':
    app.run(debug=True, port=8080)
    