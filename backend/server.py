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
from keras.models import model_from_json
import sounddevice as sd
import soundfile as sf
import pickle
from pydub import AudioSegment
import pandas as pd


app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'your_secret_key'
BASE_DIR = Path(__file__).resolve().parent

model_path = 'models/model.pth'
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print(device)
p = pyaudio.PyAudio()
stream = None


# Loading and preparing CNN model
with open('models/CNN_model/CNN_model.json', 'r') as json_file:
    cnn_model_json = json_file.read()
    cnn_model = model_from_json(cnn_model_json)
    
cnn_model.load_weights('models/CNN_model/CNN_model_weights.h5')
print("loaded CNN model")

with open('models/CNN_model/encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

with open('models/CNN_model/scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)
    
print("wczytanie enkodera i scalara")

# Loading and preparing LSTM model
with open('models/LSTM_MODEL/LSTM_model.json', 'r') as json_file:
    lstm_model_json = json_file.read()
    lstm_model = model_from_json(lstm_model_json)
lstm_model.load_weights("models/LSTM_MODEL/LSTM_model_weights.h5")
print("Loaded LSTM Model")

# Loading and preparing MFCC model
with open('models/MFCC_MODEL/MFCC_MODEL.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    mfcc_model = model_from_json(loaded_model_json)
mfcc_model.load_weights("models/MFCC_MODEL/MFCC_MODEL.h5")
with open('models/MFCC_model/encoder2.pickle', 'rb') as f:
    encoder_mfcc = pickle.load(f)
print("Loaded model from disk")


# Funkcje CNN
def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features_CNN(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

def get_predict_feat_CNN(path):
    d, s_rate= librosa.load(path, duration=2.5, offset=0.6)
    res=extract_features_CNN(d)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,2376))
    i_result = scaler2.transform(result)
    final_result=np.expand_dims(i_result, axis=2)
    return final_result

def prediction_CNN(path1):
    res=get_predict_feat_CNN(path1)
    predictions=cnn_model.predict(res)
    y_pred = encoder2.inverse_transform(predictions)
    return y_pred

# Funkcje LSTM
def encode_LSTM(labels):
    emotion_dic = {
        0 : 'neutral' ,
        1 : 'happy'  ,
        2 : 'sad'    ,
        3 : 'angry'  ,
        4 : 'fear'   ,
        5 : 'disgust'
    }
    return [emotion_dic.get(label) for label in labels]

def preprocess_audio_LSTM(path):
    _, sr = librosa.load(path)
    raw_audio = AudioSegment.from_file(path)
    samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
    trimmed, _ = librosa.effects.trim(samples, top_db=25)
    padded = np.pad(trimmed, (0, 180000-len(trimmed)), 'constant')
    return padded, sr

def extract_features_LSTM(y, sr):
    zcr_list = []
    rms_list = []
    mfccs_list = []
    FRAME_LENGTH = 2048
    HOP_LENGTH = 512
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
    zcr_list.append(zcr)
    rms_list.append(rms)
    mfccs_list.append(mfccs)
    X = np.concatenate((
        np.swapaxes(zcr_list, 1, 2),
        np.swapaxes(rms_list, 1, 2),
        np.swapaxes(mfccs_list, 1, 2)),
        axis=2
    )
    X = X.astype('float32')
    return X

# Funkcje MFCC

def noise_mfcc(data):
    noise_amp = 0.04*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch_mfcc(data, rate=0.70):
    return librosa.effects.time_stretch(data,rate=rate)

def shift_mfcc(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch_mfcc(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def higher_speed_mfcc(data, speed_factor = 1.25):
    return librosa.effects.time_stretch(data,rate=speed_factor)

def lower_speed_mfcc(data, speed_factor = 0.75):
    return librosa.effects.time_stretch(data,rate=speed_factor)

def extract_features_mfcc(data):
    result = np.array([])
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    result = np.array(mfccs_processed)
    return result

def get_features_mfcc(path):
    data, sample_rate = librosa.load(path, duration=3, offset=0.5)
    res1 = extract_features_mfcc(data)
    result = np.array(res1)
    noise_data = noise_mfcc(data)
    res2 = extract_features_mfcc(noise_data)
    result = np.vstack((result, res2))
    stretch_data = stretch_mfcc(data)
    res3 = extract_features_mfcc(stretch_data)
    result = np.vstack((result, res3))
    shift_data = shift_mfcc(data)
    res4 = extract_features_mfcc(shift_data)
    result = np.vstack((result, res4))
    pitch_data = pitch_mfcc(data, sample_rate)
    res5 = extract_features_mfcc(pitch_data)
    result = np.vstack((result, res5))
    higher_speed_data = higher_speed_mfcc(data)
    res6 = extract_features_mfcc(higher_speed_data)
    result = np.vstack((result, res6))
    lower_speed_data = higher_speed_mfcc(data)
    res7 = extract_features_mfcc(lower_speed_data)
    result = np.vstack((result, res7))
    return result

def prediction_MFCC(audio_path):
    X = []
    features = get_features_mfcc(audio_path)
    for feature in features:
        X.append(feature)
    Features = pd.DataFrame(X)
    X = Features.iloc[:, :].values
    pred = np.expand_dims(X, axis=2)
    preds = mfcc_model.predict(pred)
    y_pred = encoder_mfcc.inverse_transform(preds)
    prediction = y_pred[0][0]
    return prediction

def detect_emotion_from_audio_transformer(audio):
    
    speech, _ = librosa.load(audio, sr=16000, mono=True)

    input_values = feature_extractor(speech, sampling_rate=16000, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**input_values)

    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()

    labels = {0: 'neutral', 1: 'happy', 2: 'angry', 3: 'sad', 4: 'fear', 5:'disgust'}
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
                    
                emotion_segment = detect_emotion_from_audio_transformer(temp_audio_path)
                emotion_results.append(emotion_segment)
                
                os.remove(temp_audio_path)
            
        except Exception as e:
            print(f"An error occurred while processing audio: {str(e)}")
            error = ("Invalid audio uploaded., Please upload a wav file")
            
    return jsonify({'error': error, 'emotion_results': emotion_results, 'audio_name': audio_name})


@app.route('/detect-emotion-live/model-1')   
def detect_emotion_live_Transformer():
    detected_emotion = "Unknown"
    global stream
    audio_data = np.frombuffer(stream.read(32000), dtype=np.float32)
    
    try:
        unique_name = f"temp_{uuid.uuid4()}.txt"
        audio_path = os.path.join(BASE_DIR, "audio", unique_name)
        with wave.open(audio_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(audio_data.tobytes())

        detected_emotion = detect_emotion_from_audio_transformer(audio_path)
        
        os.remove(audio_path)
        
    except Exception as e:
        print(f"An error occurred while processing audio: {str(e)}")
        
    return jsonify({'detected_emotion': detected_emotion})

@app.route('/detect-emotion-live/model-2')   
def detect_emotion_live_CNN(duration=3.2, samplerate=22050, channels=1, filename="temp_recording.wav"):
    detected_emotion = "Unknown"    
    try:
        myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
        sd.wait()
        sf.write(filename, myrecording, samplerate)
        detection = prediction_CNN(filename)
        detected_emotion = detection[0][0]
        os.remove(filename)
        
    except Exception as e:
        print(f"An error occurred while processing audio: {str(e)}")
        
    return jsonify({'detected_emotion': detected_emotion})

@app.route('/detect-emotion-live/model-3')   
def detect_emotion_live_LSTM(duration=2.7, samplerate=22050, channels=1, filename="temp_recording.wav"):
    detected_emotion = "Unknown"    
    try:
        myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
        sd.wait()
        sf.write(filename, myrecording, samplerate)
        y, sr = preprocess_audio_LSTM(filename)
        X = extract_features_LSTM(y, sr)
        y_pred = np.argmax(lstm_model.predict(X), axis=1)
        emotion_predicted = encode_LSTM(y_pred)
        detected_emotion = emotion_predicted[0]
        os.remove(filename)
        
    except Exception as e:
        print(f"An error occurred while processing audio: {str(e)}")
        
    return jsonify({'detected_emotion': detected_emotion})

@app.route('/detect-emotion-live/model-4')
def detect_emotion_live_MFCC(duration=3.2, samplerate=22050, channels=1, filename="temp_recording.wav"):
    myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
    sd.wait()
    sf.write(filename, myrecording, samplerate)
    detected_emotion = prediction_MFCC(filename)
    os.remove(filename)
    return jsonify({'detected_emotion': detected_emotion})

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
    