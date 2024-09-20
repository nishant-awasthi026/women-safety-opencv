import pyaudio
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import speech_recognition as sr
import soundfile as sf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from alert_system import AlertSystem

def record_audio(duration=5, sample_rate=16000):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1

    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk)
    
    print("Recording...")
    frames = []

    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Convert to a floating-point format
    audio_signal = np.concatenate(frames).astype(np.float32)
    audio_signal = audio_signal / np.max(np.abs(audio_signal))  # Normalize to [-1, 1]

    print("Audio data:", audio_signal)  # Print audio data for debugging

    return audio_signal

def extract_features(audio_signal, sample_rate=16000):
    mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def train_model():
    X = np.array([
        np.random.rand(13) for _ in range(100)
    ])
    y = np.random.randint(0, 2, size=(100,))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier()
    model.fit(X_scaled, y)
    
    if not os.path.exists('model'):
        os.makedirs('model')
    
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model trained and saved.")

def detect_distress(audio_signal, model, scaler, sample_rate=16000):
    features = extract_features(audio_signal, sample_rate)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return prediction[0] == 1

def alert_if_distress(audio_signal, model, scaler):
    if detect_distress(audio_signal, model, scaler):
        print("Distress detected! Triggering alert...")
    else:
        print("No distress detected.")

def transcribe_audio(audio_signal, sample_rate=16000):
    recognizer = sr.Recognizer()
    
    # Save audio to a file for speech_recognition
    audio_file = "temp_audio.wav"
    sf.write(audio_file, audio_signal, sample_rate)  # Use soundfile to write the audio file
    
    transcription = ""
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
            print("Transcription:", transcription)
    except sr.UnknownValueError:
        transcription = "Google Speech Recognition could not understand the audio"
        print(transcription)
    except sr.RequestError as e:
        transcription = f"Could not request results from Google Speech Recognition service; {e}"
        print(transcription)
    
    # Remove the temporary audio file
    os.remove(audio_file)

    return transcription

def analyze_speech(duration=5):
    # Load model and scaler
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Create an instance of the AlertSystem
    alert_system = AlertSystem(recipient_phone="+916307257097")  # Replace with the actual recipient phone number
    
    # Record audio and process
    audio_signal = record_audio(duration)
    transcription = transcribe_audio(audio_signal)  # Ensure this function returns a string
    distress_detected = detect_distress(audio_signal, model, scaler)  # Boolean result

    # Send an alert if distress is detected
    if distress_detected:
        message = f"Distress detected during speech analysis. \n\nTranscription: {transcription}"
        alert_system.send_alert(message)

    return {
        'transcription': transcription,
        'distress_detected': distress_detected
    }
