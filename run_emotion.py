import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array
import os
import warnings
from transformers import logging

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Define the emotion labels for both models
audio_emotion_labels = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised", "Calm"]
image_emotion_labels = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

# Audio Model
class Wav2Vec2WithAttention(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(Wav2Vec2WithAttention, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        self.attention = nn.MultiheadAttention(embed_dim=self.wav2vec2.config.hidden_size, num_heads=4, batch_first=True)
        self.fc = nn.Linear(self.wav2vec2.config.hidden_size, num_classes)
    
    def forward(self, input_values, attention_mask=None):
        hidden_states = self.wav2vec2(input_values, attention_mask=attention_mask).last_hidden_state
        attention_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        pooled_output = torch.mean(attention_output, dim=1)
        logits = self.fc(pooled_output)
        return logits

def preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample(waveform)
    return waveform.squeeze().numpy()

def predict_emotion_audio(audio_path):
    input_audio = preprocess_audio(audio_path)
    inputs = processor(input_audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask)
        probabilities = nn.Softmax(dim=-1)(logits)
    return probabilities.cpu().numpy().flatten()

# Initialize audio model
PRETRAINED_MODEL_NAME = "facebook/wav2vec2-base"
NUM_CLASSES_AUDIO = 8
processor = Wav2Vec2Processor.from_pretrained(PRETRAINED_MODEL_NAME)
model = Wav2Vec2WithAttention(PRETRAINED_MODEL_NAME, NUM_CLASSES_AUDIO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Image Model
def load_emotion_image_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.load_weights('models/emotion_model.h5')
    return model

emotion_model = load_emotion_image_model()

def predict_emotion_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (48, 48))
    gray_image = gray_image.astype('float32') / 255.0
    gray_image = np.expand_dims(gray_image, -1)
    gray_image = np.expand_dims(gray_image, 0)
    probabilities = emotion_model.predict(gray_image)[0]
    return probabilities

def map_audio_to_image(audio_probs):
    audio_probs_mapped = audio_probs.copy()
    audio_probs_mapped[4] += audio_probs[7]
    audio_probs_mapped = np.delete(audio_probs_mapped, 7)
    return audio_probs_mapped

def predict_multimodal_emotion(audio_path, image_path):
    audio_probs = predict_emotion_audio(audio_path)
    image_probs = predict_emotion_image(image_path)
    audio_probs_mapped = map_audio_to_image(audio_probs)
    fusion_probs = 0.5 * audio_probs_mapped + 0.5 * image_probs
    final_prediction = np.argmax(fusion_probs)
    return image_emotion_labels[final_prediction]

# GUI Implementation
def browse_audio_file():
    audio_file_path.set(filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")]))

def browse_image_file():
    image_file_path.set(filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]))

def predict_emotion():
    audio_path = audio_file_path.get()
    image_path = image_file_path.get()
    if not audio_path or not image_path:
        messagebox.showerror("Error", "Please select both audio and image files.")
        return
    try:
        emotion = predict_multimodal_emotion(audio_path, image_path)
        result_label.config(text=f"Predicted Emotion: {emotion}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create the GUI
root = tk.Tk()
root.title("Multimodal Emotion Recognition")

# Audio File
audio_file_path = tk.StringVar()
tk.Label(root, text="Audio File:").grid(row=0, column=0, sticky="e")
audio_entry = tk.Entry(root, textvariable=audio_file_path, width=40)
audio_entry.grid(row=0, column=1)
tk.Button(root, text="Browse", command=browse_audio_file).grid(row=0, column=2)

# Image File
image_file_path = tk.StringVar()
tk.Label(root, text="Image File:").grid(row=1, column=0, sticky="e")
image_entry = tk.Entry(root, textvariable=image_file_path, width=40)
image_entry.grid(row=1, column=1)
tk.Button(root, text="Browse", command=browse_image_file).grid(row=1, column=2)

# Predict Button
tk.Button(root, text="Predict Emotion", command=predict_emotion, bg="green", fg="white").grid(row=2, column=0, columnspan=3, pady=10)

# Result Label
result_label = tk.Label(root, text="Predicted Emotion: ", font=("Arial", 14))
result_label.grid(row=3, column=0, columnspan=3, pady=20)

root.mainloop()
