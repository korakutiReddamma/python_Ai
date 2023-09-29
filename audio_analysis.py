import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydub import AudioSegment
from sklearn.ensemble import RandomForestClassifier

# Load audio file
def load_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    return audio

# Extract audio features (MFCCs in this example)
def extract_features(audio):
    samples = np.array(audio.get_array_of_samples())
    mfccs = librosa.feature.mfcc(samples, audio.frame_rate)
    return mfccs

# Emotion detection model (dummy model for illustration)
def train_emotion_model(features, labels):
    model = RandomForestClassifier()
    model.fit(features, labels)
    return model

# Keyword detection (dummy function for illustration)
def keyword_detection(audio):
    # Replace with your keyword detection logic
    return "Keyword Detected" if "your_keyword" in audio else "No Keyword"

# Main function
def main():
    audio_dir = r'E:\Bipolar'  # Replace with the actual path
    emotion_labels = ['happy', 'sad']  # Emotion labels
    keyword_results = []

    # Create data frames for results
    emotion_results = pd.DataFrame(columns=['Audio File', 'Emotion'])
    keyword_results = pd.DataFrame(columns=['Audio File', 'Keyword'])

    # Iterate through audio files
    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith('.wav'):
            file_path = os.path.join(audio_dir, audio_file)

            # Load and process audio
            audio = load_audio(file_path)
            features = extract_features(audio)

            # Emotion detection
            emotion_model = train_emotion_model(features, [0])  # Dummy label for illustration
            emotion_prediction = emotion_model.predict(features)[0]
            emotion_results = emotion_results.append({'Audio File': audio_file, 'Emotion': emotion_labels[emotion_prediction]}, ignore_index=True)

            # Keyword detection
            keyword_result = keyword_detection(audio)
            keyword_results = keyword_results.append({'Audio File': audio_file, 'Keyword': keyword_result}, ignore_index=True)

    # Print emotion results
    print("Emotion Detection Results:")
    print(emotion_results)

    # Plot emotion predictions
    plot_emotion_predictions(emotion_results['Emotion'].apply(lambda x: emotion_labels.index(x)))

    # Print keyword detection results
    print("\nKeyword Detection Results:")
    print(keyword_results)

# Plot emotion predictions as a line chart
def plot_emotion_predictions(predictions):
    plt.plot(predictions)
    plt.xlabel('Audio File')
    plt.ylabel('Emotion (0: happy, 1: sad)')
    plt.title('Emotion Detection Results')
    plt.show()

if __name__ == "__main__":
    main()
