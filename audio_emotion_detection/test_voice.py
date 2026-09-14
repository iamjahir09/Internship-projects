from src.voice_emotion import predict_emotion

audio_file = r"C:/Users/pc/Desktop/Emotion_detector/audio_emotion_detection/OAF_bar_happy.wav"

emotion, confidence = predict_emotion(audio_file)

print("Predicted Emotion:", emotion)
print("Confidence:", f"{confidence * 100:.2f}%")