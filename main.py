# -*- coding: utf-8 -*-
import scipy
import torch
from transformers import AutoTokenizer, VitsModel

from m2m import translate_text
from whisper_tiny_stt import speech_to_text
from yourtts import text_to_speech


def main():
    # Input audio file for STT
    audio_file = "./input_tts.mp3"  # Replace with your audio file path

    print("Performing Speech-to-Text (STT)...")
    # Convert speech to text
    transcription = speech_to_text(audio_file)
    print(f"Transcription: {transcription}")

    # Define source and target languages for TT
    source_language = "en"  # English (adjust based on your STT language setting)
    target_language = "hi"  # Hindi

    print("Performing Text-to-Text Translation (TT)...")
    # Translate the transcribed text
    translated_text = translate_text(transcription, source_language, target_language)
    print(f"Translated Text: {translated_text}")

    print("Performing Text-to-Speech (TTS)...")
    
    text_to_speech(translated_text, "hindiOutput.wav", language=target_language, speaker_wav="./output_tts.wav")

    print("Process completed successfully!")


def tamil_speech():
    model = VitsModel.from_pretrained("facebook/mms-tts-tam")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tam")
    print("Model loaded successfully!")
    
    text = "இலக்கியமும் வரலாறும் வாழ்வியலோடு தொடர்புடையவை. அவை மறைக்கப்படவில்லை. புதைக்கப்பட்டிருக்கின்றன. ஆனால், தற்போது பல தடங்களின் வழியாக அவை வெளியேறி மீண்டும் செய்தியாக நம் தலைமுறை யினரிடம் வந்து சேர்ந்துகொண்டிருக்கின்றன. காரணம் மனிதனின் மரபு, உணர்வு, வீரம், பழக்க வழக்கம், செயல்பாடு, சிந்தனை என இவற்றோடு பின்னிப்பிணைந்திருப்பதால். பாட்டி சொல்லும் கதை வழி, ஒரு குழந்தை ஒரு செய்தியை அறிந்துகொள்கிறது. அதுபோல் இன்று பத்திரிகை, தொலைக்காட்சி, சினிமா, செல்போன் என பல ஊடகங்கள் வழியாக அறிந்துகொள்கிறோம். புதைந்திருக்கும் வரலாற்றுக் கதைகளும். கதைகளாக சொல்லப்படும் வரலாற்று உண்மைகளும் இன்று மக்களிடையே தாக்கத்தை ஏற்படுத்துகின்றன. அது ஓர் எழுத்தாளனின் எழுதுகோல் வழியே எழுந்த தமிழர்களின் மரபே இந்த நூல். பரங்கியர் படையை நடுங்கச்செய்த தென் தமிழகத்து போர் ஆயுதங்களின் குறியீடான வளரி முதல் மரணத்தொழில் செய்யும் போக்கிரிகள், மோசடி செய்யும் கும்பல்களின் அட்டகாசங்கள், கீழடி செய்திகள், மாடோட்டிகளின் மரபு விளக்கங்கள், கல்வெட்டுச் செய்திகள் இலக்கியம், வரலாறு, கணக்கு."
    inputs = tokenizer(text, return_tensors="pt")
    print("Tokenization completed successfully!")
    
    with torch.no_grad():
        output = model(**inputs).waveform
        print(f"Output waveform shape: {output.shape}")
        print(f"Sampling rate: {model.config.sampling_rate}")
        
        # Ensure the sampling rate is a valid integer
        sampling_rate = int(model.config.sampling_rate)
        
        # Ensure the waveform data is in the correct format
        if output.ndim == 2 and output.shape[0] == 1:
            output = output.squeeze(0)
        
        scipy.io.wavfile.write("techno.wav", rate=sampling_rate, data=output.numpy())
        print("Text-to-Speech completed successfully!")

if __name__ == "__main__":
    tamil_speech()

