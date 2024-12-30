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

if __name__ == "__main__":
    main()
