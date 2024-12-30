from TTS.api import TTS


def text_to_speech(text, output_path, language="hi", speaker_wav="./output_tts.wav"):
    # Initialize the TTS model (you can provide a local model path here if needed)
    # tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    # tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    # tts = TTS(model_name="tts_models/en/ljspeech/vits")

    # Synthesize speech and save to an audio file
    tts.tts_to_file(text, file_path=output_path,language=language ,speaker_wav=speaker_wav )

    print(f"Audio saved as {output_path}")

# # Example usage
# text_to_speech("नमस्ते, यह ऑफ़लाइन टीटीएस के लिए एक परीक्षण संदेश है।", "hindi.wav")
