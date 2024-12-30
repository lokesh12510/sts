from transformers import pipeline


def initialize_stt_pipeline(model_name="facebook/wav2vec2-base-960h"):
    
    """
    Initializes the speech-to-text pipeline using a Hugging Face model.

    Args:
        model_name (str): The name of the Hugging Face model from the model hub.
                          Default is "facebook/wav2vec2-base-960h".

    Returns:
        pipeline: A Hugging Face pipeline for speech recognition.
    """
    return pipeline(task="automatic-speech-recognition", model=model_name)

if __name__ == "__main__":
     stt = initialize_stt_pipeline()
     audio_file = "./input_tts.mp3"  # Replace with your audio file path
    
     try:
        result = stt(audio_file)
        print(result["text"])
     except Exception as e:
        print(f"Error during speech-to-text conversion: {e}")