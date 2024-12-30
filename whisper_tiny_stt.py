import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Ensure torch uses the GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths for the pre-downloaded Whisper-tiny model
WHISPER_MODEL_PATH = "./models/whisper_tiny"

# Load Whisper-tiny model and processor
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_PATH)
model.to(device)

def load_and_resample_audio(audio_path, target_sample_rate=16000):
    """
    Load audio file and resample to the target sample rate.
    Args:
        audio_path (str): Path to the audio file.
        target_sample_rate (int): Desired sample rate.
    Returns:
        torch.Tensor: Resampled audio waveform.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform

def speech_to_text(audio_path):
    """
    Perform Speech-to-Text (STT) on the given audio file using the Whisper-tiny model.
    Args:
        audio_path (str): Path to the input audio file.
    Returns:
        str: Transcribed text.
    """
    # Load and resample audio
    waveform = load_and_resample_audio(audio_path)

    # Extract input features
    input_features = processor.feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features


    # Add attention mask for reliable results
    attention_mask = torch.ones_like(input_features, dtype=torch.bool).to(device)
    input_features = input_features.to(device)

    # Perform inference
    with torch.no_grad():
        generated_ids = model.generate(input_features, attention_mask=attention_mask, language='en')

    # Decode the output to text
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription