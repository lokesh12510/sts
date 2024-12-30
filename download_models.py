# from transformers import WhisperForConditionalGeneration, WhisperProcessor

# # Load the model and processor
# model_name = "openai/whisper-large"
# processor = WhisperProcessor.from_pretrained(model_name)
# model = WhisperForConditionalGeneration.from_pretrained(model_name)

# # Save the model locally
# model.save_pretrained("models/whisper")
# processor.save_pretrained("models/whisper")




# from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# # Load the M2M-100 model and tokenizer
# model_name = "facebook/m2m100_418M"
# model = M2M100ForConditionalGeneration.from_pretrained(model_name)
# tokenizer = M2M100Tokenizer.from_pretrained(model_name)

# # Save the model locally
# model.save_pretrained("models/facebook/m2m100_418M")
# tokenizer.save_pretrained("models/facebook/m2m100_418M")


# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# # Load the Wav2Vec2 model and processor
# model_name = "facebook/wav2vec2-base-960h"
# model = Wav2Vec2ForCTC.from_pretrained(model_name)
# processor = Wav2Vec2Processor.from_pretrained(model_name)


# # Save the model locally
# model.save_pretrained("models/facebook/wav2vec2-base-960h")
# processor.save_pretrained("models/facebook/wav2vec2-base-960h")



import os

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer
from TTS.api import TTS

# device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "./models/your_tts"

# Create the directory if it doesn't exist
os.makedirs(model_path, exist_ok=True)

tts = TTS(
     model_name="tts_models/multilingual/multi-dataset/your_tts", 
     model_path=model_path,
     ).to('cpu')


# from transformers import WhisperForConditionalGeneration, WhisperProcessor

# # Define the local path where the model will be saved
# LOCAL_MODEL_DIR = "models/whisper_tiny"

# print("Downloading Whisper-tiny model...")
# processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# # Save the model and processor locally
# processor.save_pretrained(LOCAL_MODEL_DIR)
# model.save_pretrained(LOCAL_MODEL_DIR)
# print(f"Model downloaded and saved to {LOCAL_MODEL_DIR}")

