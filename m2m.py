import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Paths for the pre-downloaded M2M100 model
M2M100_MODEL_PATH = "./models/facebook/m2m100_418M"

# Load M2M100 model and tokenizer
tokenizer = M2M100Tokenizer.from_pretrained(M2M100_MODEL_PATH)
model = M2M100ForConditionalGeneration.from_pretrained(M2M100_MODEL_PATH)

# Ensure torch uses the GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def translate_text(input_text, source_lang, target_lang):
    """
    Translate text from source language to target language using M2M100.
    Args:
        input_text (str): The input text to translate.
        source_lang (str): Source language code (e.g., 'en' for English).
        target_lang (str): Target language code (e.g., 'hi' for Hindi).
    Returns:
        str: Translated text.
    """
    # Set the tokenizer source language
    tokenizer.src_lang = source_lang

    # Tokenize the input text
    encoded_text = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate translated text
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded_text,
            forced_bos_token_id=tokenizer.get_lang_id(target_lang)  # Specify the target language
        )

    # Decode the generated tokens to text
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translated_text
