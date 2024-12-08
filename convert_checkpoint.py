from transformers import WhisperForConditionalGeneration

# Paths
checkpoint_path = "scratch/whisper-tiny-ATCO2-ASR/checkpoint-"
output_path = "scratch/whisper-tiny-ATCO2-ASR/final_model"

# Load the model from checkpoint
model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)

# Save the model in the standard format
model.save_pretrained(output_path)

# If you also want to save the processor/tokenizer (though it's usually unchanged from base model)
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="English", task="transcribe")
processor.save_pretrained(output_path)
