from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Load the processor and model from checkpoint
checkpoint_path = "scratch/whisper-tiny-ATCO2-ASR/checkpoint-400"

# Load processor from the original model since checkpoints don't save it
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="English", task="transcribe")

# Load the fine-tuned model from checkpoint
model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)

# Prepare model for inference
model.eval()  # Set to evaluation mode

# Example inference function
def transcribe_audio(audio_path):
    """
    Transcribe audio using the loaded checkpoint
    
    Args:
        audio_path: Path to audio file
    Returns:
        Transcribed text
    """
    # Load and process audio
    audio_input = processor(
        audio_path, 
        sampling_rate=16000, 
        return_tensors="pt"
    )
    
    # Generate transcription
    generated_ids = model.generate(
        input_features=audio_input.input_features,
        max_length=448,
        # Add any other generation parameters as needed
    )
    
    # Decode the generated tokens to text
    transcription = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )[0]
    
    return transcription

transcription = transcribe_audio("data/target_test_audio/audio.wav")
print(transcription)