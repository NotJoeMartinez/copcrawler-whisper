import torch
import json
import argparse

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from faster_whisper import WhisperModel
from multiprocessing import cpu_count

# Use a pipeline as a high-level helper
# from transformers import pipeline
# pipe = pipeline("automatic-speech-recognition", model="youngsangroh/whisper-small-finetuned-atco2-asr-atcosim")

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        required=True
    )
    parser.add_argument(
        '--input',
        required=True
    )
    args = parser.parse_args()

    model = args.model
    audio_path = args.input

    if model == 'tiny.en':
        run_tiny_model(model, audio_path)
    elif model == 'tiny':
        run_tiny_model(model, audio_path)

    else:
        run_fine_tuned_model(args)

def run_tiny_model(model, file_path):
    model = WhisperModel(
                        model, 
                        device="cpu", 
                        compute_type="int8", 
                        num_workers=8,
                        cpu_threads=cpu_count())

    segments, info = model.transcribe(file_path, 
                     beam_size=5,
                     condition_on_previous_text=False,
                     language="en",
                     no_speech_threshold=1.5, 
                     repetition_penalty=1.7)
    
    output = ""

    for segment in segments:
        output += segment.text + " "
    
    print(output)
    
def run_fine_tuned_model(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = args.model

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        'automatic-speech-recognition',
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device
    )

    result = pipe(args.input, generate_kwargs={'task': 'transcribe'})
    print(result)
    text = result['text']
    print(text)



if __name__ == '__main__':
    main()