import torch
import argparse

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from faster_whisper import WhisperModel
from multiprocessing import cpu_count

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

    if args.model == 'tiny':
        run_tiny_model(args.input)


def run_tiny_model(file_path):
    model = WhisperModel('tiny', 
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
    text = result['text']
    print(text)



if __name__ == '__main__':
    main()