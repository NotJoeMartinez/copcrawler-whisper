from faster_whisper import WhisperModel

model_path = "data/models/whisper_tiny_atco2_v2_ct2"
model = WhisperModel(model_path, device="cpu")  # or "cuda"

audio_file = "data/target_test_audio/202412050011-39647-34171.mp3"
segments, info = model.transcribe(audio_file, beam_size=5)

for segment in segments:
    print(segment.start, segment.end, segment.text)
