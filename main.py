import json
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Form
from transformers import Wav2Vec2Processor
import soundfile as sf
import librosa
import io

app = FastAPI()

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
MODEL_PATH = "models/onnx/model_quantized.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

@app.post("/analyse")
async def analyse_speech(file: UploadFile = File(...), transcript: str = Form(...)):
    transcript_data = json.loads(transcript)
    
    audio_bytes = await file.read()
    audio_data, samplerate = sf.read(io.BytesIO(audio_bytes))
    
    if samplerate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
    
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    inputs = processor(audio_data, sampling_rate=16000, return_tensors="np")
    input_values = inputs.input_values.astype(np.float32)

    onnx_inputs = {session.get_inputs()[0].name: input_values}
    logits = session.run(None, onnx_inputs)[0] 
    
    logits_max = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    word_details = []
    total_score = 0
    words_processed = 0

    for segment in transcript_data.get('segments', []):
        for word_info in segment.get('words', []):
            start_frame = max(0, int(word_info['start'] * 50) - 1)
            end_frame = int(word_info['end'] * 50) + 1
            
            word_probs = probs[0, start_frame:end_frame, :]
            
            if word_probs.shape[0] > 0:
                word_score = np.mean(np.max(word_probs, axis=-1))
            else:
                word_score = 0.0

            word_details.append({
                "word": word_info['word'],
                "word_score": float(round(word_score, 2)),
                "phonemes": [] 
            })
            total_score += word_score
            words_processed += 1

    overall_score = (total_score / words_processed) if words_processed > 0 else 0

    return {
        "overall_pronunciation_score": float(round(overall_score, 2)),
        "word_details": word_details
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)