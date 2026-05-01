import json
import os
import shutil
import io
import base64
import numpy as np
import onnxruntime as ort
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from transformers import Wav2Vec2Processor

app = FastAPI()

# --- CONFIGURATION & MODEL LOADING ---
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
MODEL_PATH = "models/onnx/model_quantized.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

@app.get("/health")
async def health_check():
    """Connectivity check for Spring Boot."""
    return {"status": "healthy", "service": "pronunciation-analysis"}

@app.post("/analyse")
async def analyse(file: UploadFile = File(...), transcript: str = Form(...)):
    try:
        transcript_data = json.loads(transcript)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in transcript field.")
    
    allowed_extensions = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Allowed: {allowed_extensions}")

    temp_filename = f"temp_analysis_{file.filename}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        sr = 16000
        audio_data, _ = librosa.load(temp_filename, sr=sr, mono=True)
        
        inputs = processor(audio_data, sampling_rate=sr, return_tensors="np")
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
                    predicted_ids = np.argmax(word_probs, axis=-1)
                    active_speech_frames = word_probs[predicted_ids != 0]

                    if active_speech_frames.shape[0] > 0:
                        word_score = np.mean(np.max(active_speech_frames, axis=-1))
                    else:
                        word_score = 0.0 
                else:
                    word_score = 0.0

                final_score = float(round(word_score, 2))

                word_data = {
                    "word": word_info['word'],
                    "word_score": final_score,
                    "phonemes": [] 
                }
                
                # --- NEW: BASE64 ENCODING LOGIC ---
                if final_score >= 0.65:
                    pad_before_ms = 0
                    pad_after_ms = 150
                    
                    pad_before_samples = int((pad_before_ms / 1000.0) * sr)
                    pad_after_samples = int((pad_after_ms / 1000.0) * sr)

                    start_sample = max(0, int(word_info['start'] * sr) - pad_before_samples)
                    end_sample = min(len(audio_data), int(word_info['end'] * sr) + pad_after_samples)
                    
                    audio_slice = audio_data[start_sample:end_sample]

                    # Save slice to an in-memory buffer instead of disk
                    buffer = io.BytesIO()
                    sf.write(buffer, audio_slice, sr, format='WAV')
                    wav_bytes = buffer.getvalue()
                    
                    # Encode to Base64 string and add to the JSON node
                    word_data["audio_base64"] = base64.b64encode(wav_bytes).decode('utf-8')
                # ----------------------------------

                word_details.append(word_data)
                total_score += word_score
                words_processed += 1

        overall_score = (total_score / words_processed) if words_processed > 0 else 0

        return {
            "overall_pronunciation_score": float(round(overall_score, 2)),
            "word_details": word_details
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis processing failed: {str(e)}")
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)