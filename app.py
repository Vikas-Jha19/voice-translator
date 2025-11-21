"""
Multilingual Voice Translation System
Uses public HF Spaces via Gradio Client (most reliable)
"""

import gradio as gr
from gradio_client import Client
import os
import tempfile
from gtts import gTTS
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANGUAGES = {
    "hi": "Hindi", "en": "English", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali",
    "gu": "Gujarati", "pa": "Punjabi"
}

def translate(audio, src_lang, tgt_lang):
    if audio is None:
        return None, "Please provide audio input", "", ""
    
    lang_map = {v: k for k, v in LANGUAGES.items()}
    src = lang_map.get(src_lang, "hi")
    tgt = lang_map.get(tgt_lang, "en")
    
    # Handle microphone
    if isinstance(audio, tuple):
        import soundfile as sf
        sr, data = audio
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, data, sr)
        audio = tmp.name
    
    try:
        start = time.time()
        
        # ASR via public Whisper Space
        logger.info("Starting ASR...")
        whisper_client = Client("openai/whisper")
        asr_result = whisper_client.predict(
            audio,
            "transcribe",
            api_name="/predict"
        )
        # Result format: (transcription, None)
        src_text = asr_result[0] if isinstance(asr_result, tuple) else str(asr_result)
        src_text = src_text.strip()
        
        if not src_text:
            return None, "ASR failed - no text extracted", "", ""
            
        asr_time = time.time() - start
        logger.info(f"ASR complete: {src_text}")
        
        # Translation via NLLB Space
        logger.info("Starting translation...")
        trans_start = time.time()
        
        # Map to NLLB codes
        nllb_map = {
            "hi": "Hindi", "en": "English", "ta": "Tamil", "te": "Telugu",
            "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", 
            "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi"
        }
        
        nllb_client = Client("facebook/seamless_m4t")
        trans_result = nllb_client.predict(
            src_text,
            nllb_map.get(src, "Hindi"),
            nllb_map.get(tgt, "English"),
            api_name="/translate"
        )
        tgt_text = str(trans_result).strip()
        trans_time = time.time() - trans_start
        logger.info(f"Translation complete: {tgt_text}")
        
        # TTS
        logger.info("Starting TTS...")
        tts_start = time.time()
        tts_codes = {
            "hi": "hi", "en": "en", "ta": "ta", "te": "te",
            "kn": "kn", "ml": "ml", "mr": "mr", "bn": "bn",
            "gu": "gu", "pa": "pa"
        }
        tts = gTTS(text=tgt_text, lang=tts_codes.get(tgt, "en"), slow=False)
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(out.name)
        tts_time = time.time() - tts_start
        
        total = time.time() - start
        
        return (
            out.name,
            f"**Source ({src_lang}):** {src_text}",
            f"**Target ({tgt_lang}):** {tgt_text}",
            f"⚡ ASR: {asr_time:.1f}s | Translation: {trans_time:.1f}s | TTS: {tts_time:.1f}s | Total: {total:.1f}s"
        )
        
    except Exception as e:
        import traceback
        error = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        logger.error(error)
        return None, error, "", ""

# Build interface
demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio Input"),
        gr.Dropdown(list(LANGUAGES.values()), label="Source Language", value="Hindi"),
        gr.Dropdown(list(LANGUAGES.values()), label="Target Language", value="English")
    ],
    outputs=[
        gr.Audio(label="Translated Audio", type="filepath"),
        gr.Textbox(label="Transcription", lines=3),
        gr.Textbox(label="Translation", lines=3),
        gr.Textbox(label="Processing Time")
    ],
    title="🌍 Voice Translation System",
    description="Multilingual voice translation"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting on port {port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
