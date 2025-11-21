"""
Multilingual Voice Translation System
Uses Hugging Face Inference API
"""

import gradio as gr
import os
from huggingface_hub import InferenceClient
import tempfile
from gtts import gTTS
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize HF client
hf_token = os.environ.get("HF_TOKEN")
client = InferenceClient(token=hf_token) if hf_token else InferenceClient()

LANGUAGES = {
    "hi": "Hindi", "en": "English", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali",
    "gu": "Gujarati", "pa": "Punjabi"
}

NLLB_CODES = {
    "hi": "hin_Deva", "en": "eng_Latn", "ta": "tam_Taml", "te": "tel_Telu",
    "kn": "kan_Knda", "ml": "mal_Mlym", "mr": "mar_Deva", "bn": "ben_Beng",
    "gu": "guj_Gujr", "pa": "pan_Guru"
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
        
        # ASR
        with open(audio, "rb") as f:
            result = client.automatic_speech_recognition(
                f.read(),
                model="openai/whisper-large-v3"
            )
        src_text = result["text"].strip()
        asr_time = time.time() - start
        
        # Translation
        trans_start = time.time()
        trans = client.translation(
            src_text,
            model="facebook/nllb-200-distilled-600M",
            src_lang=NLLB_CODES.get(src, "hin_Deva"),
            tgt_lang=NLLB_CODES.get(tgt, "eng_Latn")
        )
        tgt_text = trans["translation_text"]
        trans_time = time.time() - trans_start
        
        # TTS
        tts_start = time.time()
        tts_codes = {"hi": "hi", "en": "en", "ta": "ta", "te": "te",
                     "kn": "kn", "ml": "ml", "mr": "mr", "bn": "bn",
                     "gu": "gu", "pa": "pa"}
        tts = gTTS(text=tgt_text, lang=tts_codes.get(tgt, "en"), slow=False)
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(out.name)
        tts_time = time.time() - tts_start
        
        total = time.time() - start
        
        return (
            out.name,
            f"**Source ({src_lang}):** {src_text}",
            f"**Target ({tgt_lang}):** {tgt_text}",
            f"ASR: {asr_time:.1f}s | Translation: {trans_time:.1f}s | TTS: {tts_time:.1f}s | Total: {total:.1f}s"
        )
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return None, str(e), "", ""

# Simple interface
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
    title="Voice Translation System",
    description="Multilingual voice translation using Whisper Large V3 + NLLB-600M"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting on port {port}")
    demo.launch(server_name="0.0.0.0", server_port=port)
