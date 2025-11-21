"""
Multilingual Voice Translation System
Uses Hugging Face Inference API (models run on HF, code stays private)
"""

import gradio as gr
import os
from huggingface_hub import InferenceClient
import tempfile
from gtts import gTTS
import time

# Initialize HF Inference Client (free API)
client = InferenceClient(token=os.environ.get("HF_TOKEN"))

SUPPORTED_LANGUAGES = {
    "hi": "Hindi", "en": "English", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali",
    "gu": "Gujarati", "pa": "Punjabi"
}

NLLB_CODES = {
    "hi": "hin_Deva", "en": "eng_Latn", "ta": "tam_Taml", "te": "tel_Telu",
    "kn": "kan_Knda", "ml": "mal_Mlym", "mr": "mar_Deva", "bn": "ben_Beng",
    "gu": "guj_Gujr", "pa": "pan_Guru"
}

def process_translation(audio_input, source_language, target_language):
    """Voice translation using HF Inference API"""
    
    if audio_input is None:
        return None, "No audio input provided", "", ""
    
    lang_map = {v: k for k, v in SUPPORTED_LANGUAGES.items()}
    src_lang = lang_map.get(source_language, "hi")
    tgt_lang = lang_map.get(target_language, "en")
    
    # Handle microphone input
    if isinstance(audio_input, tuple):
        import soundfile as sf
        sr, audio_data = audio_input
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_audio.name, audio_data, sr)
        audio_input = temp_audio.name
    
    try:
        start_time = time.time()
        
        # Stage 1: ASR via HF API (Whisper Large V3)
        with open(audio_input, "rb") as f:
            audio_bytes = f.read()
        
        transcription = client.automatic_speech_recognition(
            audio_bytes,
            model="openai/whisper-large-v3"
        )
        source_text = transcription["text"].strip()
        asr_time = time.time() - start_time
        
        # Stage 2: Translation via HF API (NLLB)
        nmt_start = time.time()
        src_code = NLLB_CODES.get(src_lang, "hin_Deva")
        tgt_code = NLLB_CODES.get(tgt_lang, "eng_Latn")
        
        translation = client.translation(
            source_text,
            model="facebook/nllb-200-distilled-600M",
            src_lang=src_code,
            tgt_lang=tgt_code
        )
        target_text = translation["translation_text"]
        nmt_time = time.time() - nmt_start
        
        # Stage 3: TTS (Google)
        tts_start = time.time()
        gtts_codes = {
            "hi": "hi", "en": "en", "ta": "ta", "te": "te",
            "kn": "kn", "ml": "ml", "mr": "mr", "bn": "bn",
            "gu": "gu", "pa": "pa"
        }
        tts_lang = gtts_codes.get(tgt_lang, "en")
        
        tts = gTTS(text=target_text, lang=tts_lang, slow=False)
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(output_file.name)
        tts_time = time.time() - tts_start
        
        total_time = time.time() - start_time
        
        transcription_text = f"**Source ({source_language}):**\n\n{source_text}"
        translation_text = f"**Target ({target_language}):**\n\n{target_text}"
        metadata = f"""**Processing Time:**
- ASR: {asr_time:.2f}s
- Translation: {nmt_time:.2f}s
- TTS: {tts_time:.2f}s
- Total: {total_time:.2f}s

**Models Used (via HF API):**
- ASR: Whisper Large V3
- NMT: NLLB-200-600M
- TTS: Google Text-to-Speech"""
        
        return output_file.name, transcription_text, translation_text, metadata
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg, "", ""

# UI
language_options = list(SUPPORTED_LANGUAGES.values())

css = """
#main-container {
    max-width: 1200px;
    margin: 0 auto;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
#header {
    text-align: center;
    padding: 2rem 0;
    border-bottom: 1px solid #e5e7eb;
}
"""

with gr.Blocks(title="Voice Translation System", css=css) as demo:
    with gr.Column(elem_id="main-container"):
        with gr.Column(elem_id="header"):
            gr.Markdown("# Multilingual Voice Translation System")
            gr.Markdown("State-of-the-art speech translation for Indian languages")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                audio_input = gr.Audio(
                    label="Audio Input",
                    type="filepath",
                    sources=["microphone", "upload"]
                )
                with gr.Row():
                    source_lang = gr.Dropdown(
                        choices=language_options,
                        label="Source Language",
                        value="Hindi"
                    )
                    target_lang = gr.Dropdown(
                        choices=language_options,
                        label="Target Language",
                        value="English"
                    )
                process_btn = gr.Button("Translate", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                audio_output = gr.Audio(label="Synthesized Speech", type="filepath")
                transcription_output = gr.Markdown()
                translation_output = gr.Markdown()
                metadata_output = gr.Markdown()
        
        gr.Markdown("---")
        gr.Markdown("Powered by Hugging Face Inference API")
    
    process_btn.click(
        fn=process_translation,
        inputs=[audio_input, source_lang, target_lang],
        outputs=[audio_output, transcription_output, translation_output, metadata_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False
    )
