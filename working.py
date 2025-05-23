import os
import json
import random
import asyncio
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
import pyttsx3  # Free TTS
from vosk import Model, KaldiRecognizer  # Free STT
from scipy.signal import resample  # For resampling audio
from agents.run import RunConfig
from agents import set_default_openai_client
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from agents import (
    Agent,
    function_tool,
    AsyncOpenAI,
    OpenAIChatCompletionsModel
)
from agents.voice import (
    AudioInput,
    VoicePipeline,
    SingleAgentVoiceWorkflow,
)

load_dotenv()

# Initialize free TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Initialize free STT model
vosk_model = Model("vosk-model-small-en-us-0.15")  # Download from vosk.ai

# Gemini configuration
external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

set_default_openai_client(external_client)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."

spanish_agent = Agent(
    name="Spanish",
    handoff_description="A spanish speaking agent.",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. Speak in Spanish.",
    ),
    model=model,
)

agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. If the user speaks in Spanish, handoff to the spanish agent.",
    ),
    model=model,
    handoffs=[spanish_agent],
    tools=[get_weather],
)

async def record_audio(duration=5.0, sample_rate=44100):
    """Record at a supported rate (e.g., 44100 Hz)."""
    print("Recording...")
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    audio_data = recording.flatten()
    print(f"Audio max amplitude: {np.max(np.abs(audio_data))}")
    return audio_data

async def speech_to_text(audio_data, original_rate=44100, target_rate=16000):
    """Resample audio to 16 kHz for Vosk."""
    # Resample if needed
    if original_rate != target_rate:
        num_samples = int(len(audio_data) * target_rate / original_rate)
        audio_data = resample(audio_data, num_samples).astype(np.int16)
    
    recognizer = KaldiRecognizer(vosk_model, target_rate)
    recognizer.AcceptWaveform(audio_data.tobytes())
    result = json.loads(recognizer.Result())
    return result.get("text", "").strip()

async def text_to_speech(text):
    """Convert text to speech (with pyttsx3 workaround)."""
    tts_engine.say(text)
    try:
        tts_engine.runAndWait()
    except RuntimeError:
        # Workaround for pyttsx3 async issues
        pass

async def main():
    try:
        pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))
        sd.default.device = [9, 8]  # WASAPI mic (ID 9), speakers (ID 8)
        sd.default.samplerate = 44100  # Use a supported rate

        while True:
            # Record once per loop
            audio_data = await record_audio(sample_rate=44100)  # Record at 44.1 kHz
            text = await speech_to_text(audio_data, original_rate=44100, target_rate=16000)  # Resample to 16 kHz
            print(f"Raw Vosk output: {text}")  # Debug
            
            if not text:
                print("No speech detected")
                await text_to_speech("I didn't hear you. Please try again.")
                continue
            
            print(f"You said: {text}")
            
            if text.lower() in ["exit", "quit", "stop"]:
                await text_to_speech("Goodbye!")
                break
                
            # Get agent response
            response = await pipeline.run(text)
            print(f"Assistant: {response}")
            await text_to_speech(response)
            
    except KeyboardInterrupt:
        await text_to_speech("Goodbye!")
    except Exception as e:
        print(f"Error: {e}")
        await text_to_speech("Sorry, I encountered an error.")

if __name__ == "__main__":
    # Configure audio devices
    print("Available audio devices:")
    print(sd.query_devices())
    sd.default.device = [9, 8]  # WASAPI microphone (ID 9) and speakers (ID 8)
    sd.default.samplerate = 16000  # Must match Vosk's expected rate
    
    asyncio.run(main())