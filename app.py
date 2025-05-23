import os
import random
import asyncio
import json
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer
import pyttsx3
import wave
import tempfile
from typing import AsyncIterator, Dict, Any
import time

# Local Gemini client implementation
class AsyncGeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

# Modified agent imports to avoid OpenAI dependencies
from agents import Agent, function_tool
from agents.voice import (
    AudioInput,
    VoicePipeline,
    SingleAgentVoiceWorkflow,
    STTModel,
    TTSModel
)
from agents.run import RunConfig

load_dotenv()

# Custom STT Model using Vosk (completely local)
class VoskSTTModel(STTModel):
    def __init__(self, model_path="vosk-model-small-en-us-0.15"):
        if not os.path.exists(model_path):
            raise ValueError(f"Download Vosk model and place at: {model_path}")
        self.model = Model(model_path)
        self.sample_rate = 24000
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
    
    @property
    def model_name(self) -> str:
        return "vosk-local"
    
    async def transcribe(self, audio_input: AudioInput, *args, **kwargs) -> str:
        audio_data = audio_input.buffer.tobytes()
        if self.recognizer.AcceptWaveform(audio_data):
            result = json.loads(self.recognizer.FinalResult())
            return result.get("text", "")
        return ""

    async def create_session(self, audio_input, *args, **kwargs):
        # Implement a basic session handler for Vosk
        # This is a simplified version - you might need to adapt it to your needs
        class VoskSession:
            def __init__(self, recognizer, buffer):
                self.recognizer = recognizer
                self.buffer = buffer
            
            async def transcribe_turns(self):
                # For Vosk, we'll just process the whole buffer at once
                audio_data = self.buffer.tobytes()
                if self.recognizer.AcceptWaveform(audio_data):
                    result = json.loads(self.recognizer.FinalResult())
                    yield result.get("text", "")
            
            async def close(self):
                pass

        return VoskSession(self.recognizer, audio_input.buffer)

# Robust TTS Model using PyTTSx3 with proper temp file handling
class PyTTSx3TTSModel(TTSModel):
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
    
    @property
    def model_name(self) -> str:
        return "pyttsx3-local"
    
    async def _create_temp_file(self) -> str:
        """Creates a uniquely named temp file with proper permissions"""
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time() * 1000)
        return os.path.join(temp_dir, f"tts_{timestamp}_{os.getpid()}.wav")
    
    async def synthesize(self, text: str, *args, **kwargs) -> np.ndarray:
        temp_path = await self._create_temp_file()
        
        try:
            self.engine.save_to_file(text, temp_path)
            self.engine.runAndWait()
            
            # Wait for file to be written
            for _ in range(10):
                if os.path.exists(temp_path):
                    break
                await asyncio.sleep(0.1)
            
            with wave.open(temp_path, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                return np.frombuffer(frames, dtype=np.int16)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    async def run(self, *args, **kwargs) -> Any:
        """Implementation of the abstract run method"""
        # Since pyttsx3 doesn't have a streaming interface, we'll just use synthesize
        # You might need to adapt this based on what the base class expects
        text = kwargs.get('text', '')
        return await self.synthesize(text)

# Initialize Gemini client (if you still want to use it)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    gemini_client = AsyncGeminiClient(api_key=GEMINI_API_KEY)
else:
    print("Warning: No Gemini API key found, using local models only")
    gemini_client = None

# Initialize models
stt_model = VoskSTTModel()
tts_model = PyTTSx3TTSModel()

# Create a simple local model wrapper if not using Gemini
class LocalModel:
    async def generate_response(self, prompt: str) -> str:
        # Very basic local response - replace with your own logic
        return f"I received your message: {prompt}"

local_model = LocalModel()

# Agent setup
@function_tool
def get_weather(city: str) -> str:
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"Weather in {city} is {random.choice(choices)}."

agent = Agent(
    name="Assistant",
    instructions="You're a helpful local assistant.",
    model=local_model,  # Using our local model
    tools=[get_weather]
)

async def main():
    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent),
        stt_model=stt_model,
        tts_model=tts_model
    )
    
    # Initialize with 1 second of silence
    buffer = np.zeros(24000, dtype=np.int16)
    audio_input = AudioInput(buffer=buffer)

    result = await pipeline.run(audio_input)

    with sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16) as player:
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                player.write(event.data)

if __name__ == "__main__":
    asyncio.run(main())