import os
import random
import asyncio

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

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
from agents.run import RunConfig
from agents import set_default_openai_client
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key="AIzaSyCNU7yLsXzjxyd-sVDIZTMQyAKkaRay3pg",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

set_default_openai_client(external_client)

model = OpenAIChatCompletionsModel (
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig (
    model=model,
    model_provider = external_client,
    tracing_disabled = True
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


async def main():
    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))
    buffer = np.zeros(24000 * 3, dtype=np.int16)
    audio_input = AudioInput(buffer=buffer)

    result = await pipeline.run(audio_input)

    # Create an audio player using `sounddevice`
    player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
    player.start()

    # Play the audio stream as it comes in
    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            player.write(event.data)


if __name__ == "__main__":
    asyncio.run(main())