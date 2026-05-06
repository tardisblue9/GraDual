"""Slimmed model registry for the GraDual deliverable.

Only OpenAI-compatible providers are wired up. Add your own model here if you
need it; pointing the provider at "openai" (with custom OPENAI_BASE_URL) covers
most third-party endpoints that speak the OpenAI API.
"""
from agentdojo.strenum import StrEnum


class ModelsEnum(StrEnum):
    """Currently supported models in this slimmed distribution."""

    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4O_MINI_2024_07_18 = "gpt-4o-mini-2024-07-18"
    QWEN3_CODER = "qwen3-coder"
    QWEN3_235B = "qwen3-235b-a22b-thinking-2507"
    QWEN2_5_CODER_32B = "qwen2.5-coder-32b-instruct"
    DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3"
    DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1"


MODEL_PROVIDERS = {
    ModelsEnum.GPT_4O_2024_05_13: "openai",
    ModelsEnum.GPT_4O_MINI_2024_07_18: "openai",
    ModelsEnum.QWEN3_CODER: "openai",
    ModelsEnum.QWEN3_235B: "openai",
    ModelsEnum.QWEN2_5_CODER_32B: "openai",
    ModelsEnum.DEEPSEEK_V3: "deepseek",
    ModelsEnum.DEEPSEEK_R1: "deepseek",
}


MODEL_NAMES = {
    "gpt-4o-2024-05-13": "GPT-4",
    "gpt-4o-mini-2024-07-18": "GPT-4",
    "qwen3-coder": "Qwen created by Alibaba Cloud.",
    "qwen3-235b-a22b-thinking-2507": "Qwen created by Alibaba Cloud.",
    "qwen2.5-coder-32b-instruct": "Qwen created by Alibaba Cloud.",
    "deepseek-ai/DeepSeek-V3": "DeepSeek V3",
    "deepseek-ai/DeepSeek-R1": "DeepSeek R1",
}
