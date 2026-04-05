"""
Example usage of RVLM (Recursive Vision Language Model) with image inputs.

Run with: python -m examples.rvlm_example

Requires:
    - A vision-capable model (e.g., gemini-2.5-flash, gpt-4o, claude-3-5-sonnet)
    - GEMINI_API_KEY (or OPENAI_API_KEY) environment variable set in .env
"""

from dotenv import load_dotenv

from rvlm import RVLM
from rvlm.logger import RLMLogger

load_dotenv()

logger = RLMLogger(log_dir="./logs")

# Create an RVLM instance with a vision-capable model
rvlm = RVLM(
    backend="gemini",
    backend_kwargs={
        "model_name": "gemini-2.5-flash",
    },
    environment="local",
    max_depth=1,
    logger=logger,
    verbose=True,
)

# Example 1: Analyze an image from a URL
result = rvlm.completion(
    prompt="Describe what you see in this image and identify any text present.",
    images=[
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
    ],
)
print(f"Result: {result.response}")
print(f"Execution time: {result.execution_time:.2f}s")
print(f"Usage: {result.usage_summary.to_dict()}")

# Example 2: Compare multiple images (using local file paths)
# result = rvlm.completion(
#     prompt="Compare these two images and describe the differences.",
#     images=["path/to/image1.jpg", "path/to/image2.jpg"],
#     root_prompt="What are the key differences between the two images?",
# )

# Example 3: Without images, RVLM falls back to standard RLM behavior
# (Commented out to avoid hitting free-tier rate limits)
# result_text = rvlm.completion(
#     prompt="What is 2 + 2?",
# )
# print(f"Text-only result: {result_text.response}")
