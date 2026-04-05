import os

from dotenv import load_dotenv

from rvlm import RLM
from rvlm.logger import RLMLogger

load_dotenv()

logger = RLMLogger(log_dir="./logs")

rvlm = RLM(
    backend="gemini",
    backend_kwargs={
        "model_name": "gemini-2.5-flash",
        "api_key": os.getenv("GOOGLE_API_KEY"),
    },
    environment="local",
    environment_kwargs={},
    max_depth=1,
    logger=logger,
    verbose=True,  # For printing to console with rich, disabled by default.
)

result = rvlm.completion("Print me the first 5 powers of two, each on a newline.")

print(result)
