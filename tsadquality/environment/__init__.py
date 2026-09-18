import os

from dotenv import load_dotenv

load_dotenv()
EXECUTION_PROFILE = os.getenv("EXECUTION_PROFILE")
RANDOM_SEEDS = [
    int(seed.strip()) for seed in os.getenv("RANDOM_SEEDS", "").split(",") if seed.strip()
]
