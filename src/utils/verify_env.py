import os
import sys
from dotenv import load_dotenv

REQUIRED_ENV_VARS = [
    "LABEL_STUDIO_URL",
    "LABEL_STUDIO_PROJECT_ID",
    "LABEL_STUDIO_USERNAME",
    "LABEL_STUDIO_PASSWORD",
]


def verify_env(required_vars=None):
    """
    Verifies that all required environment variables exist in the environment.
    """
    load_dotenv()
    if required_vars is None:
        required_vars = REQUIRED_ENV_VARS

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(
            f"❌ Error: Missing required environment variables in .env: {', '.join(missing_vars)}"
        )
        sys.exit(1)
