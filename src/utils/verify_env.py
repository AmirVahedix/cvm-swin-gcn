import os
import sys
from dotenv import load_dotenv

REQUIRED_CVM_API_ENV_VARS = [
    "CVM_API_URL",
    "CVM_API_TOKEN",
]


def verify_env(required_vars=None, data_source=None):
    """
    Verifies that all required environment variables exist in the environment.
    Single source of truth is the CVM Backend API.
    """
    load_dotenv()
    if required_vars is None:
        required_vars = REQUIRED_CVM_API_ENV_VARS

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(
            f"❌ Error: Missing required CVM API environment variables in .env: {', '.join(missing_vars)}"
        )
        sys.exit(1)
