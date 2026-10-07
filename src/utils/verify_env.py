import os
import sys
from dotenv import load_dotenv

REQUIRED_ENV_VARS = [
    "LABEL_STUDIO_URL",
    "LABEL_STUDIO_PROJECT_ID",
]


def verify_env(required_vars=None):
    """
    Verifies that all required environment variables exist in the environment.
    Supports either LABEL_STUDIO_API_TOKEN or (LABEL_STUDIO_USERNAME and LABEL_STUDIO_PASSWORD).
    """
    load_dotenv()
    if required_vars is None:
        required_vars = REQUIRED_ENV_VARS

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    # Check authentication credentials: at least API token or username/password must be present
    has_token = bool(os.getenv("LABEL_STUDIO_API_TOKEN"))
    has_user_pass = bool(os.getenv("LABEL_STUDIO_USERNAME")) and bool(os.getenv("LABEL_STUDIO_PASSWORD"))

    if not has_token and not has_user_pass:
        missing_vars.append("LABEL_STUDIO_API_TOKEN (or LABEL_STUDIO_USERNAME and LABEL_STUDIO_PASSWORD)")

    if missing_vars:
        print(
            f"❌ Error: Missing required environment variables in .env: {', '.join(missing_vars)}"
        )
        sys.exit(1)
