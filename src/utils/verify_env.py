import os
import sys
from dotenv import load_dotenv

REQUIRED_ENV_VARS = [
    "LABEL_STUDIO_URL",
    "LABEL_STUDIO_PROJECT_ID",
    "LABEL_STUDIO_USERNAME",
    "LABEL_STUDIO_PASSWORD",
    "MINIO_ENDPOINT",
]


def verify_env(required_vars=None):
    """
    Verifies that all required environment variables exist in the environment.
    """
    load_dotenv()
    if required_vars is None:
        required_vars = REQUIRED_ENV_VARS

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    # Check MinIO access key (accept MINIO_ACCESS_KEY, MINIO_USERNAME, or MINIO_USER)
    if not (os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_USERNAME") or os.getenv("MINIO_USER")):
        missing_vars.append("MINIO_ACCESS_KEY (or MINIO_USERNAME)")

    # Check MinIO secret key (accept MINIO_SECRET_KEY or MINIO_PASSWORD)
    if not (os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_PASSWORD")):
        missing_vars.append("MINIO_SECRET_KEY (or MINIO_PASSWORD)")

    if missing_vars:
        print(
            f"❌ Error: Missing required environment variables in .env: {', '.join(missing_vars)}"
        )
        sys.exit(1)
