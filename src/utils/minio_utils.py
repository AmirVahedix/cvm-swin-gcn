import os
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


def get_minio_credentials():
    """Extracts MinIO connection parameters from environment variables."""
    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = (
        os.getenv("MINIO_ACCESS_KEY")
        or os.getenv("MINIO_USERNAME")
        or os.getenv("MINIO_USER")
    )
    secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_PASSWORD")
    bucket = os.getenv("MINIO_BUCKET", "cvm-artifacts")

    return {
        "endpoint": endpoint,
        "access_key": access_key,
        "secret_key": secret_key,
        "bucket": bucket,
    }


def get_minio_client():
    """Initializes and returns a boto3 S3 client configured for MinIO."""
    creds = get_minio_credentials()

    if not creds["endpoint"]:
        raise ValueError("MINIO_ENDPOINT environment variable is not set.")
    if not creds["access_key"]:
        raise ValueError(
            "MINIO_ACCESS_KEY (or MINIO_USERNAME) environment variable is not set."
        )
    if not creds["secret_key"]:
        raise ValueError(
            "MINIO_SECRET_KEY (or MINIO_PASSWORD) environment variable is not set."
        )

    endpoint_url = creds["endpoint"]
    if not endpoint_url.startswith("http://") and not endpoint_url.startswith("https://"):
        endpoint_url = f"https://{endpoint_url}"

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"],
        config=Config(
            signature_version="s3v4",
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            s3={"addressing_style": "path"},
        ),
    )
    return client


def ensure_bucket_exists(s3_client, bucket_name: str):
    """Checks if the bucket exists on MinIO; creates it if it does not exist."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in ["404", "NoSuchBucket", "NotFound"]:
            print(f"Bucket '{bucket_name}' not found. Creating bucket...")
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"✅ Bucket '{bucket_name}' created successfully.")
        else:
            raise e


def upload_artifact_to_minio(
    file_path: str,
    object_name: str = None,
    bucket_name: str = None,
    s3_client=None,
) -> bool:
    """
    Uploads a file artifact to MinIO S3 storage using put_object with explicit ContentLength.

    Args:
        file_path (str): Local path of the file to upload.
        object_name (str, optional): Target object key in MinIO. Defaults to file basename.
        bucket_name (str, optional): Target MinIO bucket. Defaults to MINIO_BUCKET env var.
        s3_client (boto3.client, optional): Existing boto3 S3 client instance.

    Returns:
        bool: True if upload succeeded, False otherwise.
    """
    if not os.path.isfile(file_path):
        print(f"❌ Upload failed: Local file '{file_path}' does not exist.")
        return False

    if object_name is None:
        object_name = os.path.basename(file_path)

    if bucket_name is None:
        bucket_name = os.getenv("MINIO_BUCKET", "cvm-artifacts")

    try:
        if s3_client is None:
            s3_client = get_minio_client()
        ensure_bucket_exists(s3_client, bucket_name)
        file_size = os.path.getsize(file_path)
        with open(file_path, "rb") as f:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=object_name,
                Body=f,
                ContentLength=file_size,
            )
        print(f"✅ Successfully uploaded '{file_path}' as '{object_name}' to MinIO bucket '{bucket_name}'.")
        return True
    except Exception as e:
        print(f"❌ Failed to upload '{object_name}' to MinIO bucket '{bucket_name}': {e}")
        return False
