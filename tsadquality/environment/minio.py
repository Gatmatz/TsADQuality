import os

from dotenv import load_dotenv

load_dotenv()
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD")
MINIO_HOST = os.getenv("MINIO_HOST")
MINIO_MAPPED_PORT = os.getenv("MINIO_MAPPED_PORT")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")
