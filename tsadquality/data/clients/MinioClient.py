import io

import numpy as np
from minio import Minio
from minio.error import S3Error

from tsadquality.environment.minio import (
    MINIO_BUCKET,
    MINIO_HOST,
    MINIO_MAPPED_PORT,
    MINIO_ROOT_PASSWORD,
    MINIO_ROOT_USER,
)
from tsadquality.logging.logging import get_logger

LOG = get_logger(__file__)


class SingletonMinioClient(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class _MinioClient:
    _client = Minio(
        endpoint=f"{MINIO_HOST}:{MINIO_MAPPED_PORT}",
        access_key=MINIO_ROOT_USER,
        secret_key=MINIO_ROOT_PASSWORD,
        secure=False,
    )
    _bucket_ready = False


class MinioClient(_MinioClient, metaclass=SingletonMinioClient):
    @classmethod
    def _ensure_bucket(cls) -> None:
        if cls._bucket_ready:
            return
        if not cls._client.bucket_exists(MINIO_BUCKET):
            try:
                cls._client.make_bucket(MINIO_BUCKET)
                LOG.info(f"Created MinIO bucket '{MINIO_BUCKET}'")
            except S3Error as e:
                # Another shard created it between the check and this call.
                if e.code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                    raise
        cls._bucket_ready = True

    @classmethod
    def put_array(cls, object_name: str, array: np.ndarray) -> None:
        cls._ensure_bucket()
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        size = buffer.tell()
        buffer.seek(0)
        cls._client.put_object(
            MINIO_BUCKET,
            object_name,
            buffer,
            length=size,
            content_type="application/octet-stream",
        )

    @classmethod
    def get_array(cls, object_name: str) -> np.ndarray:
        response = cls._client.get_object(MINIO_BUCKET, object_name)
        try:
            return np.load(io.BytesIO(response.read()), allow_pickle=False)
        finally:
            response.close()
            response.release_conn()

    @classmethod
    def object_exists(cls, object_name: str) -> bool:
        try:
            cls._client.stat_object(MINIO_BUCKET, object_name)
            return True
        except S3Error as e:
            if e.code in ("NoSuchKey", "NoSuchBucket"):
                return False
            raise

    @staticmethod
    def decision_scores_object_name(detector: str, random_seed: str, dataset_name: str) -> str:
        return f"perfect/{detector}/{random_seed}/{dataset_name}.npy"

    @classmethod
    def write_decision_scores(
        cls, detector: str, random_seed: str, dataset_name: str, scores: np.ndarray
    ) -> None:
        object_name = cls.decision_scores_object_name(detector, random_seed, dataset_name)
        try:
            cls.put_array(object_name, np.asarray(scores, dtype=float))
            LOG.info(f"Wrote decision scores to MinIO '{MINIO_BUCKET}/{object_name}'")
        except Exception as e:
            LOG.error(f"Failed to write decision scores '{object_name}' to MinIO. Error: {e}")
            raise

    @classmethod
    def read_decision_scores(cls, detector: str, random_seed: str, dataset_name: str) -> np.ndarray:
        return cls.get_array(cls.decision_scores_object_name(detector, random_seed, dataset_name))
