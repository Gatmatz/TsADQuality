import logging


class PostgresDatabaseHandler(logging.Handler):
    """
    Custom logging handler that sends ERROR logs to the Postgres database.
    """

    def emit(self, record):
        from tsadquality.data.clients.PostgresClient import PostgresClient

        try:
            file_info = f"{record.name}:{record.lineno}"
            experiment_id = getattr(record, "experiment_id", "SYSTEM_LOG")

            error_msg = record.getMessage()
            if record.exc_info:
                error_msg += f"\nTraceback: {self.format(record)}"

            PostgresClient.write_runtime_error(
                experiment_id=experiment_id,
                file_path=file_info,
                error_message=error_msg,
            )
        except Exception as e:
            import sys
            if sys.stderr is not None:
                sys.stderr.write(f"[Logging Warning] PostgresDatabaseHandler could not log to DB: {e}\n")


def get_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger to be used across the project.
    - `name`: typically `__name__` from the caller.
    - `level`: logging level (default INFO).
    Ensures a single StreamHandler is added only once to avoid duplicate logs.
    """
    logger_name = name or "TsADQuality"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # If no handlers attached, add a StreamHandler with a standard formatter.
    if not logger.handlers:
        # 1. Standard Console Logger for INFO and DEBUG
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s:%(lineno)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # 2. Automated Postgres Logging only for ERROR
        db_handler = PostgresDatabaseHandler()
        db_handler.setLevel(logging.ERROR)
        logger.addHandler(db_handler)

        # Prevent double logging if root logger is also configured.
        logger.propagate = False

    return logger