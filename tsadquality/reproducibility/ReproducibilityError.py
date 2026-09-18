class ReproducibilityError(Exception):
    """Exception raised when reproducibility of random operations cannot be ensured."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return str(self.message)