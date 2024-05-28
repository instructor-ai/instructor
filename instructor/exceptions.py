class IncompleteOutputException(Exception):
    """Exception raised when the output from LLM is incomplete due to max tokens limit reached."""

    def __init__(
        self,
        *args,
        last_completion=None,
        message: str = "The output is incomplete due to a max_tokens length limit.",
        **kwargs,
    ):
        self.last_completion = last_completion
        super().__init__(message, *args, **kwargs)
        super().__init__(*args, **kwargs)
