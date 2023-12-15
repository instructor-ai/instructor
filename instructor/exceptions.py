class IncompleteOutputException(Exception):
    """Exception raised when the output from LLM is incomplete due to max tokens limit reached."""

    def __init__(self, message="The output is incomplete due to a max_tokens length limit."):
        self.message = message
        super().__init__(self.message)