import os
import instructor

models = [os.getenv("GOOGLE_GENAI_MODEL", "google/gemini-pro")]
modes = [instructor.Mode.GENAI_STRUCTURED_OUTPUTS]
