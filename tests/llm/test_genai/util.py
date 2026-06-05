import os
import instructor

models = [os.getenv("GOOGLE_GENAI_MODEL", "google/gemini-2.0-flash")]
modes = [instructor.Mode.GENAI_STRUCTURED_OUTPUTS]
