import os
import instructor

models = [os.getenv("GOOGLE_GENAI_MODEL", "google/gemini-3.5-flash")]
modes = [instructor.Mode.GENAI_STRUCTURED_OUTPUTS]
