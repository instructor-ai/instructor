import os
import instructor

if os.getenv("OPENAI_BASE_URL", None) == "https://api.endpoints.anyscale.com/v1":
    models = ["mistralai/Mistral-7B-Instruct-v0.1"]
    modes = [instructor.Mode.JSON_SCHEMA]
else:
    models = ["gpt-3.5-turbo-1106", "gpt-4", "gpt-4-1106-preview"]
    modes = [
        instructor.Mode.FUNCTIONS,
        instructor.Mode.JSON,
        instructor.Mode.TOOLS,
        instructor.Mode.MD_JSON,
    ]

if __name__ == "__main__":
    print(models)
    print(modes)
