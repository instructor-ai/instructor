import os
import instructor

models = ["gpt-3.5-turbo-1106", "gpt-4", "gpt-4-1106-preview"]
modes = [
    instructor.Mode.FUNCTIONS,
    instructor.Mode.JSON,
    instructor.Mode.TOOLS,
    instructor.Mode.MD_JSON,
]
