import instructor

models = ["gpt-4.1-mini"]
modes = [
    # instructor.Mode.TOOLS,
    # instructor.Mode.TOOLS_STRICT,
    instructor.Mode.RESPONSES_TOOLS,
    instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
]
