# Complex Meetings Artifact Generator

This project demonstrates how to use OpenAI's GPT-4 model to automatically generate relevant artifacts from a meeting transcript. The artifacts are based on predefined document types specified in `descriptions.yaml`.

## How it works

1. The meeting transcript is provided as input through the clipboard.
2. The `extract_artifacts` function in `extract.py` uses the GPT-4 model to analyze the transcript and determine which types of artifacts are most relevant based on the content. It consults the document type descriptions loaded from `descriptions.yaml` to make this determination.
3. For each relevant artifact type, the model extracts the necessary information from the transcript to populate the artifact template.
4. The `generate_artifact` function in `generate.py` then uses the GPT-4 model to generate the full content of each artifact based on the extracted information and the template.
5. The generated artifacts are returned as a list of `Artifact` objects, which include the artifact type, title, and content.

## How to run

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure your OpenAI API key:
   - Set the `OPENAI_API_KEY` environment variable with your API key.

3. Copy the meeting transcript to your clipboard.

4. Run the script:
   ```
   python run.py
   ```

5. The generated artifacts will be printed to the console.

## Configuration

- `descriptions.yaml`: Contains the predefined document types and their templates. You can modify this file to add, remove, or update the available artifact types.

## Key Files

- `run.py`: The main entry point for running the artifact generation process from the clipboard.
- `extract.py`: Contains the `extract_artifacts` function for determining the relevant artifact types based on the transcript.
- `generate.py`: Contains the `generate_artifact` function for generating the full content of each artifact.
- `configuration.py`: Loads the document type descriptions from `descriptions.yaml`.

## Dependencies

- OpenAI API
- instructor 
- pydantic
- typer
- pyperclip
- rich
- pyyaml

Make sure to have an OpenAI API key to use this project.

