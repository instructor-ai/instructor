from configuration import descriptions
from models import Artifacts
from openai import OpenAI
import instructor

client = instructor.from_openai(OpenAI())


def extract_artifacts(transcript: str):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""
        Generates the appropriate artifacts from a meeting transcript.
        
        Instructions:
        1. Read through the transcript carefully to understand the main topics, decisions, and action items discussed.
        2. Consult the descriptions loaded in the configuration to determine which artifact types are most relevant based on the transcript content. Key things to look for:
            - Was this a decision-making meeting? Consider generating meeting minutes.
            - Were specific tasks or next steps identified? An action plan may be appropriate.
            - Did the meeting focus on reviewing or providing feedback? A feedback report could be useful.
            - Was this an interview or Q&A discussion? An interview summary or Q&A document may be best.
            - Was there recommendations or made? A memo or spec document may be useful.
        3. For each relevant artifact type identified, extract the necessary information from the transcript to populate the artifact template. This may include:
            - Meeting details like date, time, attendees for meeting minutes
            - Specific tasks, owners, and due dates for action plans
            - Key points, insights, and recommendations for feedback reports
            - Questions, responses, and highlights for interview summaries or Q&A docs
        4. Generate a clear, concise title for each artifact based on the main topic or purpose.
        5. Compile the populated templates into the list of Artifact objects to return.
        6. Provide a detailed summary of your planning process and rationale in the 'planning' field of the Artifacts object.
        
        The number of artifacts generated will depend on the content and scope of the transcript. Focus on quality and relevance rather than quantity. The goal is to create a focused set of artifacts that capture and communicate the essential information and outcomes from the discussion.

        Here are some examples of artifacts that might be generated:
        {descriptions}
            """,
            },
            {"role": "user", "content": f"transcript: {transcript}"},
        ],
        response_model=Artifacts,
    )


if __name__ == "__main__":
    with open("/transcript.txt", "r") as f:
        transcript = f.read()

    from rich.console import Console

    console = Console()
    for artifacts in extract_artifacts(transcript):
        console.clear()
        console.print(artifacts.model_dump_json(indent=2))
