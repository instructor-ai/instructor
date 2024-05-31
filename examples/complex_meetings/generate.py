from models import Artifact, Document
from openai import OpenAI
from datetime import datetime
import instructor

client = instructor.from_openai(OpenAI())


def generate_artifact(transcript: str, artifact: Artifact):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""
        Generates a specific artifact based on the provided transcript and instructions.
        
        Artifact Type: {artifact.artifact_type}
        Title: {artifact.title}
        Date: {datetime.now().strftime("%Y-%m-%d")}
        
        Instructions:
        1. Carefully review the transcript to identify the relevant information for the {artifact.artifact_type}.
        2. Extract key details like dates, attendees, decisions, action items, etc. as specified in the artifact instructions.
        3. Organize the content according to the provided template or structure.
        4. Ensure the artifact is clear, concise, and captures the essential points from the transcript.
        5. Generate the full artifact content, aiming for a {artifact.lenght}.
        6. Never include any template or placeholder text like [Name]
        7. Never mention any names or dates that are not present in the transcript.
        
        Detailed Instructions:
        {artifact.instructions}
        """,
            },
            {"role": "user", "content": f"transcript: {transcript}"},
        ],
        response_model=Document,
    )


if __name__ == "__main__":
    with open("transcript.txt", "r") as f:
        transcript = f.read()

    document = generate_artifact(
        transcript,
        artifact=Artifact(
            artifact_type="followup-email",
            title="Project Update Follow-Up Email",
            lenght="200-300 words",
            instructions="Write a short, concise, and cordial email to the meeting attendees.",
        ),
    )
    print(document.model_dump_json(indent=2))
