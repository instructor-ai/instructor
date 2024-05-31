from pydantic import BaseModel, Field
from typing import List


class Artifact(BaseModel):
    artifact_type: str
    title: str = Field(
        description="The title of the document / artifact, should be short and descriptive and give a preview of what kind of content is included in the artifact"
    )
    lenght: str = Field(
        description="Reason about the length of this artifact. Should it be short, medium, or long? Give an estimate in the number of words."
    )
    instructions: str = Field(
        description="Detailed Instructions for another subsystem on how to generate a high quality artifact, include both instructions templates and also key information that is needed to generate the artifact"
    )


class Artifacts(BaseModel):
    planning: str = Field(
        description="Detailed plan to determine what artifacts are needed"
    )
    artifacts: List[Artifact]


class Document(BaseModel):
    filename: str = Field(
        description="The filename of the document, not extentions, it'll all be .md"
    )
    title: str = Field(description="The title of the document")
    content: str = Field(description="must be in markdown format")

    def save(self):
        with open(f"{self.filename}.md", "w") as f:
            f.write(self.content)
