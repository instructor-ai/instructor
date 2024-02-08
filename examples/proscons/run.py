import instructor
import openai
from pydantic import BaseModel
from typing import List

client = instructor.patch(openai.Client())


class Analysis(BaseModel):
    pros: List[str]
    cons: List[str]


analysis = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=Analysis,
    messages=[
        {
            "role": "system",
            "content": "You are a perfect entity extraction system",
        },
        {
            "role": "user",
            "content": "Give me a pro-con analysis of joining South Park Commons. ",
        },
    ],
)

print(analysis.model_dump_json(indent=2))
"""{
  "pros": [
    "Access to a supportive community of like-minded individuals",
    "Opportunities for collaboration and networking",
    "Access to shared resources and knowledge",
    "Exposure to diverse perspectives and ideas",
    "Potential for personal and professional growth"
  ],
  "cons": [
    "Membership fees and financial commitment",
    "Limited autonomy and flexibility",
    "Possible conflicts or disagreements within the community",
    "Adherence to community rules and guidelines",
    "Time commitment for participation in community activities"
  ]
}
"""
