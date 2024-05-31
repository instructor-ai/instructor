import yaml
from pydantic import BaseModel, TypeAdapter
from typing import List


class DocumentType(BaseModel):
    slug: str
    description: str
    noteable_features: List[str]
    instructions_or_template: str


Config = TypeAdapter(List[DocumentType])


with open("./descriptions.yaml", "r") as f:
    try:
        descriptions = yaml.safe_load(f)
        descriptions = Config.validate_python(descriptions)
        print("Loaded descriptions.yaml successfully")
    except yaml.YAMLError as exc:
        print(f"Error loading descriptions.yaml: {exc}")
