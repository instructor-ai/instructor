from pydantic import BaseModel,Field,model_validator,FieldValidationInfo
from typing import List
import spacy

nlp = spacy.load("en_core_web_sm")

class Entity(BaseModel):
    """
    An entity is a real-world object that's assigned a name - for example, a person, country a product or a book title.
    """
    entity_name: str = Field(
        ...,description="This is the associated name with the entity that exists in the text"
    )


# Note that we utilise Spacy for entity recognition so that it is consistent with the original paper implementation which uses it as an original prompt
class Summary(BaseModel):
    """
    This represents a summary of some text passed to the model

    Guidelines
    - Make every word count : Rewrite the previous summary to improve flow and make space for additional entities
    - Never remove an existing entity from the original text
    - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses"
    - Missing entities can appear anywhere in the new summary
    """
    denser_summary:str = Field(...,description="Concise yet self-contained summary")

    @model_validator(mode="after")
    def validate_sources(self, info: FieldValidationInfo) -> "Summary":
        # We first extract out the original summary and compute the entity count
        original_content = info.context["original_content"]
        if original_content is None:
            raise ValueError("Plese supply an original summary to compare the generated summary to")
        doc = nlp(original_content)
        original_entities = doc.ents

        new_summary = self.denser_summary
        new_doc = nlp(new_summary)
        new_entities = new_doc.ents

        missing_entities = [entity for entity in original_entities if entity not in new_entities]

        # Validate that we have at least the same number of entities
        if len(missing_entities) >= 1:
            raise ValueError(f"Entites were removed. Please regenerate a new summary that contains the entities {','.join(missing_entities)}")


        return self


    

