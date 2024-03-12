import re
import json
import xmltodict
from pydantic import BaseModel
import xml.etree.ElementTree as ET
from typing import Type, Any, Dict, TypeVar

T = TypeVar("T", bound=BaseModel)


def json_to_xml(model: Type[BaseModel]) -> str:
    """Takes a Pydantic model and returns XML format for Anthropic function calling."""
    model_dict = model.model_json_schema()

    root = ET.Element("tool_description")
    tool_name = ET.SubElement(root, "tool_name")
    tool_name.text = model_dict.get("title", "Unknown")
    description = ET.SubElement(root, "description")
    description.text = (
        "This is the function that must be used to construct the response."
    )
    parameters = ET.SubElement(root, "parameters")
    references = model_dict.get("$defs", {})
    _add_params(parameters, model_dict, references)

    return ET.tostring(root, encoding="unicode")


def _add_params(
    root: ET.Element, model_dict: Dict[str, Any], references: Dict[str, Any]
) -> None:  # check to handle nested params with the same name?
    properties = model_dict.get("properties", {})

    for field_name, details in properties.items():
        parameter = ET.SubElement(root, "parameter")
        name = ET.SubElement(parameter, "name")
        name.text = field_name
        type_element = ET.SubElement(parameter, "type")
        type_element.text = details.get(
            "type", "unknown"
        )  # Might be better to fail here if there is no type since pydantic models require types
        param_description = ET.SubElement(parameter, "description")
        param_description.text = f"The {field_name} of the {model_dict['title']} model"

        if (
            isinstance(details, dict) and "$ref" in details
        ):  # Checking if there are nested params
            nested_params = ET.SubElement(parameter, "parameters")
            _add_params(
                nested_params,
                _resolve_reference(references, details["$ref"]),
                references,
            )


def _resolve_reference(references: Dict[str, Any], reference: str) -> Dict[str, Any]:
    parts = reference.split("/")[2:]  # Remove "#" and "$defs"
    for part in parts:
        references = references[part]
    return references


def extract_xml(content: str) -> str:
    """Extracts XML content in Anthropic's schema from a string."""
    pattern = r"<function_calls>.*?</function_calls>"
    matches = re.findall(pattern, content, re.DOTALL)
    return "".join(matches)


# todo: make function better (edge cases, robustness, etc.)
def xml_to_model(model: Type[T], xml_string: str) -> T:
    """Converts XML in Anthropic's schema to an instance of the provided class."""
    parsed_xml = xmltodict.parse(xml_string)
    model_dict = parsed_xml["function_calls"]["invoke"]["parameters"]
    return model(**model_dict)
