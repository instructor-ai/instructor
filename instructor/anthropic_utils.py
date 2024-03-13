import re
import xmltodict
from pydantic import BaseModel
import xml.etree.ElementTree as ET
from typing import Type, Any, Dict, TypeVar

T = TypeVar("T", bound=BaseModel)


def json_to_xml(model: Type[BaseModel]) -> str:
    """Takes a Pydantic model and returns XML format for Anthropic function calling."""
    model_dict = model.model_json_schema()
    
    #TODO: model_dict structure is different for Partial classes during streaming, need to adjust for this for better prompting + List types

    root = ET.Element("tool_description")
    tool_name = ET.SubElement(root, "tool_name")
    tool_name.text = model_dict.get("title", "Unknown")
    description = ET.SubElement(root, "description")
    description.text = (
        "This is the function that must be used to construct the response."
    )
    parameters = ET.SubElement(root, "parameters")
    references = model_dict.get("$defs", {})
    list_type_found = _add_params(parameters, model_dict, references)
    
    if list_type_found: # Need to append to system prompt for List type handling
        return ET.tostring(root, encoding="unicode") + "\nFor any List[] types, include multiple <$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME> tags for each item in the list."
    else:
        return ET.tostring(root, encoding="unicode")


def _add_params(
    root: ET.Element, model_dict: Dict[str, Any], references: Dict[str, Any]
) -> bool: # Return value indiciates if we ever came across a param with type List
    # TODO: check handling of nested params with the same name
    properties = model_dict.get("properties", {})
    list_found = False

    for field_name, details in properties.items():
        parameter = ET.SubElement(root, "parameter")
        name = ET.SubElement(parameter, "name")
        name.text = field_name
        type_element = ET.SubElement(parameter, "type")
        
        if details.get("type") == "array":
            type_element.text = f"List[{details['title']}]"
            list_found = True
        else:
            type_element.text = details.get(
                "type", "unknown"
            )  # Might be better to fail here if there is no type since pydantic models require types

        param_description = ET.SubElement(parameter, "description")
        param_description.text = f"The {field_name} of the {model_dict['title']} model"

        if (
            isinstance(details, dict) and "$ref" in details
        ):  # Checking if there are nested params
            nested_params = ET.SubElement(parameter, "parameters")
            list_found |= _add_params(
                nested_params,
                _resolve_reference(references, details["$ref"]),
                references,
            )
        elif details.get("type") == "array": # Handling for List[] type
            nested_params = ET.SubElement(parameter, "parameters")
            list_found |= _add_params(
                nested_params,
                _resolve_reference(references, details["items"]["$ref"]),
                references,
            )
    
    return list_found

def _resolve_reference(references: Dict[str, Any], reference: str) -> Dict[str, Any]:
    parts = reference.split("/")[2:]  # Remove "#" and "$defs"
    for part in parts:
        references = references[part]
    return references


def extract_xml(content: str) -> str: # Currently assumes 1 function call only
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

class AnthropicContextManager:
    def __init__(self, create_func, *args, **kwargs):
        self.create_func = create_func
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        self.result = self.create_func(*self.args, **self.kwargs)
        return self.result

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.result, 'close'):
            self.result.close()