import json
import xml.etree.ElementTree as ET

def json_to_xml(json_str: str) -> str:
    """Takes a JSON string representing a Pydantic model and returns XML format for Anthropic function calling."""
    try:
        model_dict = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON string.")
    
    if "properties" not in model_dict:
        raise ValueError("JSON structure is not as expected.")

    root = ET.Element("tool_description")
    tool_name = ET.SubElement(root, "tool_name")
    tool_name.text = model_dict.get("title", "Unknown")
    
    description = ET.SubElement(root, "description")
    description.text = "This is the function that must be used to construct the response."
    
    parameters = ET.SubElement(root, "parameters")

    properties = model_dict.get("properties", {})
    for field_name, details in properties.items():
        parameter = ET.SubElement(parameters, "parameter")
        
        name = ET.SubElement(parameter, "name")
        name.text = field_name
        
        type_element = ET.SubElement(parameter, "type")
        type_element.text = details.get("type", "unknown") # might be better to fail here if there is no type since pydantic models require types
        
        param_description = ET.SubElement(parameter, "description")
        param_description.text = f"The {field_name} of the {model_dict.get('title', 'Unknown')} model"
    
    return ET.tostring(root, encoding="unicode") # todo: handle non-unicode

# todo: make function better (edge cases, robustness, etc.)
# todo: super primitive parsing, make better
def extract_xml(content: str) -> str:
    start_index = content.find('<')
    end_index = content.rfind('>') + 1
    return content[start_index:end_index]

# todo: make function better (edge cases, robustness, etc.)
def xml_to_model(model, xml_string):
    root = ET.fromstring(xml_string)
    parameters = {}
    for param in root.find('.//parameters'):
        # todo: this assumes all values are strings, fix
        field_type = model.__annotations__.get(param.tag)
        if field_type:
            parameters[param.tag] = field_type(param.text)
    return model(**parameters)