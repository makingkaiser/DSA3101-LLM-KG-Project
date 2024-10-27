import os
import json
import re
from openai import OpenAI


# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

eval_prompt = """Your task is to extract entities, as well as define any relationships between them. The entity types to extract are :

Person/Organization
Role
Location
Product/Service

The head-tail relationships (if present) to be extracted are:
Person-Organization "AFFILIATED_WITH"
Person-Role Association "HAS_ROLE"
Person-Product/Service "INVOLVED_WITH"
Organization-Location "LOCATED_AT"

JSON Format with dummy examples:

{{
  "entities": {{
    "persons": [
      {{
        "name": "John Doe"
      }}
    ],
    "organizations": [
      {{
        "name": "Acme Corp"
      }}
    ],
    "roles": [
      {{
        "title": "CEO"
      }}
    ],
    "locations": [
      {{
        "name": "New York"
      }}
    ],
    "products_services": [
      {{
        "name": "Widget X"
      }}
    ]
  }},
  "relationships": [
    {{
      "type": "AFFILIATED_WITH",
      "person": "John Doe",
      "organization": "Acme Corp"
    }},
    {{
      "type": "HAS_ROLE",
      "person": "John Doe",
      "role": "CEO"
    }},
    {{
      "type": "INVOLVED_WITH",
      "person": "John Doe",
      "product_service": "Widget X"
    }},
    {{
      "type": "LOCATED_AT",
      "organization": "Acme Corp",
      "location": "New York"
    }}
  ]
}}

TEXT:
{text}

"""

def extract_json_from_response(response_text):
    """
    Extracts JSON from response text, handling code blocks and multiline content.
    """
    # First try to find JSON within code blocks
    code_block_pattern = r'```json\s*([\s\S]*?)\s*```'
    code_block_match = re.search(code_block_pattern, response_text)
    
    if code_block_match:
        try:
            json_str = code_block_match.group(1)
            json_object = json.loads(json_str)
            return json_object
        except json.JSONDecodeError:
            pass
    
    # If no code block or invalid JSON, try general JSON pattern
    json_pattern = r'(?s)\{.*?\}(?=\s*$)'  # (?s) enables dot to match newlines
    json_match = re.search(json_pattern, response_text)
    
    if json_match:
        json_str = json_match.group(0)
        json_object = json.loads(json_str)
        return json_object
    
    # Return None if no valid JSON is found
    return None

    

def eval_text_files(output_folder):
    # Check if the output folder exists
    if not os.path.exists(output_folder):
        print(f"The folder '{output_folder}' does not exist.")
        return

    # List all files in the output folder
    files = os.listdir(output_folder)

    # Filter out only .txt files
    txt_files = [f for f in files if f.endswith('.txt')]

    # evaluate the content of each .txt file
    for txt_file in txt_files:
        txt_file_path = os.path.join(output_folder, txt_file)
        with open(txt_file_path, 'r') as file:
            text_content = file.read()
        
        populated_prompt = eval_prompt.format(
            text=text_content
        )
        completion = client.chat.completions.create(
        model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
        messages=[
            {"role": "user", "content": populated_prompt},
        ],
        )
        response_text = completion.choices[0].message.content
        print(response_text)
        _, extracted_json = extract_json_from_response(response_text)
        base_name, _ = os.path.splitext(txt_file)
        evaluated_filename = f"{base_name}_evaluated.json"
        evaluated_file_path = os.path.join(output_folder, evaluated_filename)

        # Save the evaluated content to the new JSON file
        with open(evaluated_file_path, 'w') as eval_file:
            json.dump(extracted_json, eval_file, indent=4)

          

# Example usage of the dummy function
# eval_text_files("output_test")

