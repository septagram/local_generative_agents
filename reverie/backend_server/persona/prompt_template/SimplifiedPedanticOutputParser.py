"""
 Copyright 2024 Igor Novikov

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import json
from typing import Any, Dict, List, Optional, Union
from langchain.schema import AIMessage
from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation
from langchain_core.exceptions import OutputParserException
from langchain_core.pydantic_v1 import ValidationError

JSONType = Union[Dict[str, Any], List[Any]]

def find_and_parse_json(text: str) -> JSONType:
  OpeningToClosingBrace = {"{": "}", "[": "]"}
  start_indices = [text.find('{'), text.find('[')]
  start_index = min(i for i in start_indices if i != -1)

  if start_index == -1:
    return None
  
  closing_brace = OpeningToClosingBrace[text[start_index]]
  text = text[start_index:]
  last_error = ValueError("Expected JSON object/array")

  while True:
    end_index = text.rfind(closing_brace)
    if end_index == -1:
      raise last_error
    try:
      return json.loads(text[:end_index+1])
    except json.JSONDecodeError as e:
      last_error = e
      text = text[:end_index]

class SimplifiedPydanticOutputParser(PydanticOutputParser):
  context: Dict[str, Any] = {}

  def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
    json_object = find_and_parse_json(result[0].text)
    try:
      json_object['context'] = self.context
      return self.pydantic_object.parse_obj(json_object)
    except ValidationError as errors_object:
      msg = '\n'.join(
        [
          f"- {error['msg']}: (root){''.join(map(lambda field: f'[{field}]' if field is int else f'.{field}', error['loc']))}"
          for error in errors_object.errors()
        ]
      )
      raise OutputParserException(msg, llm_output=json_object)

  def get_format_instructions(self) -> str:
    return f"The output must be {self._field_format_instructions(self.pydantic_object.schema())}"
  
  def _schema_by_ref(self, ref: str) -> Dict:
    if ref.startswith('#/definitions/'):
      return self.pydantic_object.schema()['definitions'][ref.split('/')[-1]]
    else:
      raise ValueError(f"This ref format is unsupported: {ref}")
    
  def _object_format_instructions(self, object_schema: Dict, indent: int = 0):
    fields_instructions = [
      (
        '  ' * (indent + 1) +
        json.dumps(field_name) +
        ': ' +
        self._field_format_instructions(field_schema, field_name, object_schema, indent + 1) +
        ','
      )
      for field_name, field_schema in object_schema['properties'].items()
      if field_name != 'context' or indent > 0
    ]
    complete_instructions = '\n'.join(['{'] + fields_instructions + ['  ' * indent + '}'])
    return complete_instructions

  def _field_format_instructions(
    self,
    field_schema: Dict,
    field_name: Optional[str] = None,
    object_schema: Optional[Dict] = None,
    indent: int = 0,
    is_array_item: bool = False
  ):
    ref_schema = self._schema_by_ref(field_schema['$ref']) if '$ref' in field_schema else field_schema
    ref_type = ref_schema['type']
    if is_array_item:
      type = f"{ref_type}s"
    else:
      article = 'an' if ref_type[0] in 'aeiou' else 'a'
      type = f"{article} {ref_type}"

    description = required = recursive_instructions = None
    if 'description' in field_schema:
      description = field_schema['description'] + ','
    if object_schema and field_name and field_name in object_schema.get('required', []):
      required = '(required)'
    if ref_type == 'array':
      recursive_instructions = f"of {self._field_format_instructions(ref_schema['items'], indent=indent, is_array_item=True)}"
    elif ref_type == 'object':
      recursive_instructions = f"like this: {self._object_format_instructions(ref_schema, indent)}"
    # elif ref_type == 'string' and ref_schema.get('format', '') in string_format_instructions:
    #   recursive_instructions = f'like "{string_format_instructions[ref_schema["format"]]}"'
    return ' '.join(filter(None, [required, description, type, recursive_instructions]))
