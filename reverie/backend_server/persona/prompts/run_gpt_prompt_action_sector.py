"""
 Copyright 2024 Joon Sung Park, Igor Novikov

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

from persona.prompt_template.ResponseModel import ResponseModel, Field

from persona.prompt_template.InferenceStrategy import functor, InferenceStrategy
from persona.common import with_json

class ActionSectorResponse(ResponseModel):
  sector: str = Field(..., description="the Sector name that this character should go to, must be an exact match")

with_json_transformer = with_json([
  'living_sector',
  'living_sector_arenas',
  'current_sector',
  'current_sector_arenas',
  'all_sectors',
  'action_description',
])

@functor
class run_gpt_prompt_action_sector(InferenceStrategy):
  output_type = ActionSectorResponse
  config = {
    "temperature": 0.3,
  }
  example_prompt = """
    {name} lives in the {living_sector_json} Sector that has the following Arenas: {living_sector_arenas_json}.
    {name} is currently in the {current_sector_json} Sector that has the following Arenas: {current_sector_arenas_json}.
    All Sectors: {all_sectors_json}.
    Which Sector should {name} go to for performing the {action_description_json} Action?
  """
  prompt = """
    We need to choose an appropriate Sector for the task at hand.

    * Stay in the current sector if the activity can be done there. Only go out if the activity needs to take place in another place.
    * Must be one of the Sectors from "All Sectors," verbatim. It must be a Sector, and not an Arena.
    * If none of those fit very well, we must still choose the one that's the closest fit.

    {format_instructions}

    Here are some examples:

    {examples}

    Now, let's consider {name}:

    {example_prompt}
  """

  def prepare_context(self, action_description, persona, maze):
    world_area = maze.access_tile(persona.scratch.curr_tile)['world']
    path_to_living_sector = persona.scratch.living_area.split(":")[:2]
    path_to_current_sector = [
      world_area,
      maze.access_tile(persona.scratch.curr_tile)['sector'],
    ]
    known_sectors = persona.s_mem.get_str_accessible_sectors(world_area).split(", ")

    return with_json_transformer({
      "persona": persona,
      "name": persona.scratch.get_str_name(),
      "action_description": action_description,
      "living_sector": path_to_living_sector[1],
      "living_sector_arenas": persona.s_mem.get_array_accessible_sector_arenas(
        ":".join(path_to_living_sector)
      ),
      "current_sector": path_to_current_sector[1],
      "current_sector_arenas": persona.s_mem.get_array_accessible_sector_arenas(
        ":".join(path_to_current_sector)
      ),
      "all_sectors": [sector for sector in known_sectors if "'s house" not in sector or persona.scratch.last_name in sector],
    })
  
  def postprocess(self, output: ActionSectorResponse):
    if output.sector not in self.context["all_sectors"]:
      available_sectors = f"Select one of {self.context['all_sectors_json']}, verbatim."
      if output.sector in self.context["living_sector_arenas"] or output.sector in self.context["current_sector_arenas"]:
        raise ValueError(f"Arena name was returned instead of the Sector name. {available_sectors}")
      else:
        raise ValueError(f"Specified Sector doesn't exist or isn't available to {self.context['persona'].scratch.get_str_firstname()}. {available_sectors}")
    return output.sector
  
  def fallback(self, action_description, persona, maze):
    return maze.access_tile(persona.scratch.curr_tile)['sector']

  examples = [
    with_json_transformer({
      "name": "Sam Kim",
      "action_description": "taking a walk",
      "living_sector": "Sam Kim's House",
      "living_sector_arenas": ["Sam Kim's room", "bathroom", "kitchen"],
      "current_sector": "Sam Kim's House",
      "current_sector_arenas": ["Sam Kim's room", "bathroom", "kitchen"],
      "all_sectors": [
        "Sam Kim's house",
        "The Rose and Crown Pub",
        "Hobbs Cafe",
        "Oak Hill College",
        "Johnson Park",
        "Harvey Oak Supply Store",
        "The Willows Market and Pharmacy",
      ],
      "output": ActionSectorResponse(sector="Johnson Park").json()
    }),
    with_json_transformer({
      "name": "Jane Anderson",
      "action_description": "eating dinner",
      "living_sector": "Oak Hill College Student Dormitory",
      "living_sector_arenas": ["Jane Anderson's room"],
      "current_sector": "Oak Hill College",
      "current_sector_arenas": ["classroom", "library"],
      "all_sectors": [
        "Sam Kim's house",
        "The Rose and Crown Pub",
        "Hobbs Cafe",
        "Oak Hill College",
        "Johnson Park",
        "Harvey Oak Supply Store",
        "The Willows Market and Pharmacy",
      ],
      "output": ActionSectorResponse(sector="Johnson Park").json()
    }),
  ]
