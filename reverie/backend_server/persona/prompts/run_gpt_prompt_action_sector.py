from persona.prompt_template.InferenceStrategy import JSONType, OutputType, functor, InferenceStrategy
from persona.common import with_json

@functor
class run_gpt_prompt_action_sector(InferenceStrategy):
  output_type = OutputType.JSON
  config =  {
    "temperature": 0.3,
  }
  prompt = """
    We need to choose an appropriate Sector for the task at hand.

    * Stay in the current sector if the activity can be done there. Only go out if the activity needs to take place in another place.
    * Must be one of the sectors from "All Sectors," verbatim. It must be a Sector, and not an Arena.
    * If none of those fit very well, we must still choose the one that's the closest fit.
    * Return the answer as a JSON object with a single key "area". The value is the chosen area name.

    Sam Kim lives in the "Sam Kim's house" Sector that has the following Arenas: ["Sam Kim's room", "bathroom", "kitchen"]
    Sam Kim is currently in the "Sam Kim's house" Sector that has the following Arenas: ["Sam Kim's room", "bathroom", "kitchen"]
    All Sectors: ["Sam Kim's house", "The Rose and Crown Pub", "Hobbs Cafe", "Oak Hill College", "Johnson Park", "Harvey Oak Supply Store", "The Willows Market and Pharmacy"].
    For performing the "taking a walk" Action, Sam Kim should go to the following Sector:
    {{"area": "Johnson Park"}}
    ---
    Jane Anderson lives in the "Oak Hill College Student Dormitory" Sector that has the following Arenas: ["Jane Anderson's room"]
    Jane Anderson is currently in the "Oak Hill College" Sector that has the following Arenas: ["classroom", "library"]
    All Sectors: ["Oak Hill College Student Dormitory", "The Rose and Crown Pub", "Hobbs Cafe", "Oak Hill College", "Johnson Park", "Harvey Oak Supply Store", "The Willows Market and Pharmacy"].
    For performing the "eating dinner" Action, Jane Anderson should go to the following Sector:
    {{"area": "Hobbs Cafe"}}
    ---
    {name} lives in the {living_sector_json} Sector that has the following Arenas: {living_sector_arenas_json}.
    {name} is currently in the {current_sector_json} Sector that has the following Arenas: {current_sector_arenas_json}.
    All Sectors: {all_sectors_json}.
    Pick the Sector for performing {name}'s current activity.
    * Stay in the current sector if the activity can be done there. Only go out if the activity needs to take place in another place.
    * Must be one of the sectors from "All Sectors," verbatim. It must be a Sector, and not an Arena.
    * If none of those fit very well, we must still choose the one that's the closest fit.
    * Return the answer as a JSON object with a single key "area". The value is the chosen area name.
    For performing the {action_description_json} Action, {name} should go to the following Sector:
  """

  def prepare_context(self, action_description, persona, maze):
    world_area = maze.access_tile(persona.scratch.curr_tile)['world']
    path_to_living_sector = persona.scratch.living_area.split(":")[:2]
    path_to_current_sector = [
      world_area,
      maze.access_tile(persona.scratch.curr_tile)['sector'],
    ]
    known_sectors = persona.s_mem.get_str_accessible_sectors(world_area).split(", ")

    return with_json([
      'living_sector',
      'living_sector_arenas',
      'current_sector',
      'current_sector_arenas',
      'all_sectors',
      'action_description'
    ])({
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
  
  def validate_json(self, json: JSONType):
    if "area" not in json:
      return "Missing area name"
    if json["area"] not in self.context["all_sectors"]:
      if json["area"] in self.context["living_sector_arenas"] or json["area"] in self.context["current_sector_arenas"]:
        return "Arena name was returned instead of the Sector name"
      else:
        return f"Specified Sector doesn't exist or isn't available to {self.context['persona'].scratch.get_str_firstname()}"
  
  def extract_json(self, json: JSONType):
    return json["area"]
  
  def fallback(self, action_description, persona, maze):
    return maze.access_tile(persona.scratch.curr_tile)['sector']
