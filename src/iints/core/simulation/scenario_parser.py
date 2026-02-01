# src/simulation/scenario_parser.py

import json
from typing import List, Dict, Any, Tuple

from iints.core.simulator import StressEvent

def parse_scenario(file_path: str) -> Tuple[Dict[str, Any], List[StressEvent]]:
    """
    Parses a scenario from a JSON file.

    Args:
        file_path: The path to the scenario JSON file.

    Returns:
        A tuple containing:
        - A dictionary with scenario metadata (name, description).
        - A list of StressEvent objects.
        
    Raises:
        ValueError: If the file format or content is invalid.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Scenario file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in scenario file {file_path}: {e}")

    # Validate top-level keys
    if not all(k in data for k in ["scenario_name", "events"]):
        raise ValueError("Scenario file must contain 'scenario_name' and 'events' keys.")
        
    metadata = {
        "name": data.get("scenario_name", "Unnamed Scenario"),
        "description": data.get("description", ""),
        "source_file": file_path
    }
    
    events: List[StressEvent] = []
    if not isinstance(data["events"], list):
        raise ValueError("'events' key must contain a list.")

    for i, event_data in enumerate(data["events"]):
        try:
            # The 'carb_error' type is just a meal the algorithm isn't told about.
            # We can model this by mapping it to a 'meal' event with a reported_value of 0.
            event_type = event_data["type"]
            if event_type == "carb_error":
                event_data["reported_value"] = 0
                event_type = "meal" # Treat as a meal for the StressEvent

            event = StressEvent(
                start_time=int(event_data["time"]),
                event_type=event_type,
                value=event_data.get("value"),
                reported_value=event_data.get("reported_value"),
                absorption_delay_minutes=int(event_data.get("absorption_delay_minutes", 0)),
                duration=int(event_data.get("duration", 0))
            )
            events.append(event)
        except KeyError as e:
            raise ValueError(f"Missing required key {e} in event #{i+1} in {file_path}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value in event #{i+1} in {file_path}: {e}")

    return metadata, events

if __name__ == '__main__':
    # A simple test to demonstrate the scenario parser
    print("--- Testing Scenario Parser ---")
    
    # Use the example scenario created earlier
    example_path = "scenarios/example_scenario.json"
    
    try:
        print(f"Parsing scenario file: {example_path}")
        scenario_metadata, scenario_events = parse_scenario(example_path)
        
        print("\nSuccessfully parsed scenario:")
        print(f"  Name: {scenario_metadata['name']}")
        print(f"  Description: {scenario_metadata['description']}")
        
        print("\nEvents:")
        for ev in scenario_events:
            print(f"  - {ev}")
            
    except ValueError as e:
        print(f"\nError parsing scenario: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
