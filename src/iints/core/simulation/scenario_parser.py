# src/simulation/scenario_parser.py

from typing import List, Dict, Any, Tuple

from iints.core.simulator import StressEvent
from iints.validation import load_scenario, scenario_to_payloads, build_stress_events

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
        scenario = load_scenario(file_path)
    except Exception as e:
        raise ValueError(f"Invalid scenario file {file_path}: {e}")

    metadata = {
        "name": scenario.scenario_name,
        "description": scenario.description or "",
        "version": scenario.scenario_version,
        "source_file": file_path,
    }

    payloads = scenario_to_payloads(scenario)
    events: List[StressEvent] = build_stress_events(payloads)
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
