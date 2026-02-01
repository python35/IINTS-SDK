# src/algorithm/discovery.py

import os
import importlib
import inspect
from pathlib import Path
from typing import Dict, Type

from iints.api.base_algorithm import InsulinAlgorithm

def discover_algorithms() -> Dict[str, Type[InsulinAlgorithm]]:
    """
    Dynamically discovers and loads all algorithm classes that inherit from InsulinAlgorithm.

    It scans the `src/algorithm` directory and its subdirectories (like `user`).

    Returns:
        A dictionary mapping the algorithm's display name to its class type.
        e.g., {"PID Controller": PIDController, "My Custom Algo": MyCustomAlgo}
    """
    algorithms: Dict[str, Type[InsulinAlgorithm]] = {}
    
    # The root directory for algorithm discovery
    # We start from 'src' to make the imports work correctly (e.g., src.algorithm.pid_controller)
    root_path = Path(__file__).parent.parent.parent # Corrected to point to project root
    algorithm_dir = Path(__file__).parent
    
    for root, _, files in os.walk(algorithm_dir):
        for filename in files:
            # Consider only Python files, excluding __init__ and base files
            if filename.endswith(".py") and not filename.startswith(("_", "base_")):
                
                # Construct the module path for importlib
                # e.g., /path/to/project/src/algorithm/user/my_algo.py
                # becomes -> src.algorithm.user.my_algo
                
                module_path = Path(root) / filename
                # Get relative path from the 'src' directory
                relative_path = module_path.relative_to(root_path)
                # Convert path to module name (e.g., algorithm/user/my_algo.py -> algorithm.user.my_algo)
                module_name = str(relative_path).replace(os.sep, '.')[:-3]

                try:
                    module = importlib.import_module(module_name)
                    
                    # Find all classes in the module that are subclasses of InsulinAlgorithm
                    for _, member_class in inspect.getmembers(module, inspect.isclass):
                        # Ensure the class is defined in this module (not imported)
                        # and is a subclass of InsulinAlgorithm (but not the base class itself)
                        if (
                            issubclass(member_class, InsulinAlgorithm) 
                            and member_class is not InsulinAlgorithm
                            and member_class.__module__ == module_name
                        ):
                            # Instantiate the class to get its metadata
                            try:
                                instance = member_class()
                                metadata = instance.get_algorithm_metadata()
                                display_name = metadata.name
                                
                                # Avoid overwriting algorithms with the same display name
                                if display_name in algorithms:
                                    print(f"Warning: Duplicate algorithm name '{display_name}' found. Skipping {member_class.__name__}.")
                                else:
                                    algorithms[display_name] = member_class
                            except Exception as e:
                                print(f"Warning: Could not instantiate or get metadata for {member_class.__name__}. Error: {e}")
                
                except ImportError as e:
                    print(f"Warning: Could not import module {module_name}. Error: {e}")

    return algorithms

if __name__ == '__main__':
    # A simple test to demonstrate the discovery mechanism
    print("Discovering all available IINTS-AF algorithms...")
    discovered_algos = discover_algorithms()
    
    if not discovered_algos:
        print("No algorithms found.")
    else:
        print("\nFound the following algorithms:")
        for name, algo_class in discovered_algos.items():
            print(f"- '{name}' (Class: {algo_class.__name__})")
    
    # Example of instantiating and using a discovered algorithm
    if "Template Algorithm" in discovered_algos:
        print("\n--- Testing 'Template Algorithm' ---")
        TemplateAlgoClass = discovered_algos["Template Algorithm"]
        template_instance = TemplateAlgoClass()
        metadata = template_instance.get_algorithm_metadata()
        print(f"Successfully instantiated. Author: {metadata.author}")
