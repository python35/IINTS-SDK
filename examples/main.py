#!/usr/bin/env python3
"""
IINTS-AF: Intelligent Insulin Titration System
Autonomous Learning Framework for Diabetes Care Research

Main terminal interface for algorithm analysis and research.
For interactive Pro Terminal, use: python iints_pro.py --interactive
"""

import sys
import os
import subprocess

class IINTSTerminalApp:
    def __init__(self):
        self.scenarios = ['Standard_Meal', 'Unannounced_Meal', 'Hyperglycemia_Correction', 'Noisy_Standard_Meal']
        self.algorithms = ['rule_based', 'lstm', 'hybrid']
        self.running = True
        
    def clear_screen(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        
    def show_header(self):
        # Display IINTS logo
        logo = """+------------------+
|                  |
|      I I N       |
|                  |
|       T S        |
|                  |
+------------------+"""
        print("\033[32m" + logo + "\033[0m")
        print("="*80)
        print("IINTS-AF: Intelligent Insulin Titration System")
        print("Autonomous Learning Framework for Diabetes Care Research")
        print("="*80)
        
    def show_main_menu(self):
        self.clear_screen()
        self.show_header()
        print("\nMAIN MENU")
        print("-" * 40)
        print("[1] Algorithm Analysis")
        print("[2] Battle Mode (Multi-Algorithm)")
        print("[3] Legacy Pump Emulation")
        print("[4] Data Validation")
        print("[5] View Results")
        print("[6] Pro Terminal (Interactive)")
        print("[7] Settings")
        print("[0] Exit")
        print("-" * 40)
        
    def show_algorithm_menu(self):
        self.clear_screen()
        self.show_header()
        print("\nALGORITHM ANALYSIS")
        print("-" * 40)
        print("[1] Basic Analysis (Single Run)")
        print("[2] Comprehensive Analysis (All Features)")
        print("[3] Data Validation Test")
        print("[4] Autonomous Learning Test")
        print("[5] Clinical Reality Check")
        print("[6] Custom Analysis")
        print("[0] Back to Main Menu")
        print("-" * 40)
        
    def show_battle_menu(self):
        self.clear_screen()
        self.show_header()
        print("\nBATTLE MODE")
        print("-" * 40)
        print("[1] Run Battle (LSTM vs PID vs Hybrid)")
        print("[2] Select Algorithms to Compare")
        print("[3] Custom Battle Configuration")
        print("[4] View Previous Battles")
        print("[0] Back to Main Menu")
        print("-" * 40)
        
    def show_legacy_menu(self):
        self.clear_screen()
        self.show_header()
        print("\nLEGACY PUMP EMULATION")
        print("-" * 40)
        print("[1] Medtronic 780G Analysis")
        print("[2] Tandem Control-IQ Analysis")
        print("[3] Omnipod 5 Analysis")
        print("[4] Compare All Legacy Pumps")
        print("[5] Legacy vs New AI Comparison")
        print("[0] Back to Main Menu")
        print("-" * 40)
        
    def select_scenario(self):
        print("\nSCENARIO SELECTION:")
        for i, scenario in enumerate(self.scenarios, 1):
            print(f"[{i}] {scenario.replace('_', ' ')}")
        
        while True:
            try:
                choice = int(input("\nEnter scenario number: ")) - 1
                if 0 <= choice < len(self.scenarios):
                    return self.scenarios[choice]
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
                
    def select_algorithm(self):
        print("\nALGORITHM SELECTION:")
        for i, algo in enumerate(self.algorithms, 1):
            print(f"[{i}] {algo.replace('_', ' ').title()}")
        
        while True:
            try:
                choice = int(input("\nEnter algorithm number: ")) - 1
                if 0 <= choice < len(self.algorithms):
                    return self.algorithms[choice]
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
                
    def select_battle_algorithms(self):
        """Select multiple algorithms for battle mode"""
        print("\nSELECT ALGORITHMS FOR BATTLE (comma-separated, e.g., 1,2,3):")
        for i, algo in enumerate(self.algorithms, 1):
            print(f"[{i}] {algo.replace('_', ' ').title()}")
        
        # Add legacy pumps
        print("[4] Medtronic 780G")
        print("[5] Tandem Control-IQ")
        print("[6] Omnipod 5")
        
        while True:
            try:
                choice = input("\nEnter numbers: ")
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                selected = []
                for idx in indices:
                    if 0 <= idx < len(self.algorithms):
                        selected.append(self.algorithms[idx])
                    elif idx == 3:
                        selected.append('medtronic_780g')
                    elif idx == 4:
                        selected.append('tandem_controliq')
                    elif idx == 5:
                        selected.append('omnipod_5')
                if selected:
                    return selected
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter valid numbers.")
                
    def run_script(self, script_name, args=None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(project_root, "scripts", script_name)
        
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
            
        print(f"\nExecuting: {script_name}")
        print("=" * 60)
        
        try:
            result = subprocess.run(cmd, check=True, cwd=project_root)
            print("=" * 60)
            print("EXECUTION COMPLETED SUCCESSFULLY")
        except subprocess.CalledProcessError as e:
            print("=" * 60)
            print(f"EXECUTION FAILED - Exit code: {e.returncode}")
        except FileNotFoundError:
            print(f"ERROR: Script not found - {script_path}")
            
        input("\nPress Enter to continue...")
        
    def run_battle_mode(self):
        """Run the new battle mode with the BattleRunner"""
        print("\nBATTLE MODE - Multi-Algorithm Comparison")
        print("=" * 60)
        
        scenario = self.select_scenario()
        algorithms = self.select_battle_algorithms()
        
        print(f"\nBATTLE CONFIGURATION:")
        print(f"Scenario: {scenario}")
        print(f"Algorithms: {', '.join(algorithms)}")
        
        # Run battle using Python import
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Create and run battle script
        battle_script = f'''
import sys
sys.path.insert(0, "{project_root}")

import numpy as np
import pandas as pd
from iints.core.algorithms.battle_runner import BattleRunner
from iints.core.algorithms.pid_controller import PIDController
from iints.core.algorithms.correction_bolus import CorrectionBolus
from iints.core.algorithms.hybrid_algorithm import HybridInsulinAlgorithm
from iints.emulation.medtronic_780g import Medtronic780GEmulator
from iints.emulation.tandem_controliq import TandemControlIQEmulator
from iints.emulation.omnipod_5 import Omnipod5Emulator

# Generate sample data
np.random.seed(42)
n_points = 288  # 24 hours
time = np.arange(n_points) * 5
glucose = 120 + 30 * np.sin(time / (24 * 12 / (2 * np.pi))) + np.random.normal(0, 15, n_points)
glucose = np.clip(glucose, 40, 400)

data = pd.DataFrame({{
    'timestamp': time,
    'glucose': glucose,
    'carbs': np.random.choice([0, 30, 60], n_points, p=[0.8, 0.15, 0.05]),
    'insulin': 0.0
}})

# Create battle runner
runner = BattleRunner()

# Register algorithms based on selection
algorithm_map = {{
    'rule_based': lambda: CorrectionBolus(),
    'lstm': lambda: HybridInsulinAlgorithm(),
    'hybrid': lambda: HybridInsulinAlgorithm(),
    'pid': lambda: PIDController(),
    'medtronic_780g': lambda: Medtronic780GEmulator(),
    'tandem_controliq': lambda: TandemControlIQEmulator(),
    'omnipod_5': lambda: Omnipod5Emulator()
}}

for algo_name in {algorithms}:
    if algo_name in algorithm_map:
        runner.register_algorithm(algo_name.title(), algorithm_map[algo_name]())
    else:
        try:
            runner.register_algorithm(algo_name.title(), algorithm_map['rule_based']())
        except:
            pass

# Run battle
report = runner.run_battle(data, scenario="{scenario}")

# Save report
runner.save_report(report, "{project_root}/results/reports/battle_report.json")

# Print summary
print(report.get_summary())
'''
        
        # Execute battle script
        cmd = [sys.executable, '-c', battle_script]
        print("\nSTARTING BATTLE...")
        
        try:
            result = subprocess.run(cmd, cwd=project_root, capture_output=False)
            print("\n[OK] Battle completed!")
            print("Report saved to: results/reports/battle_report.json")
        except Exception as e:
            print(f"[FAIL] Battle failed: {e}")
        
        input("\nPress Enter to continue...")
        
    def handle_basic_analysis(self):
        scenario = self.select_scenario()
        algorithm = self.select_algorithm()
        
        print(f"\nBASIC ANALYSIS CONFIGURATION:")
        print(f"Scenario: {scenario}")
        print(f"Algorithm: {algorithm}")
        
        print("\nADDITIONAL OPTIONS:")
        no_safety = input("Disable safety supervisor? [y/N]: ").lower() == 'y'
        benchmark = input("Enable hardware benchmarking? [y/N]: ").lower() == 'y'
        
        args = ['--scenario', scenario, '--algorithm', algorithm]
        if no_safety:
            args.append('--no-safety')
        if benchmark:
            args.append('--benchmark')
            
        self.run_script('run_final_analysis.py', args)
        
    def handle_algorithm_menu(self):
        while True:
            self.show_algorithm_menu()
            choice = input("Select option: ")
            
            if choice == '1':
                self.handle_basic_analysis()
            elif choice == '2':
                print("\nExecuting comprehensive analysis...")
                self.run_script('run_comprehensive_analysis.py')
            elif choice == '3':
                print("\nExecuting data validation test...")
                self.run_script('test_validation.py')
            elif choice == '4':
                print("\nExecuting autonomous learning test...")
                self.run_script('test_autonomous_learning.py')
            elif choice == '5':
                print("\nExecuting clinical reality check...")
                self.run_script('clinical_reality_check.py')
            elif choice == '6':
                self.handle_basic_analysis()  # Reuse for custom
            elif choice == '0':
                break
            else:
                print("Invalid choice. Press Enter to continue...")
                input()
                
    def handle_battle_menu(self):
        while True:
            self.show_battle_menu()
            choice = input("Select option: ")
            
            if choice == '1':
                self.run_battle_mode()
            elif choice == '2':
                print("\nAlgorithm selection...")
                self.select_battle_algorithms()
            elif choice == '3':
                print("\nCustom battle configuration - coming soon")
            elif choice == '4':
                print("\nPrevious battles - coming soon")
            elif choice == '0':
                break
            else:
                print("Invalid choice. Press Enter to continue...")
                input()
                
    def handle_legacy_menu(self):
        while True:
            self.show_legacy_menu()
            choice = input("Select option: ")
            
            if choice == '1':
                print("\nMedtronic 780G Analysis")
                print("-" * 40)
                print("Running Medtronic 780G emulation...")
                # Import and run demo
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                legacy_script = f'''
import sys
sys.path.insert(0, "{project_root}")
from iints.emulation.medtronic_780g import Medtronic780GEmulator
emu = Medtronic780GEmulator()
print(emu.get_algorithm_personality())
'''
                subprocess.run([sys.executable, '-c', legacy_script])
                
            elif choice == '2':
                print("\nTandem Control-IQ Analysis")
                print("-" * 40)
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                legacy_script = f'''
import sys
sys.path.insert(0, "{project_root}")
from iints.emulation.tandem_controliq import TandemControlIQEmulator
emu = TandemControlIQEmulator()
print(emu.get_algorithm_personality())
'''
                subprocess.run([sys.executable, '-c', legacy_script])
                
            elif choice == '3':
                print("\nOmnipod 5 Analysis")
                print("-" * 40)
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                legacy_script = f'''
import sys
sys.path.insert(0, "{project_root}")
from iints.emulation.omnipod_5 import Omnipod5Emulator
emu = Omnipod5Emulator()
print(emu.get_algorithm_personality())
'''
                subprocess.run([sys.executable, '-c', legacy_script])
                
            elif choice == '4':
                print("\nComparing All Legacy Pumps")
                print("-" * 40)
                print("This will run all legacy emulators on the same data...")
                
            elif choice == '5':
                print("\nLegacy vs New AI Comparison")
                print("-" * 40)
                print("This will compare legacy pumps against new algorithms...")
                self.run_battle_mode()
                
            elif choice == '0':
                break
            else:
                print("Invalid choice. Press Enter to continue...")
                input()
                
    def show_results_menu(self):
        self.clear_screen()
        self.show_header()
        print("\nRESULTS OVERVIEW")
        print("-" * 40)
        
        results_found = False
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, 'results')
        
        # Check for battle reports
        battle_reports = os.path.join(results_dir, 'reports', 'battle_report.json')
        if os.path.exists(battle_reports):
            print("\nBattle Report: Available")
            results_found = True
            
        plots_dir = os.path.join(results_dir, 'plots')
        if os.path.exists(plots_dir):
            plots = []
            for root, dirs, files in os.walk(plots_dir):
                plots.extend([f for f in files if f.endswith('.png')])
            if plots:
                print(f"\nVisualization files: {len(plots)} plots found")
                results_found = True
                
        reports_dir = os.path.join(results_dir, 'reports')
        if os.path.exists(reports_dir):
            reports = [f for f in os.listdir(reports_dir) if f.endswith(('.json', '.png'))]
            if reports:
                print(f"\nReport files: {len(reports)} reports found")
                results_found = True
                
        if not results_found:
            print("\nNo results found. Execute analyses to generate results.")
        else:
            print("\nRESULT DIRECTORIES:")
            print("- results/plots/   : All visualizations")
            print("- results/reports/ : Analysis reports")
            print("- results/data/    : Raw data files")
            
        input("\nPress Enter to continue...")
        
    def show_settings_menu(self):
        self.clear_screen()
        self.show_header()
        print("\nSETTINGS")
        print("-" * 40)
        print("[1] System Validation")
        print("[2] Clean Results Directory")
        print("[3] Reset Configuration")
        print("[4] System Information")
        print("[0] Back to Main Menu")
        print("-" * 40)
        
        choice = input("Select option: ")
        
        if choice == '1':
            print("\nExecuting system validation...")
            self.run_script('validate_system.py')
        elif choice == '2':
            confirm = input("\nWARNING: Clean all results? This cannot be undone! [y/N]: ")
            if confirm.lower() == 'y':
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                results_dir = os.path.join(project_root, 'results')
                os.system(f'find {results_dir} -name "*.png" -delete')
                os.system(f'find {results_dir} -name "*.json" -delete')
                os.system(f'find {results_dir} -name "*.csv" -delete')
                print("Results directory cleaned successfully.")
                input("Press Enter to continue...")
        elif choice == '3':
            print("\nConfiguration reset functionality not implemented.")
            input("Press Enter to continue...")
        elif choice == '4':
            print("\nSYSTEM INFORMATION")
            print("-" * 40)
            print(f"Python Version: {sys.version}")
            print(f"Working Directory: {os.getcwd()}")
            print(f"Platform: {os.name}")
            input("\nPress Enter to continue...")
            
    def run(self):
        while self.running:
            self.show_main_menu()
            choice = input("Select option: ")
            
            if choice == '1':
                self.handle_algorithm_menu()
            elif choice == '2':
                self.handle_battle_menu()
            elif choice == '3':
                self.handle_legacy_menu()
            elif choice == '4':
                print("\nExecuting data validation...")
                self.run_script('test_validation.py')
            elif choice == '5':
                self.show_results_menu()
            elif choice == '6':
                print("\nStarting Pro Terminal...")
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                pro_terminal_path = os.path.join(project_root, 'bin', 'iints_pro.py')
                if os.path.exists(pro_terminal_path):
                    subprocess.run([sys.executable, pro_terminal_path, '--interactive'])
                else:
                    print("Pro Terminal not found.")
                    input("Press Enter to continue...")
            elif choice == '7':
                self.show_settings_menu()
            elif choice == '0':
                print("\nExiting IINTS-AF. Thank you for using the framework.")
                self.running = False
            else:
                print("Invalid selection. Press Enter to continue...")
                input()

if __name__ == '__main__':
    app = IINTSTerminalApp()
    app.run()
