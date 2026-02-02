#!/usr/bin/env python3
"""
Quick test of the IINTS-AF Terminal App
"""

import os
import sys

def test_terminal_app():
    print("Testing IINTS-AF Terminal App")
    print("=" * 50)
    
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    framework_dir = os.path.dirname(script_dir)
    os.chdir(framework_dir)
    
    print(f" Working directory: {os.getcwd()}")
    
    # Import and test the terminal app
    sys.path.insert(0, framework_dir)
    from main import IINTSTerminalApp
    
    app = IINTSTerminalApp()
    
    # Test basic functionality
    print(" Terminal app imported successfully")
    print(" App initialized")
    
    # Show what the app can do
    print("\n Available Features:")
    print("    Algorithm Analysis (Basic, Comprehensive, Custom)")
    print("    Reverse Engineering (Commercial Pump Analysis)")
    print("    Population Studies")
    print("    System Validation")
    print("    Results Viewing")
    print("     Settings & Configuration")
    
    print("\n To run the interactive terminal app:")
    print("   cd /home/rune/Documents/IINTS/iints-reveng-framework")
    print("   python3 main.py")
    
    print("\n The terminal app provides:")
    print("   - Interactive menus with emojis")
    print("   - Scenario and algorithm selection")
    print("   - Real-time progress feedback")
    print("   - Results management")
    print("   - System validation and settings")

if __name__ == '__main__':
    test_terminal_app()
