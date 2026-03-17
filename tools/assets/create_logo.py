#!/usr/bin/env python3
"""
IINTS Logo Generator
Creates professional logo for PDF reports
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_iints_logo():
    """Create professional IINTS logo"""
    fig, ax = plt.subplots(figsize=(4, 2))
    
    # Background
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.axis('off')
    
    # Green background rectangle
    bg_rect = patches.Rectangle((0.1, 0.3), 3.8, 1.4, 
                               facecolor='#228B22', edgecolor='#1F5F1F', linewidth=2)
    ax.add_patch(bg_rect)
    
    # White text
    ax.text(2, 1, 'IINTS-AF', fontsize=24, fontweight='bold', 
           color='white', ha='center', va='center', family='monospace')
    
    # Subtitle
    ax.text(2, 0.6, 'Intelligent Insulin Technology System', 
           fontsize=8, color='white', ha='center', va='center')
    
    # Save logo
    logo_path = Path("img/iints_logo.png")
    plt.savefig(logo_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"IINTS logo created: {logo_path}")
    return str(logo_path)

if __name__ == "__main__":
    create_iints_logo()