#!/usr/bin/env python3
"""
Quick launcher for the Amharic NER Dashboard
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages for dashboard"""
    packages = [
        'streamlit',
        'plotly',
        'pandas'
    ]
    
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def run_dashboard():
    """Launch the Streamlit dashboard"""
    dashboard_path = Path(__file__).parent / "dashboard.py"
    
    print("Starting Amharic NER Dashboard...")
    print("Dashboard will open in your browser")
    print("URL: http://localhost:8501")
    print("\n" + "="*50)
    
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', 
        str(dashboard_path),
        '--server.port=8501',
        '--server.headless=false'
    ])

if __name__ == "__main__":
    install_requirements()
    run_dashboard()