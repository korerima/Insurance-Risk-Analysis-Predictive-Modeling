#!/usr/bin/env python3
"""
Simple script to run the Insurance Risk Analysis Dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the Streamlit dashboard."""
    
    # Check if we're in the right directory
    if not Path("src/dashboard/streamlit_app.py").exists():
        print("❌ Error: Please run this script from the project root directory.")
        print("   Make sure you're in the 'insurance-risk-analysis' folder.")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} is installed")
    except ImportError:
        print("❌ Error: Streamlit is not installed.")
        print("   Please install it with: pip install streamlit")
        sys.exit(1)
    
    # Check if required packages are installed
    required_packages = ['plotly', 'shap', 'pandas', 'numpy', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Error: Missing required packages: {', '.join(missing_packages)}")
        print("   Please install them with: pip install -r requirements.txt")
        sys.exit(1)
    
    print("🚀 Starting Insurance Risk Analysis Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Run the dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/dashboard/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
