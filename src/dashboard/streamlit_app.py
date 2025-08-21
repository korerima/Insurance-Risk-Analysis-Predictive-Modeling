"""
Main entry point for the Streamlit dashboard application.
"""

import streamlit as st
from src.dashboard.app import create_app

if __name__ == "__main__":
    create_app()
