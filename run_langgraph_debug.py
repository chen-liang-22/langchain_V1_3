#!/usr/bin/env python
"""
LangGraph Dev Server debug launcher.

Usage:
    - In PyCharm: set this file as Script Path, Parameters: dev
    - Click Debug (green bug icon)
    - Breakpoints in your agent code will work directly
"""
import sys
from multiprocessing import freeze_support
from langgraph_cli.cli import cli

if __name__ == '__main__':
    freeze_support()
    sys.exit(cli())
