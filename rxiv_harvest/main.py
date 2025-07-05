#!/usr/bin/env python3
"""
Backward compatibility wrapper for main.py -> download.py rename.
This ensures existing running processes and scripts continue to work.
"""

import sys
import subprocess

if __name__ == "__main__":
    # Forward all arguments to download.py
    cmd = [sys.executable, "download.py"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))