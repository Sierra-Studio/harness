"""Shared pytest setup for the suite.

Puts the repo root on sys.path once, so every test module can `import harness`
without an installed package and without repeating a per-file path shim.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
