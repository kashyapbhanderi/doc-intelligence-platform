import sys
import os

# Add project root to Python path
# This fixes ALL import errors in tests
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))