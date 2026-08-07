import os
import sys

# Ensure repository root is in sys.path so app package imports work seamlessly on Vercel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
