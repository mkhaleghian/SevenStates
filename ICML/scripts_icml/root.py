# Project path
# Auto-detected as the ICML project root (parent of scripts_icml/).
# Override with the EV_GNN_ROOT environment variable if needed.
import os

PROJECT_ROOT = os.environ.get(
    'EV_GNN_ROOT',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
