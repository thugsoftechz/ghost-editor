
import sys
import os

# Robust path setup to allow running from anywhere
# We need the parent directory of 'ghost_autovid_world' to be in sys.path
# so that 'import ghost_autovid_world.xxx' works.

current_dir = os.path.dirname(os.path.abspath(__file__)) # .../ghost_autovid_world
parent_dir = os.path.dirname(current_dir) # .../ (root containing the package)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ghost_autovid_world.ui.dashboard import render_dashboard

if __name__ == "__main__":
    render_dashboard()
