
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from ghost_autovid_world.ui.dashboard import render_dashboard

if __name__ == "__main__":
    render_dashboard()
