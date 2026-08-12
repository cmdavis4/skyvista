"""
``python -m skyvista`` runs the environment doctor.

A quick, dependency-free way to check that headless rendering works on this
machine and to get actionable fixes if it doesn't -- e.g. right after installing
skyvista on a remote server.
"""

from .headless import doctor

if __name__ == "__main__":
    doctor()
