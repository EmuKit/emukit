# Legacy shim retained for backward compatibility.
# Metadata & configuration have migrated to pyproject.toml (PEP 621).
# This file can be removed in a future major/minor release once
# users and downstream tooling no longer invoke `python setup.py ...`.

from setuptools import setup

if __name__ == "__main__":  # pragma: no cover
    setup()
