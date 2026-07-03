
"""
Module to expose more detailed version info for the installed `numpy`
"""
version = "2.5.0"
__version__ = version
full_version = version

git_revision = "6910b28fc12f4c3e821f315e24c51a6a2d89ba49"
release = 'dev' not in version and '+' not in version
short_version = version.split("+")[0]
