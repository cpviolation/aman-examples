"""Sphinx configuration of the AmAn examples documentation."""

import os

ORGANIZATION = "cpviolation"
PACKAGE = "aman_examples"
REPO_NAME = "aman-examples"
REPO_TITLE = "AmAn examples"

project = REPO_TITLE
author = "Maurizio Martinelli"
copyright = "2026, Maurizio Martinelli"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]
exclude_patterns = ["**.ipynb_checkpoints", "_build", "AGENTS.md", "CLAUDE.md"]

html_title = REPO_TITLE
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": f"https://github.com/{ORGANIZATION}/{REPO_NAME}",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 2,
}

autodoc_default_options = {
    "exclude-members": "default_assumptions, doit, evaluate, is_commutative",
    "members": True,
    "undoc-members": True,
}
autodoc_member_order = "bysource"
autodoc_typehints_format = "short"
default_role = "py:obj"
napoleon_google_docstring = True
intersphinx_mapping = {
    "ampform": ("https://ampform.readthedocs.io/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "sympy": ("https://docs.sympy.org/latest", None),
    "tensorwaves": ("https://tensorwaves.readthedocs.io/stable", None),
}

myst_enable_extensions = ["amsmath", "colon_fence", "dollarmath"]
# Notebooks are executed (with caching) only if EXECUTE_NB is set, as in ComPWA projects
nb_execution_mode = "cache" if os.getenv("EXECUTE_NB") else "off"
nb_execution_timeout = -1
nb_execution_raise_on_error = True
nb_output_stderr = "remove"
