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


def _prepare_parallel_execution(app):
    """Make notebook execution safe with ``sphinx-build --jobs``.

    The processes of a parallel build would otherwise create the tables of a new
    jupyter-cache database simultaneously (``table settings already exists``), and start
    kernels on the same port: jupyter_client picks a free port by binding to it and
    releasing it, and the kernel binds to it only later (``Address already in use``).
    The cache is therefore created here, before the parallel read, and a file lock lets
    one process at a time start a kernel, until the kernel replies.
    """
    config = app.env.mystnb_config
    if config.execution_mode != "cache":
        return
    from jupyter_cache import get_cache  # ruff: ignore[import-outside-top-level]

    get_cache(config.execution_cache_path).list_cache_records()
    if app.parallel <= 1 or os.name != "posix":
        return
    import fcntl  # ruff: ignore[import-outside-top-level]
    from pathlib import Path  # ruff: ignore[import-outside-top-level]

    from nbclient import NotebookClient  # ruff: ignore[import-outside-top-level]

    lock_path = Path(config.execution_cache_path).parent / "kernel-start.lock"
    start_kernel = NotebookClient.async_start_new_kernel
    start_client = NotebookClient.async_start_new_kernel_client

    def release(client):
        lock = client.__dict__.pop("_kernel_start_lock", None)
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    async def locked_start_kernel(self, **kwargs):
        lock = lock_path.open("w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        self.__dict__["_kernel_start_lock"] = lock
        try:
            await start_kernel(self, **kwargs)
        except BaseException:
            release(self)
            raise

    async def start_client_and_unlock(self):
        try:
            return await start_client(self)  # waits for the kernel to reply
        finally:
            release(self)

    NotebookClient.async_start_new_kernel = locked_start_kernel
    NotebookClient.async_start_new_kernel_client = start_client_and_unlock


def setup(app):
    app.connect("builder-inited", _prepare_parallel_execution, priority=600)
