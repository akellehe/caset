import os, sys, pathlib

THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DOXY_XML = REPO_ROOT / "_doxygen" / "xml"

project = "caset"
author = "caset contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
]

myst_enable_extensions = ["dollarmath", "amsmath"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_heading_anchors = 3

mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

breathe_projects = {"caset": str(DOXY_XML)}
breathe_default_project = "caset"

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]
