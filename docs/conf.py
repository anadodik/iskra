import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))
sys.path.insert(0, str(Path(__file__).parent / "ext"))

project = "iskra ✨"
copyright = "2022, Ana Dodik"
author = "Ana Dodik"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_github_style",
    "sphinx_math_dollar",
    # "sphinx_external_toc",
    "shape_tooltips",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
root_doc = "index"
exclude_patterns = ["api/iskra.*.rst"]
templates_path = ["templates"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
default_role = "any"

myst_enable_extensions = [
    "alert",
    "amsmath",
    "dollarmath",
    "colon_fence",
    "deflist",
    "tasklist",
    "attrs_inline",
    "substitution",
    "html_admonition",
    "html_image",
]
myst_heading_anchors = 3

autosummary_generate = True
autosummary_generate_overwrite = True

autodoc_member_order = "bysource"
autodoc_inherit_docstrings = False

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_custom_sections = [("Returns", "params_style")]
napoleon_use_admonition_for_examples = False

# external_toc_path = "_toc.yml"

html_baseurl = "https://iskra-graphics.org"
html_theme = "pydata_sphinx_theme"
html_static_path = ["static"]
html_css_files = ["custom.css"]
html_favicon = "logo.png"
html_copy_source = False
html_show_sourcelink = False
html_theme_options = {
    "logo": {
        "image_light": "logo.svg",
        "image_dark": "logo.svg",
        "alt_text": "iskra ✨ - Home",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/anadodik/iskra",
            "icon": "fa-brands fa-github",
        },
    ],
    "show_toc_level": 1,
    "navigation_with_keys": True,
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
}

html_context = {
    "default_mode": "auto",
    "display_github": True,
    "github_user": "anadodik",
    "github_repo": "iskra",
    "github_version": "main",
}
linkcode_link_text = "[SOURCE]"
