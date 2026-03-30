"""caset -- Causal Set and CDT simulation library.

The heavy lifting lives in the C++ extension ``_caset``.  This
package re-exports every public name so that ``import caset`` and
``from caset import Spacetime`` continue to work exactly as before.
"""

from caset._caset import *          # noqa: F401,F403 – re-export all C++ bindings
from caset._caset import __doc__    # keep the C++ module docstring
