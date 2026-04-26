"""tessera -- Causal Set and CDT simulation library.

The heavy lifting lives in the C++ extension ``_tessera``.  This
package re-exports every public name so that ``import tessera`` and
``from tessera import Spacetime`` continue to work exactly as before.
"""

from tessera._tessera import *          # noqa: F401,F403 – re-export all C++ bindings
from tessera._tessera import __doc__    # keep the C++ module docstring
