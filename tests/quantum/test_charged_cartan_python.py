"""Python coverage for the Charged Cartan Monte Carlo v0.2.

SKIPPED in #56 (charge-observables-v0.3 follow-up: see issue #57) —
the qudit / charge / annihilate / pairCreate surface this file
exercises was removed when the simulation moved to a per-vertex
QuantumState model. The replacement charge observables, including
new tests against `QuantumState`, will land in #57; this stub keeps
the file importable so pytest discovery doesn't crater, but every
test it would have collected is `pytest.mark.skip`ped out.

The pre-refactor file content is preserved in git history at
commit f242ffd (the parent of the refactor branch).
"""

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "charge / qudit observables removed in #56; reinstated against "
        "QuantumState in charge-observables-v0.3 (issue #57)"
    )
)


def test_placeholder_until_charge_observables_v03() -> None:
    """Holder so pytest collects this module to the skip count.

    Replaced by the real test suite in #57.
    """
    pass
