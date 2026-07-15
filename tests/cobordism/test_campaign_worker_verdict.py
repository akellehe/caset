# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The campaign worker's verdict-record extremes (#597).

Stage 2 constructs every trial exactly real (Im l^2 == 0 by construction,
#589), so the worker's final-state reads can never surface a transient Im
produced upstream — the verdict must carry the TRAJECTORY extremes instead:
max_im_seen (a nonzero value flags an upstream Im producer) and min_re_min
(how close the drive ever came to the causal sector Re l^2 < 0), accumulated
over every pass/chunk snapshot.
"""
import importlib.util
import os
import unittest

_CAMPAIGN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, os.pardir, "examples", "cobordism",
                         "proton_campaign")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_CAMPAIGN, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class AttemptStateExtremes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = _load("worker")

    def test_no_snapshot_yields_null_min_re_min_not_json_infinity(self):
        fields = self.worker.AttemptState().verdict_fields()
        self.assertIsNone(fields["min_re_min"])
        self.assertEqual(fields["max_im_seen"], 0.0)

    def test_extremes_accumulate_over_snapshots(self):
        state = self.worker.AttemptState()
        state.see({"b3": 1, "holes": 2, "im_max": 0.0, "re_min": 0.9})
        state.see({"b3": 3, "holes": 1, "im_max": 0.25, "re_min": -0.1})
        state.see({"b3": 2, "holes": 4, "im_max": 0.0, "re_min": 0.5})
        self.assertEqual(state.verdict_fields(), {
            "max_holes": 4,
            "max_b3": 3,
            "max_im_seen": 0.25,
            "min_re_min": -0.1,
        })


if __name__ == "__main__":
    unittest.main()
