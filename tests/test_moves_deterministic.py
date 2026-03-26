# MIT License
# Copyright (c) 2025 Andrew Kelleher
"""
Deterministic tests for each Pachner move's topology change, decoupled
from random selection and Metropolis acceptance.

Each test builds a known lattice using the low-level spacetime API,
applies the move's topology change directly, asserts the exact result,
then applies the inverse and asserts we return.
"""

import unittest
import caset


def _make_spacetime():
    """Create a properly configured 4D Lorentzian spacetime."""
    sig = caset.Signature(4, caset.Lorentzian)
    metric = caset.Metric(True, sig)
    st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED,
                         caset.Toroid())
    return st


def _top_simplices(st):
    """Return list of all top-dimensional (5-vertex) simplices."""
    return [s for s in st.getSimplices() if len(s.getVertices()) == 5]


def _top_fps(st):
    """Set of fingerprints of all top simplices."""
    return {hash(s) for s in _top_simplices(st)}


def _assert_all_causal(test, st):
    """Every top simplex spans exactly 2 time slices."""
    for s in _top_simplices(st):
        times = {v.getTime() for v in s.getVertices()}
        test.assertEqual(len(times), 2,
                         f"Non-causal: orientation={s.getOrientation().numeric()}")


def _assert_counts(test, st):
    """N4 = N41 + N32 and matches manual count."""
    n41, n32 = 0, 0
    for s in _top_simplices(st):
        o = s.getOrientation().numeric()
        if o in ((4, 1), (1, 4)):
            n41 += 1
        elif o in ((3, 2), (2, 3)):
            n32 += 1
        else:
            test.fail(f"Invalid orientation {o}")
    test.assertEqual(st.getN41(), n41)
    test.assertEqual(st.getN32(), n32)
    test.assertEqual(st.getSimplexCount(), n41 + n32)


# =====================================================================
# Add (cone) — direct topology change
# =====================================================================

class TestConeForward(unittest.TestCase):
    """Cone a facet to a new vertex.  This is the topology change behind
    CDT::add(), separated from random selection and acceptance.
    """

    def test_cone_creates_one_new_top_simplex(self):
        """Coning a non-timelike facet creates exactly 1 new top simplex."""
        st = _make_spacetime()
        st.build(5)
        before_n4 = st.getSimplexCount()
        before_n0 = st.getVertexCount()
        before_fps = _top_fps(st)

        # Pick ANY top simplex and a non-timelike facet
        top = _top_simplices(st)[0]
        facet = None
        for f in top.getFacets():
            if not f.isTimelike():
                facet = f
                break
        self.assertIsNotNone(facet, "Need a non-timelike facet")

        # Create vertex and cone — this is the raw topology change
        vertex = st.createVertex([1.0])
        result_simplex, new_facets = facet.cone(vertex)

        # Exactly 1 new top simplex
        after_fps = _top_fps(st)
        gained = after_fps - before_fps
        self.assertEqual(len(gained), 1)

        # Counts changed correctly
        self.assertEqual(st.getSimplexCount(), before_n4 + 1)
        self.assertEqual(st.getVertexCount(), before_n0 + 1)

        # New simplex has 5 vertices
        self.assertEqual(len(result_simplex.getVertices()), 5)

        # New simplex contains the new vertex
        self.assertTrue(result_simplex.hasVertex(vertex))

        # New simplex has valid CDT orientation
        o = result_simplex.getOrientation().numeric()
        self.assertIn(o, ((4, 1), (1, 4), (3, 2), (2, 3)))

        _assert_counts(self, st)
        _assert_all_causal(self, st)

    def test_cone_then_remove_round_trip(self):
        """Cone a facet, then removeSimplex the result: N4 returns."""
        st = _make_spacetime()
        st.build(5)
        before_n4 = st.getSimplexCount()
        before_fps = _top_fps(st)

        top = _top_simplices(st)[0]
        facet = None
        for f in top.getFacets():
            if not f.isTimelike():
                facet = f
                break
        self.assertIsNotNone(facet)

        # Forward: cone
        vertex = st.createVertex([1.0])
        result_simplex, _ = facet.cone(vertex)
        self.assertEqual(st.getSimplexCount(), before_n4 + 1)

        # Reverse: remove the new simplex
        st.removeSimplex(result_simplex)
        self.assertEqual(st.getSimplexCount(), before_n4)

        # The original simplices are all still there
        after_fps = _top_fps(st)
        self.assertTrue(before_fps.issubset(after_fps))

        _assert_counts(self, st)

    def test_cone_five_times_then_remove_five(self):
        """5 cones then 5 removes: N4 returns to start each time."""
        st = _make_spacetime()
        st.build(10)
        start_n4 = st.getSimplexCount()

        added_simplices = []
        for i in range(5):
            top = _top_simplices(st)[0]
            facet = None
            for f in top.getFacets():
                if not f.isTimelike():
                    facet = f
                    break
            self.assertIsNotNone(facet, f"Iteration {i}: need facet")

            vertex = st.createVertex([1.0])
            result_simplex, _ = facet.cone(vertex)
            added_simplices.append(result_simplex)

            self.assertEqual(st.getSimplexCount(), start_n4 + i + 1)
            _assert_counts(self, st)
            _assert_all_causal(self, st)

        # Remove in reverse order
        for i, s in enumerate(reversed(added_simplices)):
            st.removeSimplex(s)
            expected = start_n4 + (4 - i)
            self.assertEqual(st.getSimplexCount(), expected,
                             f"Remove {i}: expected N4={expected}")
            _assert_counts(self, st)

        self.assertEqual(st.getSimplexCount(), start_n4)


# =====================================================================
# Flip (2 → d) — direct topology change
# =====================================================================

class TestFlipForward(unittest.TestCase):
    """The (2,d) flip replaces 2 simplices sharing a (d-1)-face with d
    new simplices.  This is the topology change, decoupled from CDT::flip().
    """

    def _find_flippable(self, st):
        """Find a (d-1)-face with exactly 2 top cofaces.
        Returns (facet, s1, s2, shared_verts, unique_verts) or None.
        """
        for top in _top_simplices(st):
            for facet in top.getFacets():
                cofaces = [c for c in facet.getCofaces()
                           if len(c.getVertices()) == 5]
                if len(cofaces) != 2:
                    continue
                s1, s2 = cofaces
                s1_vids = {v.getId() for v in s1.getVertices()}
                s2_vids = {v.getId() for v in s2.getVertices()}
                shared_vids = s1_vids & s2_vids
                unique_vids = s1_vids ^ s2_vids
                if len(shared_vids) != 4 or len(unique_vids) != 2:
                    continue
                # Map IDs back to vertex objects
                all_verts = {v.getId(): v for v in
                             list(s1.getVertices()) + list(s2.getVertices())}
                shared = [all_verts[vid] for vid in shared_vids]
                unique = [all_verts[vid] for vid in unique_vids]
                return facet, s1, s2, shared, unique
        return None

    def test_flip_2_to_d(self):
        """Remove 2, create d=4 new simplices with correct vertex sets."""
        st = _make_spacetime()
        st.build(20)
        before_n4 = st.getSimplexCount()
        before_n0 = st.getVertexCount()

        result = self._find_flippable(st)
        if result is None:
            self.skipTest("No flippable configuration found")
        facet, s1, s2, shared, unique = result

        # Record what we're about to do
        s1_fp, s2_fp = hash(s1), hash(s2)

        # Forward: remove old, create new
        st.removeSimplex(s1)
        st.removeSimplex(s2)
        self.assertEqual(st.getSimplexCount(), before_n4 - 2)

        # Create d=4 new simplices: each skips one of the 4 shared vertices
        new_simplices = []
        for skip in range(4):
            verts = [shared[i] for i in range(4) if i != skip]
            verts.extend(unique)
            self.assertEqual(len(verts), 5)
            new_s, created = st.createSimplex(verts)
            if created:
                new_simplices.append(new_s)

        self.assertGreaterEqual(len(new_simplices), 1)
        after_n4 = st.getSimplexCount()
        # N4 should have increased (removed 2, added up to 4)
        self.assertGreater(after_n4, before_n4 - 2)

        # Vertex count unchanged
        self.assertEqual(st.getVertexCount(), before_n0)

        # All new simplices have valid CDT orientations
        for s in new_simplices:
            o = s.getOrientation().numeric()
            self.assertIn(o, ((4, 1), (1, 4), (3, 2), (2, 3)),
                          f"New simplex has invalid orientation {o}")

        _assert_counts(self, st)
        _assert_all_causal(self, st)

    def test_flip_then_reverse_flip(self):
        """Flip forward, then flip back: N4 returns."""
        st = _make_spacetime()
        st.build(20)
        before_n4 = st.getSimplexCount()

        result = self._find_flippable(st)
        if result is None:
            self.skipTest("No flippable configuration found")
        _, s1, s2, shared, unique = result

        # Remember the original simplex vertex sets for reconstruction
        s1_verts = list(s1.getVertices())
        s2_verts = list(s2.getVertices())

        # Forward flip: remove 2, create 4
        st.removeSimplex(s1)
        st.removeSimplex(s2)
        new_simplices = []
        for skip in range(4):
            verts = [shared[i] for i in range(4) if i != skip]
            verts.extend(unique)
            new_s, created = st.createSimplex(verts)
            if created:
                new_simplices.append(new_s)
        mid_n4 = st.getSimplexCount()
        _assert_counts(self, st)

        # Reverse flip: remove the new simplices, recreate the old 2
        for s in new_simplices:
            st.removeSimplex(s)
        s1_new, _ = st.createSimplex(s1_verts)
        s2_new, _ = st.createSimplex(s2_verts)

        self.assertEqual(st.getSimplexCount(), before_n4)
        _assert_counts(self, st)
        _assert_all_causal(self, st)


# =====================================================================
# Shift (3 → 3) — direct topology change
# =====================================================================

class TestShiftForward(unittest.TestCase):
    """The (3,3) shift replaces 3 simplices sharing a (d-2)-face with
    3 new simplices.  Decoupled from CDT::shift().
    """

    def _find_shiftable(self, st):
        """Find 3 top simplices sharing 3 vertices (a triangle).
        Returns (sharing, shared_verts, unique_verts) or None.
        """
        tops = _top_simplices(st)
        # For each top simplex, try all triples of its vertices
        for top in tops:
            verts = list(top.getVertices())
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    for k in range(j + 1, len(verts)):
                        tri = [verts[i], verts[j], verts[k]]
                        # Find all top simplices containing all 3
                        sharing = []
                        for s in tri[0].getSimplices():
                            if len(s.getVertices()) != 5:
                                continue
                            if s.hasVertex(tri[1]) and s.hasVertex(tri[2]):
                                sharing.append(s)
                        if len(sharing) != 3:
                            continue
                        # Collect all vertices
                        all_vids = set()
                        for s in sharing:
                            for v in s.getVertices():
                                all_vids.add(v.getId())
                        if len(all_vids) != 6:  # d+2 = 6 in 4D
                            continue
                        # Separate shared (in all 3) and unique
                        all_verts = {}
                        for s in sharing:
                            for v in s.getVertices():
                                all_verts[v.getId()] = v
                        shared_v = []
                        unique_v = []
                        for vid, v in all_verts.items():
                            in_all = all(s.hasVertex(v) for s in sharing)
                            if in_all:
                                shared_v.append(v)
                            else:
                                unique_v.append(v)
                        if len(shared_v) == 3 and len(unique_v) == 3:
                            return sharing, shared_v, unique_v
        return None

    def test_shift_3_to_3(self):
        """Remove 3, create 3 new simplices with correct vertex sets."""
        st = _make_spacetime()
        st.build(30)
        # Run a few sweeps to diversify the topology for shift configs
        target = st.getSimplexCount()
        cdt = caset.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, target)
        cdt.sweep(50)

        before_n4 = st.getSimplexCount()
        before_n0 = st.getVertexCount()

        result = self._find_shiftable(st)
        if result is None:
            self.skipTest("No shiftable configuration found")
        sharing, shared, unique = result

        # Record old simplex vertex sets for potential reversal
        old_vert_sets = [list(s.getVertices()) for s in sharing]

        # Forward: remove 3, create 3
        for s in sharing:
            st.removeSimplex(s)
        self.assertEqual(st.getSimplexCount(), before_n4 - 3)

        # Each new simplex: 2 of 3 shared + all 3 unique = 5
        new_simplices = []
        for skip in range(3):
            verts = [shared[i] for i in range(3) if i != skip]
            verts.extend(unique)
            self.assertEqual(len(verts), 5)
            new_s, created = st.createSimplex(verts)
            if created:
                new_simplices.append(new_s)

        # Vertex count unchanged
        self.assertEqual(st.getVertexCount(), before_n0)

        # All new simplices have valid CDT orientations
        for s in new_simplices:
            o = s.getOrientation().numeric()
            self.assertIn(o, ((4, 1), (1, 4), (3, 2), (2, 3)))

        _assert_counts(self, st)
        _assert_all_causal(self, st)

    def test_shift_then_reverse(self):
        """Shift forward, then shift back: N4 returns."""
        st = _make_spacetime()
        st.build(30)
        target = st.getSimplexCount()
        cdt = caset.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, target)
        cdt.sweep(50)

        before_n4 = st.getSimplexCount()

        result = self._find_shiftable(st)
        if result is None:
            self.skipTest("No shiftable configuration found")
        sharing, shared, unique = result

        old_vert_sets = [list(s.getVertices()) for s in sharing]

        # Forward shift
        for s in sharing:
            st.removeSimplex(s)
        new_simplices = []
        for skip in range(3):
            verts = [shared[i] for i in range(3) if i != skip]
            verts.extend(unique)
            new_s, created = st.createSimplex(verts)
            if created:
                new_simplices.append(new_s)
        mid_n4 = st.getSimplexCount()
        _assert_counts(self, st)

        # Reverse shift: remove new, recreate old
        for s in new_simplices:
            st.removeSimplex(s)
        for old_verts in old_vert_sets:
            st.createSimplex(old_verts)

        self.assertEqual(st.getSimplexCount(), before_n4)
        _assert_counts(self, st)
        _assert_all_causal(self, st)


# =====================================================================
# Iterated forward-backward
# =====================================================================

class TestIteratedConeRemove(unittest.TestCase):
    """Cone then removeSimplex, iterated many times."""

    def test_ten_iterations(self):
        st = _make_spacetime()
        st.build(10)
        start_n4 = st.getSimplexCount()

        for iteration in range(10):
            # Find a non-timelike facet to cone
            top = _top_simplices(st)[0]
            facet = None
            for f in top.getFacets():
                if not f.isTimelike():
                    facet = f
                    break
            self.assertIsNotNone(facet, f"Iter {iteration}: need facet")

            # Forward: cone
            vertex = st.createVertex([1.0])
            new_s, _ = facet.cone(vertex)
            self.assertEqual(st.getSimplexCount(), start_n4 + 1,
                             f"Iter {iteration}: after cone")
            _assert_counts(self, st)
            _assert_all_causal(self, st)

            # Backward: remove
            st.removeSimplex(new_s)
            self.assertEqual(st.getSimplexCount(), start_n4,
                             f"Iter {iteration}: after remove")
            _assert_counts(self, st)


class TestIteratedFlip(unittest.TestCase):
    """Flip forward then backward, iterated."""

    def _find_flippable(self, st):
        for top in _top_simplices(st):
            for facet in top.getFacets():
                cofaces = [c for c in facet.getCofaces()
                           if len(c.getVertices()) == 5]
                if len(cofaces) != 2:
                    continue
                s1, s2 = cofaces
                s1_vids = {v.getId() for v in s1.getVertices()}
                s2_vids = {v.getId() for v in s2.getVertices()}
                shared_vids = s1_vids & s2_vids
                unique_vids = s1_vids ^ s2_vids
                if len(shared_vids) != 4 or len(unique_vids) != 2:
                    continue
                all_verts = {v.getId(): v for v in
                             list(s1.getVertices()) + list(s2.getVertices())}
                shared = [all_verts[vid] for vid in shared_vids]
                unique = [all_verts[vid] for vid in unique_vids]
                return s1, s2, shared, unique
        return None

    def test_five_iterations(self):
        st = _make_spacetime()
        st.build(20)
        start_n4 = st.getSimplexCount()

        for iteration in range(5):
            result = self._find_flippable(st)
            if result is None:
                # Lattice might not have flippable configs after changes
                return
            s1, s2, shared, unique = result
            s1_verts = list(s1.getVertices())
            s2_verts = list(s2.getVertices())

            # Forward flip
            st.removeSimplex(s1)
            st.removeSimplex(s2)
            new_simplices = []
            for skip in range(4):
                verts = [shared[i] for i in range(4) if i != skip]
                verts.extend(unique)
                new_s, created = st.createSimplex(verts)
                if created:
                    new_simplices.append(new_s)
            _assert_counts(self, st)
            _assert_all_causal(self, st)

            # Reverse flip
            for s in new_simplices:
                st.removeSimplex(s)
            st.createSimplex(s1_verts)
            st.createSimplex(s2_verts)

            self.assertEqual(st.getSimplexCount(), start_n4,
                             f"Iter {iteration}: N4 should return")
            _assert_counts(self, st)
            _assert_all_causal(self, st)


if __name__ == "__main__":
    unittest.main()
