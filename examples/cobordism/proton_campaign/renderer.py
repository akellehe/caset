# Throwaway sweep renderer (#555): record an attempt as an animation in the style of
# the animation scripts. Reuses emergent_proton.EmergentProtonAnimator's 2x4 panels
# verbatim (metrics F/‖∇S‖²/r_U; register+Betti; primal A/B; dual spatial/temporal
# curvature) — but frames are driven EXTERNALLY at the worker's real pass/chunk
# boundaries, so the animation shows the exact recorded attempt, not a re-run.
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/home/andrew/feat-proton-ingredients/examples/cobordism")
import emergent_proton as ep  # noqa: E402  (brings multicobordism_animation with it)


class AttemptRecorder:
    """PNG frames at the worker's phase/chunk boundaries; GIF assembly for keepers."""

    def __init__(self, nodes, frame_dir, max_frames=400, dpi=70):
        self.animator = ep.EmergentProtonAnimator(nodes)
        self.animator._frames = 10 ** 9     # never trip the "final frame" verdict path
        self.animator._setup(plt)
        self.frame_dir = frame_dir
        os.makedirs(frame_dir, exist_ok=True)
        for stale in os.listdir(frame_dir):  # a crashed prior attempt's leftovers
            os.remove(os.path.join(frame_dir, stale))
        self.count = 0
        self.max_frames = max_frames
        self.dpi = dpi

    def frame(self, node_index, phase, subtitle=""):
        """One frame: record the node's metrics into the shared history, redraw the
        inherited panels, save a PNG. Never raises — rendering must not kill physics."""
        if self.count >= self.max_frames:
            return
        try:
            self.animator._active = node_index
            node = self.animator.nodes[node_index][0]
            self.animator._record(node, node_index, phase)
            self.animator._redraw()
            label = self.animator.nodes[node_index][1]
            self.animator.fig.suptitle(
                f"ProtonIngredients sweep — {label} · {phase}{subtitle}")
            self.animator.fig.savefig(
                os.path.join(self.frame_dir, f"f{self.count:05d}.png"), dpi=self.dpi)
            self.count += 1
        except Exception:
            pass

    def finish(self, gif_path=None):
        """Close the figure; assemble the GIF when a path is given. Frames are always
        deleted afterwards (keep decisions are the caller's, via gif_path)."""
        try:
            plt.close(self.animator.fig)
        except Exception:
            pass
        try:
            if gif_path and self.count:
                from PIL import Image
                os.makedirs(os.path.dirname(os.path.abspath(gif_path)), exist_ok=True)
                paths = sorted(
                    os.path.join(self.frame_dir, f) for f in os.listdir(self.frame_dir))
                frames = [Image.open(p) for p in paths]
                frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                               duration=200, loop=0)
        except Exception:
            gif_path = None
        for f in os.listdir(self.frame_dir):
            os.remove(os.path.join(self.frame_dir, f))
        return gif_path
