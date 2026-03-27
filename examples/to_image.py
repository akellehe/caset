import caset

sig = caset.Signature(4, caset.Lorentzian)
metric = caset.Metric(True, sig)
st = caset.Spacetime(metric, caset.CDT, 1.0, 1.0, caset.PREFERRED, caset.Toroid())
st.build(2000)

# Optional: thermalize first
cdt = caset.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())
cdt.tune()
cdt.sweep(50)

st.save("spacetime.png")                          # default 800px panels
# st.save("spacetime_hires.png", panel_size=1200)    # larger
# st.save("spacetime_fast.png", layout_iters=100)    # faster layout

# Animated GIF with default precession (spin=1, precession=1, tilt=25°)
st.save("spacetime.gif")

# More precession wobble
# st.save("spacetime_prec.gif", precession=2, tilt=30)

# Pure turntable, no wobble
# st.save("spacetime_turntable.gif", precession=0, tilt=0)
