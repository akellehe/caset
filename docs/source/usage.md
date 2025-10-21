# Usage

Basic Python usage (via pybind11 extension):

```python
import caset
v = caset.Vertex(1, [0.0, 0.0, 0.0])
e = caset.Edge(v, v, 1.0)
print(v.getId(), e.getWeight())
```

> If you later add a Python package around the extension, you can document
> pure-Python helpers here and enable `autodoc`.
