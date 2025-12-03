# Getting Started

To construct a `Spacetime` in python you can run

```python
from caset import Spacetime

spacetime = Spacetime()
```

You can see an example of how the spacetime is constructed and embedded into a 3D euclidean space with the example 
script, examples/plot4D.py.

```bash
python3 plot4D.py --n-simplices 10
```

You can adjust the behavior of that `Spacetime` by defining a `Metric` with a `Signature` as well as setting an 
`alpha` value to use as the default edge length.

The default `Topology` is a `Toroid`, but you can choose others or define your own. The `Topology` is responsible for 
constructing your initial spacetime as a lattice of `Edge`s and `Vertex`(s). 

Once you've done that you can use the `Simplex` interface to interact with the `Spacetime` lattice. For example

```python
simplices = spacetime.getSimplices()
simplices[0].getVolume()
```
