# Getting Started

To construct a `Spacetime` in python you can run

```python
from caset import Spacetime

spacetime = Spacetime()
```

You can adjust the behavior of that `Spacetime` by defining a `Metric` with a `Signature` as well as setting an 
`alpha` value to use as the default edge length.

The default `Topology` is a `Toroid`, but you can choose others or define your own. The `Topology` is responsible for 
constructing your initial spacetime as a lattice of `Edge`s and `Vertex`(s). 

Once you've done that you can use the `Simplex` interface to interact with the `Spacetime` lattice. For example

```python
simplexes = spacetime.getSimplexes()
simplexes[0].getVolume()
```