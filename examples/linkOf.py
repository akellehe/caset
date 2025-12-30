import argparse
import random
import collections

from plot4D import embed_euclidean, plot_spacetime

from caset import Spacetime, Simplex, SimplexOrientation


def build_star(st, k, ti, tf):
    print("Building star...")
    seed, created = st.createSimplex((ti, tf))
    print("Created seed...")
    t = 1.
    i = ti + tf
    for exterior_facet in seed.getFacets():
        t *= -1.
        i += 1
        vertex = st.createVertex(i, [t, 0., 0., 0.])
        print("Created vertex", vertex)
        k_simplex, new_facets = exterior_facet.cone(vertex)
        for facet in new_facets:
            t *= -1.
            i += 1
            vertex = st.createVertex(i, [t, 0., 0., 0.])
            print("Created vertex", vertex)
            facet.cone(vertex)
    print("Returning seed...")
    print(seed)
    return seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the link of a particular simplex/dimension so you can see what it looks like.")
    parser.add_argument("-k", "--dimension", type=int, default=2, help="Dimension of the simplicial complex.")
    parser.add_argument("--ti", "-i", type=int, default=2, help="The number of vertices of the simplex for which you would like to see the star on the initial time slice")
    parser.add_argument("--tf", "-f", type=int, default=1, help="The number of vertices on the simplex for which you would like to see the star on the final time slice.")
    args = parser.parse_args()
    st = Spacetime()
    seed = build_star(st, args.dimension, args.ti, args.tf)
    print("Getting star...")
    star = seed.getStar()
    link = seed.getLink()
    print("Got star!")
    print(star)
    breakpoint()
    embed_euclidean(st, dimensions=4, epsilon=10e-10)

    plot_spacetime(st)

