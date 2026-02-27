import irrep
import matplotlib.pyplot as plt
import numpy as np
import ray
import scipy
import wannierberri as wberri

ray.init(num_cpus=8)

#print (f"Using WannierBerri version {wberri.__version__}")


tabulators = { "energy": wberri.calculators.tabulate.Energy(),
               "berry_curvature" : wberri.calculators.tabulate.BerryCurvature(),
               "spin" : wberri.calculators.tabulate.Spin(),
               }

tab_all_path = wberri.calculators.TabulatorAll(
    tabulators,
    ibands = np.arange(0,18),
    mode = "path"
)

system=wberri.System_w90(
    seedname='examples/GaAs/GaAs',
    berry=True,   # needed to calculate "external terms" of Berry connection or curvature , reads ".mmn" file
    spin = True , # needed for spin properties, reads ".spn" file
)

spacegroup = irrep.spacegroup.SpaceGroup.from_cell( real_lattice=system.real_lattice,
                                                    positions=[[0,0,0]],   # only 1 Fe atoms at origin
                                                    typat=[1],    # atomic number is not important here
                                                    spinor=True,
                                                    magmom=[[0,0,1]],   # magnetic moment along z
                                                    include_TR=True,   # include symmetries that flip the spin
                                                    )

system.set_pointgroup(spacegroup=spacegroup)

path , path_result= wberri.evaluate_k_path(system=system,
                                           nodes=[
                                               [0.0000, 0.0000, 0.0000 ],   #  G
                                               [0.500 ,-0.5000, -0.5000],   #  H
                                               [0.7500, 0.2500, -0.2500],   #  P
                                               [0.5000, 0.0000, -0.5000],   #  N
                                               [0.0000, 0.0000, 0.000  ] ] , #  G
                                           labels=["G","H","P","N","G"],
                                           length=200 ,
                                           quantities=["berry_curvature","spin"])


EF = 12.6
# Import the pre-computed bands from quantum espresso
A = np.loadtxt(open("bands/Fe_bands_pw.dat","r"))
bohr_ang = scipy.constants.physical_constants['Bohr radius'][0] / 1e-10
alatt = 5.4235* bohr_ang
A[:,0]*= 2*np.pi/alatt
A[:,1]-=EF
# plot it as dots
plt.scatter (A[:,0],A[:,1],s=5,label = "QE")


path_result.plot_path_fat( path,
                           quantity=None,
                           save_file="Fe_bands+QE.pdf",
                           Eshift=EF,
                           Emin=-10,  Emax=50,
                           iband=None,
                           mode="fatband",
                           fatfactor=20,
                           cut_k=False,
                           close_fig=False,
                           show_fig=False,
                           label = "WB"
                           )


plt.legend()
plt.show()
plt.close()

#path=wberri.Path(system,
#                 nodes=[
#                     [0.0000, 0.0000, 0.0000 ],   #  G
#                     [0.500 ,-0.5000, -0.5000],   #  H
#                     [0.7500, 0.2500, -0.2500],   #  P
#                     [0.5000, 0.0000, -0.5000],   #  N
#                     [0.0000, 0.0000, 0.000  ] ] , #  G
#                 labels=["G","H","P","N","G"],
#                 length=200 )   # length [ Ang] ~= 2*pi/dk
#
#
#result=wberri.run(system,
#                  grid=path,
#                  calculators = {"tabulate" : tab_all_path},
#                  print_Kpoints = False)
#
#print(result.results)

# generators=["Inversion","C4z","TimeReversal*C2x"]
# system.set_pointgroup(symmetry_gen=generators)
