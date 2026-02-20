from pymatgen.io.vasp import Vasprun
import numpy as np
import matplotlib.pyplot as plt

vr = Vasprun("/raven/ptmp/twarford/jfremote_workdirs/prod0/07/81/db/0781dbca-43ee-4817-a2f2-8aa0335ff686_1/vasprun.xml.gz", parse_dos=True, parse_potcar_file=False)
def vasprun_to_atoms_with_pdos(vr):
    atoms = vr.final_structure.to_ase_atoms()

    atoms.arrays['dos_densities'] = np.zeros((len(atoms), 2000))

    for i, site in enumerate(vr.final_structure):
        site_dos = vr.complete_dos.get_site_dos(site)
        atoms.arrays['dos_densities'][i] = site_dos.get_densities()

    atoms.info['dos_energies'] = site_dos.energies
    atoms.info['dos_efermi'] = site_dos.efermi

    return atoms

def _bin_edges_from_centers(centers):
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (midpoints[0] - centers[0])
    last = centers[-1] + (centers[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])

def plot_site_dos_heatmap(
        atoms,
        *,
        zero_at_efermi=True,
        cmap="viridis", 
        ax = None,
        e_max=3,
        norm='symlog',
        ):
    if "dos_energies" not in atoms.info or "dos_densities" not in atoms.arrays:
        raise ValueError("atoms is missing DOS data")
    
    if ax is None:
        fig, ax = plt.subplots()

    z_positions = atoms.get_positions()[:, 2]
    sort_order = np.argsort(z_positions)

    z_sorted = z_positions[sort_order]
    dos_sorted = atoms.arrays['dos_densities'][sort_order, :]

    z_edges = _bin_edges_from_centers(z_sorted)

    efermi = atoms.info["dos_efermi"]
    energies = atoms.info['dos_energies']
    if zero_at_efermi:
        energies = energies - efermi
    e_edges = _bin_edges_from_centers(energies)

    Z, E = np.meshgrid(z_edges, e_edges)

    e_min = -e_max
    vmax = np.max(dos_sorted[:, energies<e_max])

    ax.axhline(
    0,
    color=plt.rcParams['axes.edgecolor'],
    linewidth=plt.rcParams['axes.linewidth'],
    )

    pcm = ax.pcolormesh(Z, E, dos_sorted.T, shading='auto', cmap=cmap, norm=norm, vmax=vmax,)
    plt.colorbar(pcm, ax=ax)
    ax.set_ylim(e_min, e_max)
    
    return ax




def plot_site_dos_hist(
        atoms,
        *,
        zero_at_efermi=True,
        cmap="viridis", 
        ax = None,
        e_max=3,
        norm='symlog',
        ):
    if "dos_energies" not in atoms.info or "dos_densities" not in atoms.arrays:
        raise ValueError("atoms is missing DOS data")
    
    if ax is None:
        fig, ax = plt.subplots()

    z_positions = atoms.get_positions()[:, 2]
    sort_order = np.argsort(z_positions)

    z_sorted = z_positions[sort_order]
    dos_sorted = atoms.arrays['dos_densities'][sort_order, :]

    z_edges = _bin_edges_from_centers(z_sorted)

    efermi = atoms.info["dos_efermi"]
    energies = atoms.info['dos_energies']
    if zero_at_efermi:
        energies = energies - efermi
    e_edges = _bin_edges_from_centers(energies)

    Z, E = np.meshgrid(z_edges, e_edges)

    e_min = -e_max
    vmax = np.max(dos_sorted[:, energies<e_max])

    H, z_edges, e_edges = np.histogram2d(
        Z, E,
        bins=[100, 200],      # z bins, energy bins
        weights=dos_sorted
    )

    ax.pcolormesh(z_edges, e_edges, H.T, shading="auto")


    ax.axhline(
    0,
    color=plt.rcParams['axes.edgecolor'],
    linewidth=plt.rcParams['axes.linewidth'],
    )

    pcm = ax.pcolormesh(Z, E, dos_sorted.T, shading='auto', cmap=cmap, norm=norm, vmax=vmax,)
    plt.colorbar(pcm, ax=ax)
    ax.set_ylim(e_min, e_max)
    
    return ax