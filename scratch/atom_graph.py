import numpy as np 

import ase
import ase.io
from ase.neighborlist import primitive_neighbor_list

import jraph
import jax
import jax.numpy as jnp


def atoms_to_graph(
    atoms: ase.Atoms, cutoff: float, transfer_info: bool = True
):
    try:
        y_energy = atoms.get_potential_energy(force_consistent=True)
    except NotImplementedError:
        y_energy = atoms.get_potential_energy()
    y_force = atoms.get_forces(apply_constraint=False)
    try:
        # xx yy zz xy yz zx order
        # We expect this is eV/A^3 unit
        # (ASE automatically converts vasp kB to eV/A^3)
        # So we restore it
        y_stress = -1 * atoms.get_stress()
        y_stress = np.array([y_stress[[0, 1, 2, 5, 3, 4]]])
    except RuntimeError:
        y_stress = None

    pos = atoms.get_positions()
    cell = np.array(atoms.get_cell())

    # building neighbor list
    edge_src, edge_dst, edge_vec, shifts = primitive_neighbor_list(
        'ijDS', atoms.get_pbc(), cell, pos, cutoff, self_interaction=True
    )

    is_zero_idx = np.all(edge_vec == 0, axis=1)
    is_self_idx = edge_src == edge_dst
    non_trivials = ~(is_zero_idx & is_self_idx)
    cell_shift = np.array(shifts[non_trivials])

    edge_vec = edge_vec[non_trivials]
    edge_src = edge_src[non_trivials]
    edge_dst = edge_dst[non_trivials]
    edge_idx = np.array([edge_src, edge_dst])

    atomic_numbers = atoms.get_atomic_numbers()

    cell = np.array(cell)

    data = {
        "node_features": atomic_numbers,
        "atomic_numbers": atomic_numbers,
        "positions": pos,
        "edge_indices": edge_idx,
        "edge_vec": edge_vec,
        "energy": y_energy,
        "force": y_force,
        "stress": y_stress,
        "cell": cell,
        "cell_shift": cell_shift,
        "volume": np.einsum(
            'i,i', cell[0, :], np.cross(cell[1, :], cell[2, :])
        ),
        "num_nodes": len(atomic_numbers),
        "per_atom_energy": y_energy / len(pos),
    }

    if transfer_info and atoms.info is not None:
        data["info"] = atoms.info
    else:
        data["info"] = {}

    return data


def _to_graph_tuples(np_graphs):
    graphs = []
    for np_graph in np_graphs:
        nnodes = np_graph['num_nodes']
        nedges = int(len(np_graph['edge_vec']))
        graph = jraph.GraphsTuple(
                nodes={
                    'atomic_numbers': np_graph['atomic_numbers'], 
                    'force': np_graph['force']
                },
                edges={'edge_vec': np_graph['edge_vec']},
                receivers=np_graph['edge_indices'][0],
                senders=np_graph['edge_indices'][1],
                globals={
                    'energy': jnp.array([np_graph['energy']]), 
                    'stress': np_graph['stress'], 
                    'volume': jnp.array([np_graph['volume']])
                },
                n_node=jnp.array([nnodes]),
                n_edge=jnp.array([nedges]),
        )
        graphs.append(graph)
    return graphs


atoms_list = ase.io.read("./pt.extxyz", index=":")

cutoff = 5.0
np_graphs = [atoms_to_graph(atoms, cutoff, transfer_info=False) for atoms in atoms_list]
graphs = _to_graph_tuples(np_graphs)

batch = jraph.batch(graphs)

import flax
import flax.linen
import nequip_jax.nequip as nequip

import e3nn_jax as e3nn


class NequIP(flax.linen.Module):
    n_species: int = 118
    n_channel: int = 32
    lmax: int = 3
    n_layers: int = 3
    denominator: float = 1.0

    def _forward(self, x, rij, Z, edge_src, edge_dst, n_node):
        n_species = self.n_species
        n_channel = self.n_channel
        lmax = self.lmax
        x = e3nn.flax.Linear(
            irreps_out=f"{n_channel}x0e",
            irreps_in=f"{n_species}x0e"
        )(x)

        for t in range(self.n_layers):
            if t == self.n_layers - 1:
                out_irreps = e3nn.Irreps(f"{n_channel}x0e")
            else:
                out_irreps = n_channel * e3nn.Irreps.spherical_harmonics(lmax)
            x = nequip.NEQUIPLayerFlax(
                avg_num_neighbors=self.denominator,
                num_species=n_species,
                max_ell=3,
                output_irreps=out_irreps,
            )(rij, x, Z, edge_src, edge_dst)

        x = e3nn.flax.Linear(
            irreps_out=f"{n_channel // 2}x0e",
        )(x)
        atomic_energies = e3nn.flax.Linear(
            irreps_out="1x0e",
            irreps_in=f"{n_channel // 2}x0e",
        )(x)

        e = e3nn.scatter_sum(atomic_energies, nel=n_node)
        # first: for diff, second: what we need
        return e.array.sum(), e

    @flax.linen.compact
    def __call__(self, graphs):
        Z = graphs.nodes['atomic_numbers']
        rij = e3nn.IrrepsArray("1x1o", graphs.edges['edge_vec'])
        edge_src = graphs.senders
        edge_dst = graphs.receivers
        n_species = self.n_species

        x = jax.nn.one_hot(Z, n_species)
        x = e3nn.IrrepsArray(f"{n_species}x0e", x)

        fun = jax.value_and_grad(self._forward, 1, has_aux=True)
        _e, fij = fun(x, rij, Z, edge_src, edge_dst, graphs.n_node)
        _, energies = _e

        # compute per atom force from fij
        # compute virial
        return energies, fij


neq = NequIP()
variables = neq.init(jax.random.PRNGKey(777), batch)

e, f = neq.apply(variables, batch)

