# Copyright 2026 HanJun3000
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for reading protein chains from local PDB/mmCIF files.

Also provides the structure-writing helpers used by test_score_structure.py.
"""

import gzip

import numpy as np
import pytest
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

from protenix.data.constants import ATOM37_ORDER
from protenix.data.template.custom_structure import (
    list_protein_chains,
    load_structure_chain,
    map_query_to_structure,
)

THREE_LETTER = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
}
# Rough backbone geometry; only non-collinear N/CA/C and a CB are needed.
ATOM_OFFSETS = {
    "N": (-1.2, 0.8, 0.0),
    "CA": (0.0, 0.0, 0.0),
    "C": (1.2, 0.8, 0.0),
    "O": (1.2, 2.0, 0.0),
    "CB": (0.0, -1.0, 1.2),
}


def _backbone_atoms(sequence: str, chain_id: str = "A", start: int = 1):
    """Rows of (chain_id, res_id, res_name, atom_name, coord) for a chain."""
    atoms = []
    for i, code in enumerate(sequence):
        res_name = THREE_LETTER[code]
        for atom_name, offset in ATOM_OFFSETS.items():
            if res_name == "GLY" and atom_name == "CB":
                continue
            coord = np.array(offset) + np.array([3.8 * i, 0.5 * (i % 2), 0.0])
            atoms.append((chain_id, start + i, res_name, atom_name, coord))
    return atoms


def _write_pdb(path, atoms) -> None:
    atom_array = AtomArray(len(atoms))
    for i, (chain_id, res_id, res_name, atom_name, coord) in enumerate(atoms):
        atom_array.chain_id[i] = chain_id
        atom_array.res_id[i] = res_id
        atom_array.res_name[i] = res_name
        atom_array.atom_name[i] = atom_name
        atom_array.element[i] = atom_name[0]
        atom_array.coord[i] = coord
    pdb_file = PDBFile()
    pdb_file.set_structure(atom_array)
    pdb_file.write(str(path))


def _write_minimal_cif(path, atoms) -> None:
    """Write an mmCIF with only the _atom_site columns design tools tend to emit."""
    lines = [
        "data_design",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    ]
    for i, (chain_id, res_id, res_name, atom_name, coord) in enumerate(atoms):
        lines.append(
            f"ATOM {i + 1} {atom_name} {res_name} {chain_id} {res_id} "
            f"{coord[0]:.3f} {coord[1]:.3f} {coord[2]:.3f}"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def test_pdb_chain_gives_sequence_and_atom37_coordinates(tmp_path):
    path = tmp_path / "chain.pdb"
    atoms = _backbone_atoms("ACG")
    _write_pdb(path, atoms)

    chain = load_structure_chain(str(path), "A")

    assert chain.sequence == "ACG"
    assert chain.atom_positions.shape == (3, len(ATOM37_ORDER), 3)
    # Glycine has no CB; the other two residues do.
    assert not chain.atom_mask[2, ATOM37_ORDER["CB"]]
    assert chain.atom_mask[0, ATOM37_ORDER["CB"]]
    expected = [a for a in atoms if a[1] == 2 and a[3] == "CA"][0][4]
    np.testing.assert_allclose(
        chain.atom_positions[1, ATOM37_ORDER["CA"]], expected, atol=1e-3
    )
    # Atoms outside atom37 are never set.
    assert chain.atom_mask.sum() == sum(1 for a in atoms if a[3] in ATOM37_ORDER)


def test_minimal_cif_is_read(tmp_path):
    path = tmp_path / "design.cif"
    _write_minimal_cif(path, _backbone_atoms("ACDE"))

    chain = load_structure_chain(str(path), "A")

    assert chain.sequence == "ACDE"
    assert chain.atom_mask[:, ATOM37_ORDER["CA"]].all()


def test_gzipped_file_is_read(tmp_path):
    plain = tmp_path / "chain.pdb"
    _write_pdb(plain, _backbone_atoms("ACDE"))
    gz_path = tmp_path / "chain.pdb.gz"
    with open(plain, "rb") as src, gzip.open(gz_path, "wb") as dst:
        dst.write(src.read())

    assert load_structure_chain(str(gz_path), "A").sequence == "ACDE"


def test_chains_are_listed_in_file_order(tmp_path):
    path = tmp_path / "dimer.pdb"
    _write_pdb(
        path, _backbone_atoms("ACDE", chain_id="B") + _backbone_atoms("GHIK", chain_id="A")
    )

    assert list_protein_chains(str(path)) == ["B", "A"]


def test_single_chain_needs_no_chain_id(tmp_path):
    path = tmp_path / "mono.pdb"
    _write_pdb(path, _backbone_atoms("ACDE"))

    assert load_structure_chain(str(path)).sequence == "ACDE"


def test_ambiguous_chain_and_unknown_chain_are_reported(tmp_path):
    path = tmp_path / "dimer.pdb"
    _write_pdb(
        path, _backbone_atoms("ACDE", chain_id="A") + _backbone_atoms("GHIK", chain_id="B")
    )

    with pytest.raises(ValueError, match="several protein chains"):
        load_structure_chain(str(path))
    with pytest.raises(ValueError, match="not found among protein chains"):
        load_structure_chain(str(path), "Z")


def test_selenomethionine_maps_to_methionine_sd(tmp_path):
    path = tmp_path / "mse.cif"
    atoms = [
        ("A", 1, "MSE", "N", np.array([0.0, 0.0, 0.0])),
        ("A", 1, "MSE", "CA", np.array([1.0, 0.0, 0.0])),
        ("A", 1, "MSE", "C", np.array([2.0, 0.0, 0.0])),
        ("A", 1, "MSE", "SE", np.array([3.0, 1.0, 0.0])),
    ]
    _write_minimal_cif(path, atoms)

    chain = load_structure_chain(str(path), "A")

    assert chain.sequence == "M"
    assert chain.atom_mask[0, ATOM37_ORDER["SD"]]
    np.testing.assert_allclose(
        chain.atom_positions[0, ATOM37_ORDER["SD"]], [3.0, 1.0, 0.0], atol=1e-3
    )


def test_first_altloc_wins(tmp_path):
    path = tmp_path / "altloc.cif"
    atoms = [
        ("A", 1, "ALA", "N", np.array([0.0, 0.0, 0.0])),
        ("A", 1, "ALA", "CA", np.array([1.0, 0.0, 0.0])),
        ("A", 1, "ALA", "CA", np.array([9.0, 9.0, 9.0])),
    ]
    _write_minimal_cif(path, atoms)

    chain = load_structure_chain(str(path), "A")

    np.testing.assert_allclose(
        chain.atom_positions[0, ATOM37_ORDER["CA"]], [1.0, 0.0, 0.0], atol=1e-3
    )


def test_non_amino_acid_residues_are_skipped(tmp_path):
    path = tmp_path / "with_water.cif"
    atoms = _backbone_atoms("AC") + [
        ("A", 99, "HOH", "O", np.array([20.0, 0.0, 0.0])),
    ]
    _write_minimal_cif(path, atoms)

    assert load_structure_chain(str(path), "A").sequence == "AC"


def test_unsupported_format_and_missing_file(tmp_path):
    bad = tmp_path / "chain.xyz"
    bad.write_text("")
    with pytest.raises(ValueError, match="Unsupported structure format"):
        load_structure_chain(str(bad))
    with pytest.raises(FileNotFoundError, match="Structure file not found"):
        load_structure_chain(str(tmp_path / "nope.pdb"))


def test_file_without_protein_is_reported(tmp_path):
    path = tmp_path / "empty.cif"
    _write_minimal_cif(path, [("A", 1, "HOH", "O", np.array([0.0, 0.0, 0.0]))])
    with pytest.raises(ValueError, match="No protein chains found"):
        load_structure_chain(str(path))


def test_mapping_prefers_identity_then_substring_then_alignment():
    # Identical sequences map one to one.
    assert map_query_to_structure("ACDEFG", "ACDEFG") == {i: i for i in range(6)}
    # A truncated structure maps onto the matching window of the query.
    assert map_query_to_structure("ACDEFG", "DEF") == {2: 0, 3: 1, 4: 2}
    # A query shorter than the structure maps into it.
    assert map_query_to_structure("DEF", "ACDEFG") == {0: 2, 1: 3, 2: 4}
    # A point mutant is neither, so the sequences are aligned. With no kalign
    # binary this exercises the Biopython fallback.
    assert map_query_to_structure("ACDEFGHIK", "ACDAFGHIK") == {
        i: i for i in range(9)
    }


def test_mapping_handles_an_internal_deletion():
    # The structure is missing one residue in the middle.
    mapping = map_query_to_structure("ACDEFGHIK", "ACDEGHIK")
    assert mapping[0] == 0
    # Residues after the gap are offset by one in the structure.
    assert mapping[8] == 7
    assert 4 not in mapping or mapping[4] != 4


def test_explicit_indices_take_precedence():
    mapping = map_query_to_structure(
        "ACDEFG", "ACDEFG", query_indices=[0, 1], structure_indices=[3, 4]
    )
    assert mapping == {0: 3, 1: 4}
    with pytest.raises(ValueError, match="out of range"):
        map_query_to_structure(
            "ACDEFG", "ACD", query_indices=[0], structure_indices=[9]
        )
