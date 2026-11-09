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
"""Read protein chains from a local PDB or mmCIF file.

Unlike template search hits, these structures need no PDB ID, release date, or
SEQRES records: the sequence is taken from the residues present in the file.
Used by structure scoring (``protenix.data.inference.score_structure``) to turn
a user-supplied file into coordinates in the model's atom layout.

Ported from the OpenDDE ``scoring`` branch
(``opendde/data/template/custom_template.py``, commit 54faeef).
"""

from __future__ import annotations

import dataclasses
import functools
import gzip
import io
import os
from typing import Any, Dict, Optional, Sequence

import biotite.structure.info as struc_info
import biotite.structure.io.pdbx as pdbx
import numpy as np

from protenix.data.constants import ATOM37_NUM, ATOM37_ORDER
from protenix.data.tools.kalign import Kalign
from protenix.utils.logger import get_logger

logger = get_logger(__name__)

_CIF_SUFFIXES = (".cif", ".mmcif")
_PDB_SUFFIXES = (".pdb", ".ent")
_MISSING_CIF_VALUES = (".", "?")

# Deterministic fallback alignment parameters, used when kalign is unavailable.
_FALLBACK_OPEN_GAP_SCORE = -11.0
_FALLBACK_EXTEND_GAP_SCORE = -1.0


@dataclasses.dataclass(frozen=True)
class StructureChain:
    """Observed amino-acid residues of one chain in a structure file."""

    sequence: str
    # atom37 coordinates and mask: [N_res, 37, 3] and [N_res, 37]
    atom_positions: np.ndarray
    atom_mask: np.ndarray


def _read_text(path: str) -> str:
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return f.read()
    with open(path, "r") as f:
        return f.read()


def _structure_format(path: str) -> str:
    name = path.lower()
    if name.endswith(".gz"):
        name = name[: -len(".gz")]
    if name.endswith(_CIF_SUFFIXES):
        return "cif"
    if name.endswith(_PDB_SUFFIXES):
        return "pdb"
    raise ValueError(
        f"Unsupported structure format: {path}. Use .cif, .mmcif, .pdb, or .ent "
        "(optionally gzipped)."
    )


def _cif_column(
    atom_site: Any, names: Sequence[str], n_atom: int, default: str = ""
) -> np.ndarray:
    """Return the first present column, treating '.'/'?' as missing values."""
    for name in names:
        if name in atom_site:
            values = atom_site[name].as_array(str)
            if not np.all(np.isin(values, _MISSING_CIF_VALUES)):
                return values
    return np.full(n_atom, default, dtype=object)


def _read_cif_atoms(text: str) -> Dict[str, np.ndarray]:
    """Read ``_atom_site`` leniently; design tools often omit most columns."""
    block = pdbx.CIFFile.read(io.StringIO(text)).block
    if "atom_site" not in block:
        raise ValueError("mmCIF file has no _atom_site records.")
    atom_site = block["atom_site"]
    n_atom = atom_site.row_count
    atoms = {
        "chain_id": _cif_column(atom_site, ("auth_asym_id", "label_asym_id"), n_atom),
        "res_id": _cif_column(atom_site, ("auth_seq_id", "label_seq_id"), n_atom),
        "ins_code": _cif_column(atom_site, ("pdbx_PDB_ins_code",), n_atom),
        "res_name": _cif_column(atom_site, ("auth_comp_id", "label_comp_id"), n_atom),
        "atom_name": _cif_column(atom_site, ("auth_atom_id", "label_atom_id"), n_atom),
        "coord": np.stack(
            [atom_site[f"Cartn_{axis}"].as_array(np.float32) for axis in "xyz"],
            axis=-1,
        ),
    }
    model = _cif_column(atom_site, ("pdbx_PDB_model_num",), n_atom)
    keep = model == model[0]
    return {key: value[keep] for key, value in atoms.items()}


def _read_pdb_atoms(text: str) -> Dict[str, np.ndarray]:
    """Read ATOM/HETATM records of the first model by fixed PDB columns.

    Only the columns up to the coordinates are required, so truncated records
    (e.g. membrane pseudo-atoms without occupancy/B-factor) are accepted.
    """
    columns: Dict[str, list] = {
        "chain_id": [],
        "res_id": [],
        "ins_code": [],
        "res_name": [],
        "atom_name": [],
        "coord": [],
    }
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        columns["atom_name"].append(line[12:16].strip())
        columns["res_name"].append(line[17:20].strip())
        columns["chain_id"].append(line[21].strip())
        columns["res_id"].append(line[22:26].strip())
        columns["ins_code"].append(line[26].strip())
        columns["coord"].append(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )
    atoms = {key: np.array(value) for key, value in columns.items()}
    atoms["coord"] = atoms["coord"].astype(np.float32).reshape(-1, 3)
    return atoms


@functools.lru_cache(maxsize=1)
def _amino_acid_names() -> frozenset:
    return frozenset(struc_info.amino_acid_names())


@functools.lru_cache(maxsize=None)
def _one_letter_code(res_name: str) -> Optional[str]:
    """Return the one-letter code of an amino acid, or None for other residues."""
    if res_name not in _amino_acid_names():
        return None
    code = struc_info.one_letter_code(res_name)
    return code if code and len(code) == 1 else "X"


def _read_protein_residues(
    structure_path: str,
) -> Dict[str, Dict[tuple, Dict[str, Any]]]:
    """Group the amino-acid atoms of a structure file by chain and residue."""
    if not os.path.isfile(structure_path):
        raise FileNotFoundError(f"Structure file not found: {structure_path}")
    text = _read_text(structure_path)
    if _structure_format(structure_path) == "cif":
        atoms = _read_cif_atoms(text)
    else:
        atoms = _read_pdb_atoms(text)

    # Group atoms into residues in order of appearance, per chain.
    residues: Dict[str, Dict[tuple, Dict[str, Any]]] = {}
    for chain, res_id, ins_code, res_name, atom_name, coord in zip(
        atoms["chain_id"],
        atoms["res_id"],
        atoms["ins_code"],
        atoms["res_name"],
        atoms["atom_name"],
        atoms["coord"],
    ):
        chain, res_name = str(chain), str(res_name)
        code = _one_letter_code(res_name)
        if code is None:
            continue
        ins_code = "" if ins_code in _MISSING_CIF_VALUES else str(ins_code)
        residue = residues.setdefault(chain, {}).setdefault(
            (str(res_id), ins_code), {"code": code, "atoms": {}}
        )
        atom_name = str(atom_name)
        if atom_name == "SE" and res_name == "MSE":
            atom_name = "SD"
        # Keep the first occurrence, i.e. the first alternate location.
        if atom_name in ATOM37_ORDER and atom_name not in residue["atoms"]:
            residue["atoms"][atom_name] = coord

    if not residues:
        raise ValueError(f"No protein chains found in {structure_path}.")
    return residues


def list_protein_chains(structure_path: str) -> list[str]:
    """Return the author IDs of the protein chains in a file, in file order."""
    return list(_read_protein_residues(structure_path))


def load_structure_chain(
    structure_path: str, chain_id: Optional[str] = None
) -> StructureChain:
    """
    Read the observed amino-acid residues of one chain from a PDB/mmCIF file.

    Args:
        structure_path: Path to a .cif/.mmcif/.pdb/.ent file, optionally gzipped.
        chain_id: Author chain ID. May be omitted if the file has exactly one
            protein chain.

    Returns:
        The chain's one-letter sequence (in file order) with atom37 coordinates.
    """
    residues = _read_protein_residues(structure_path)
    if chain_id is None:
        if len(residues) != 1:
            raise ValueError(
                f"{structure_path} has several protein chains {sorted(residues)}; "
                "set the chain ID to select one."
            )
        chain_id = next(iter(residues))
    if chain_id not in residues:
        raise ValueError(
            f"Chain {chain_id!r} not found among protein chains "
            f"{sorted(residues)} in {structure_path}."
        )

    chain_residues = list(residues[chain_id].values())
    n_res = len(chain_residues)
    atom_positions = np.zeros((n_res, ATOM37_NUM, 3), dtype=np.float32)
    atom_mask = np.zeros((n_res, ATOM37_NUM), dtype=np.float32)
    for i, residue in enumerate(chain_residues):
        for atom_name, coord in residue["atoms"].items():
            atom_positions[i, ATOM37_ORDER[atom_name]] = coord
            atom_mask[i, ATOM37_ORDER[atom_name]] = 1.0
    sequence = "".join(residue["code"] for residue in chain_residues)
    return StructureChain(sequence, atom_positions, atom_mask)


def _align_with_biopython(query_seq: str, structure_seq: str) -> Dict[int, int]:
    """Deterministic global alignment fallback when kalign is not available."""
    from Bio.Align import PairwiseAligner, substitution_matrices

    matrix = substitution_matrices.load("BLOSUM62")
    alphabet = set(matrix.alphabet)

    def sanitize(sequence: str) -> str:
        # Keep the length, so alignment indices stay valid for the original.
        return "".join(c if c in alphabet else "X" for c in sequence)

    aligner = PairwiseAligner(
        mode="global",
        open_gap_score=_FALLBACK_OPEN_GAP_SCORE,
        extend_gap_score=_FALLBACK_EXTEND_GAP_SCORE,
    )
    aligner.substitution_matrix = matrix
    alignment = aligner.align(sanitize(query_seq), sanitize(structure_seq))[0]
    query_blocks, structure_blocks = alignment.aligned
    mapping = {}
    for (q_start, q_stop), (t_start, _) in zip(query_blocks, structure_blocks):
        for offset in range(int(q_stop) - int(q_start)):
            mapping[int(q_start) + offset] = int(t_start) + offset
    return mapping


def _align_sequences(
    query_seq: str, structure_seq: str, kalign_binary_path: Optional[str]
) -> Dict[int, int]:
    """Align two sequences with kalign, or Biopython if kalign is unavailable."""
    if kalign_binary_path:
        try:
            q_aln, t_aln = Kalign(binary_path=kalign_binary_path).align(
                [query_seq, structure_seq]
            )
        except Exception as exc:  # binary missing, too short, or kalign failed
            logger.warning(
                f"kalign unavailable or failed ({exc}); falling back to "
                "Bio.Align.PairwiseAligner."
            )
        else:
            mapping = {}
            q_idx, t_idx = -1, -1
            for qa, ta in zip(q_aln, t_aln):
                if qa != "-":
                    q_idx += 1
                if ta != "-":
                    t_idx += 1
                if qa != "-" and ta != "-":
                    mapping[q_idx] = t_idx
            return mapping
    return _align_with_biopython(query_seq, structure_seq)


def map_query_to_structure(
    query_seq: str,
    structure_seq: str,
    query_indices: Optional[Sequence[int]] = None,
    structure_indices: Optional[Sequence[int]] = None,
    kalign_binary_path: Optional[str] = None,
) -> Dict[int, int]:
    """
    Map 0-based query residue indices to 0-based structure residue indices.

    Explicit indices take precedence. Otherwise an exact or unique substring
    match (e.g. a truncated structure) is used, and the sequences are aligned
    only when neither applies.
    """
    if query_indices is not None and structure_indices is not None:
        for name, indices, length in (
            ("queryIndices", query_indices, len(query_seq)),
            ("structureIndices", structure_indices, len(structure_seq)),
        ):
            out_of_range = [i for i in indices if i >= length]
            if out_of_range:
                raise ValueError(
                    f"{name} {out_of_range[:5]} out of range for length {length}."
                )
        return dict(zip(query_indices, structure_indices))

    if query_seq.count(structure_seq) == 1:
        offset = query_seq.index(structure_seq)
        return {offset + i: i for i in range(len(structure_seq))}
    if structure_seq.count(query_seq) == 1:
        offset = structure_seq.index(query_seq)
        return {i: offset + i for i in range(len(query_seq))}

    return _align_sequences(query_seq, structure_seq, kalign_binary_path)
