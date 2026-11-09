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
"""Score a user-supplied structure with the confidence head.

A job with ``scoreStructure`` keeps the normal trunk (sequence, MSA,
templates), but the confidence head sees the coordinates from the given file
instead of a diffusion sample. Atoms that the file lacks are filled from a
diffusion sample superposed onto the given structure (see
``overlay_score_coordinates``).

Ported from the OpenDDE ``scoring`` branch
(``opendde/data/inference/score_structure.py``, commit 54faeef).
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch
from biotite.structure import AtomArray

from protenix.data.constants import ATOM37_ORDER
from protenix.data.template.custom_structure import (
    list_protein_chains,
    load_structure_chain,
    map_query_to_structure,
)
from protenix.utils.logger import get_logger

logger = get_logger(__name__)

SCORE_COORDINATE_KEY = "score_coordinate"
SCORE_COORDINATE_MASK_KEY = "score_coordinate_mask"
SCORE_STRUCTURE_INFO_KEY = "score_structure_info"
SCORE_STRUCTURE_KEYS = frozenset({"structurePath", "chainIds"})
# Atoms copied for mapped residues whose residue type differs from the query.
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


def validate_score_structure(job: Mapping[str, Any], *, location: str) -> None:
    """Validate a job-level ``scoreStructure`` entry (protein chains only)."""

    spec = job["scoreStructure"]
    if not isinstance(spec, dict):
        raise ValueError(f"{location} must be an object.")
    unknown = set(spec) - SCORE_STRUCTURE_KEYS
    if unknown:
        raise ValueError(
            f"{location} has unknown keys {sorted(unknown)}; "
            f"allowed keys are {sorted(SCORE_STRUCTURE_KEYS)}."
        )
    path = spec.get("structurePath")
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{location}.structurePath must be a non-empty path.")
    chain_ids = spec.get("chainIds")
    if chain_ids is not None:
        if (
            not isinstance(chain_ids, list)
            or not chain_ids
            or not all(isinstance(c, str) and c for c in chain_ids)
        ):
            raise ValueError(
                f"{location}.chainIds must be a non-empty list of strings."
            )
        if len(set(chain_ids)) != len(chain_ids):
            raise ValueError(f"{location}.chainIds must not repeat chains.")
    sequences = job.get("sequences")
    if not isinstance(sequences, list):
        return
    n_chains = 0
    for index, entity in enumerate(sequences):
        if not isinstance(entity, dict):
            continue
        for entity_type, body in entity.items():
            if entity_type != "proteinChain":
                raise ValueError(
                    f"{location}: only proteinChain entities can be scored; "
                    f"sequences[{index}] is {entity_type}."
                )
            if not isinstance(body, dict):
                continue
            for field in ("modifications", "glycans"):
                if body.get(field):
                    raise ValueError(
                        f"{location}: sequences[{index}] has {field!r}, which "
                        "scoring does not support; the modified residues are "
                        "outside the atom37 layout the structure file is read "
                        "into."
                    )
            if isinstance(body.get("count"), int) and body["count"] >= 0:
                n_chains += body["count"]
    if chain_ids is not None and n_chains and len(chain_ids) != n_chains:
        raise ValueError(
            f"{location}.chainIds has {len(chain_ids)} entries for {n_chains} model "
            "chain(s); give one file chain per model chain, in model chain order."
        )


def _model_protein_chains(sequences: Sequence[Mapping[str, Any]]) -> list[str]:
    """One query sequence per model chain, in model chain order."""
    chain_seqs = []
    for index, entity in enumerate(sequences):
        entity_type, body = next(iter(entity.items()))
        if entity_type != "proteinChain":
            raise ValueError(
                f"scoreStructure supports protein chains only; sequences[{index}] "
                f"is {entity_type}."
            )
        chain_seqs.extend([body["sequence"]] * int(body["count"]))
    return chain_seqs


def build_score_coordinates(
    job: Mapping[str, Any],
    atom_array: AtomArray,
    kalign_binary_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Map the coordinates of ``job["scoreStructure"]`` onto the model atoms.

    Model chains (entities in JSON order, copies in order) are paired with the
    file chains listed in ``chainIds``, or with the file's protein chains in
    file order. Residues are mapped by identity, unique substring, or pairwise
    alignment. Mapped residues of the same type contribute all their atoms;
    mapped residues of a different type contribute backbone atoms only.

    Returns:
        Coordinates [N_atom, 3] (float32), a mask [N_atom] (bool) of the model
        atoms that were found in the file, and a dict describing the mapping
        (written out as ``scoring_info.json``).
    """
    spec = job["scoreStructure"]
    path = spec["structurePath"]
    chain_seqs = _model_protein_chains(job["sequences"])
    file_chains = spec.get("chainIds") or list_protein_chains(path)
    if len(file_chains) != len(chain_seqs):
        raise ValueError(
            f"scoreStructure {path}: {len(chain_seqs)} model chain(s) but "
            f"{len(file_chains)} file chain(s) {list(file_chains)}; set chainIds "
            "to one file chain per model chain."
        )

    coords = np.zeros((len(atom_array), 3), dtype=np.float32)
    mask = np.zeros(len(atom_array), dtype=bool)
    chain_infos = []
    for asym_id, (query_seq, file_chain_id) in enumerate(zip(chain_seqs, file_chains)):
        chain = load_structure_chain(path, file_chain_id)
        mapping = map_query_to_structure(
            query_seq, chain.sequence, kalign_binary_path=kalign_binary_path
        )
        n_identical = 0
        for q_idx, t_idx in mapping.items():
            identical = query_seq[q_idx] == chain.sequence[t_idx]
            n_identical += identical
            atom_idx = np.flatnonzero(
                (atom_array.asym_id_int == asym_id) & (atom_array.res_id == q_idx + 1)
            )
            for i in atom_idx:
                name = str(atom_array.atom_name[i])
                if name not in ATOM37_ORDER or not (
                    identical or name in _BACKBONE_ATOMS
                ):
                    continue
                slot = ATOM37_ORDER[name]
                if chain.atom_mask[t_idx, slot]:
                    coords[i] = chain.atom_positions[t_idx, slot]
                    mask[i] = True
        chain_atoms = atom_array.asym_id_int == asym_id
        model_chain_id = str(atom_array.chain_id[chain_atoms][0])
        n_chain_atom = int(chain_atoms.sum())
        n_chain_mapped = int(mask[chain_atoms].sum())
        logger.info(
            f"scoreStructure {path} chain {file_chain_id} -> model chain "
            f"{model_chain_id}: {len(mapping)}/{len(query_seq)} "
            f"residues mapped ({n_identical} identical), "
            f"{n_chain_mapped}/{n_chain_atom} atoms."
        )
        chain_infos.append(
            {
                "file_chain_id": str(file_chain_id),
                "model_chain_id": model_chain_id,
                "model_asym_id": asym_id,
                "n_query_residue": len(query_seq),
                "n_mapped_residue": len(mapping),
                "n_identical_residue": int(n_identical),
                "n_model_atom": n_chain_atom,
                "n_mapped_atom": n_chain_mapped,
            }
        )

    if not mask.any():
        raise ValueError(f"scoreStructure {path}: no model atoms found in the file.")
    if not mask.all():
        logger.info(
            f"scoreStructure {path}: {int((~mask).sum())} model atom(s) missing in "
            "the file are filled from a superposed diffusion sample."
        )

    n_atom = int(mask.size)
    n_mapped = int(mask.sum())
    info = {
        "structure_path": str(path),
        "chain_ids": [chain["file_chain_id"] for chain in chain_infos],
        "n_model_atom": n_atom,
        "n_mapped_atom": n_mapped,
        "atom_coverage": round(n_mapped / n_atom, 6) if n_atom else 0.0,
        # The model skips diffusion under exactly this condition.
        "diffusion_skipped": bool(mask.all()),
        "chains": chain_infos,
    }
    return coords, mask, info


def pop_score_coordinates(
    input_feature_dict: MutableMapping[str, Any],
) -> tuple[
    MutableMapping[str, Any], Optional[torch.Tensor], Optional[torch.Tensor], bool
]:
    """
    Take the scoring tensors out of a feature dict before the trunk runs.

    The trunk never sees the coordinates, so they are removed from the dict
    passed on to it. The keys are popped in place and the same dict is returned,
    so that the inference loop's later ``del`` of the MSA and template features
    still frees that memory. The caller either passes a fresh deep copy per
    model seed or uses the dict once, so popping is safe.

    Returns:
        The feature dict without the scoring keys, the scoring coordinates,
        the scoring mask, and whether diffusion can be skipped (the file
        covers every model atom). Without scoring input the dict is returned
        untouched and the rest is ``None`` / ``False``, so the default
        inference path is unchanged.
    """
    score_coordinate = input_feature_dict.get(SCORE_COORDINATE_KEY)
    if score_coordinate is None:
        return input_feature_dict, None, None, False
    del input_feature_dict[SCORE_COORDINATE_KEY]
    score_mask = input_feature_dict.pop(SCORE_COORDINATE_MASK_KEY)
    return input_feature_dict, score_coordinate, score_mask, bool(score_mask.all())


def expand_score_coordinates(
    score_coordinate: torch.Tensor, n_sample: int
) -> torch.Tensor:
    """Repeat the given coordinates into ``[N_sample, N_atom, 3]`` (float32)."""
    return score_coordinate.float().unsqueeze(0).expand(n_sample, -1, -1).clone()


def _kabsch_transform(
    mobile: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation and translation that superpose ``mobile`` [N, 3] onto ``target``."""
    mobile_center = mobile.mean(dim=0)
    target_center = target.mean(dim=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    u, _, vh = torch.linalg.svd(covariance)
    sign = torch.sign(torch.linalg.det(u @ vh))
    correction = torch.diag(torch.stack([sign.new_ones(()), sign.new_ones(()), sign]))
    rotation = u @ correction @ vh
    translation = target_center - mobile_center @ rotation
    return rotation, translation


def overlay_score_coordinates(
    sampled: torch.Tensor,
    score_coordinate: torch.Tensor,
    score_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Replace sampled coordinates by the scored structure where it has atoms.

    Each sample [N_sample, N_atom, 3] is first superposed onto the scored
    atoms, so filled-in atoms sit in the frame of the given structure.
    """
    mask = score_mask.bool()
    target = score_coordinate.to(torch.float64)
    output = []
    for sample in sampled:
        sample64 = sample.to(torch.float64)
        if mask.sum() >= 3:
            rotation, translation = _kabsch_transform(sample64[mask], target[mask])
            sample64 = sample64 @ rotation + translation
        sample64[mask] = target[mask]
        output.append(sample64.to(sampled.dtype))
    return torch.stack(output, dim=0)
