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
"""Tests for scoring a user-supplied structure with the confidence head."""

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray

from protenix.data.inference.score_structure import (
    build_score_coordinates,
    expand_score_coordinates,
    overlay_score_coordinates,
    pop_score_coordinates,
    SCORE_COORDINATE_KEY,
    SCORE_COORDINATE_MASK_KEY,
    validate_score_structure,
)
from tests.test_custom_structure import ATOM_OFFSETS, _backbone_atoms, _write_pdb


def _model_atom_array(chains, with_oxt=True) -> AtomArray:
    """Atoms the JSON parser would build: N/CA/C/O(/CB) per residue, OXT at the end."""
    rows = []
    for asym_id, (chain_id, sequence) in enumerate(chains):
        for i, code in enumerate(sequence):
            names = [
                name for name in ATOM_OFFSETS if not (code == "G" and name == "CB")
            ]
            if with_oxt and i == len(sequence) - 1:
                names.append("OXT")
            rows += [(asym_id, chain_id, i + 1, name) for name in names]
    atom_array = AtomArray(len(rows))
    atom_array.add_annotation("asym_id_int", int)
    for i, (asym_id, chain_id, res_id, name) in enumerate(rows):
        atom_array.asym_id_int[i] = asym_id
        atom_array.chain_id[i] = chain_id
        atom_array.res_id[i] = res_id
        atom_array.atom_name[i] = name
    return atom_array


def _job(path, *sequences, chain_ids=None):
    spec = {"structurePath": str(path)}
    if chain_ids is not None:
        spec["chainIds"] = chain_ids
    return {
        "name": "score",
        "sequences": [{"proteinChain": {"sequence": s, "count": 1}} for s in sequences],
        "scoreStructure": spec,
    }


# --- Mapping a file onto the model atom layout ---------------------------------


def test_dimer_coordinates_map_by_chain_and_atom(tmp_path):
    path = tmp_path / "dimer.pdb"
    atoms_a = _backbone_atoms("ACDE", chain_id="A")
    atoms_b = [
        (c, r, n, a, x + 50.0)
        for c, r, n, a, x in _backbone_atoms("GHIK", chain_id="B")
    ]
    _write_pdb(path, atoms_a + atoms_b)
    atom_array = _model_atom_array([("A", "ACDE"), ("B", "GHIK")])

    coords, mask, info = build_score_coordinates(
        _job(path, "ACDE", "GHIK"), atom_array
    )

    # Every atom except the two OXT (absent from the file) is taken from it.
    assert (~mask).sum() == 2
    assert set(atom_array.atom_name[~mask]) == {"OXT"}
    ca_b2 = np.flatnonzero(
        (atom_array.asym_id_int == 1)
        & (atom_array.res_id == 2)
        & (atom_array.atom_name == "CA")
    )[0]
    expected = [c for c in atoms_b if c[1] == 2 and c[3] == "CA"][0][4]
    np.testing.assert_allclose(coords[ca_b2], expected, atol=1e-3)
    # The report describes the mapping.
    assert info["diffusion_skipped"] is False
    assert info["n_mapped_atom"] == int(mask.sum())
    assert [c["file_chain_id"] for c in info["chains"]] == ["A", "B"]
    assert [c["n_identical_residue"] for c in info["chains"]] == [4, 4]
    assert 0.0 < info["atom_coverage"] < 1.0


def test_full_coverage_reports_diffusion_skipped(tmp_path):
    path = tmp_path / "mono.pdb"
    _write_pdb(path, _backbone_atoms("ACDE"))
    # No OXT in the model layout, so the file covers every atom.
    atom_array = _model_atom_array([("A", "ACDE")], with_oxt=False)

    _, mask, info = build_score_coordinates(_job(path, "ACDE"), atom_array)

    assert mask.all()
    assert info["diffusion_skipped"] is True
    assert info["atom_coverage"] == 1.0


def test_chain_ids_pick_file_chains_in_model_order(tmp_path):
    path = tmp_path / "dimer.pdb"
    atoms_a = _backbone_atoms("ACDE", chain_id="A")
    atoms_b = [
        (c, r, n, a, x + 50.0)
        for c, r, n, a, x in _backbone_atoms("GHIK", chain_id="B")
    ]
    _write_pdb(path, atoms_a + atoms_b)
    # Model chain order is B's sequence first.
    atom_array = _model_atom_array([("A", "GHIK"), ("B", "ACDE")])

    coords, mask, _ = build_score_coordinates(
        _job(path, "GHIK", "ACDE", chain_ids=["B", "A"]), atom_array
    )

    first_ca = np.flatnonzero(
        (atom_array.asym_id_int == 0) & (atom_array.atom_name == "CA")
    )[0]
    assert mask[first_ca]
    assert coords[first_ca][0] > 40.0  # came from file chain B


def test_mismatched_residue_keeps_backbone_only(tmp_path, monkeypatch):
    path = tmp_path / "mutant.pdb"
    _write_pdb(path, _backbone_atoms("ACDE", chain_id="A"))
    # D3 -> A: not identical, not a substring; map position-wise without aligning.
    from protenix.data.inference import score_structure

    monkeypatch.setattr(
        score_structure,
        "map_query_to_structure",
        lambda q, t, **kwargs: {i: i for i in range(len(q))},
    )
    atom_array = _model_atom_array([("A", "ACAE")])

    _, mask, info = build_score_coordinates(_job(path, "ACAE"), atom_array)

    res3 = atom_array.res_id == 3
    assert mask[res3 & np.isin(atom_array.atom_name, ["N", "CA", "C", "O"])].all()
    assert not mask[res3 & (atom_array.atom_name == "CB")].any()
    assert info["chains"][0]["n_identical_residue"] == 3


def test_mismatched_residue_without_monkeypatch_uses_the_aligner(tmp_path):
    """Same as above, but through the real alignment fallback (no kalign needed)."""
    path = tmp_path / "mutant.pdb"
    _write_pdb(path, _backbone_atoms("ACDEFGHIK", chain_id="A"))
    atom_array = _model_atom_array([("A", "ACDAFGHIK")])

    _, mask, info = build_score_coordinates(_job(path, "ACDAFGHIK"), atom_array)

    assert info["chains"][0]["n_mapped_residue"] == 9
    assert info["chains"][0]["n_identical_residue"] == 8
    res4 = atom_array.res_id == 4
    assert mask[res4 & np.isin(atom_array.atom_name, ["N", "CA", "C", "O"])].all()
    assert not mask[res4 & (atom_array.atom_name == "CB")].any()


def test_truncated_file_leaves_unmapped_residues_masked_out(tmp_path):
    path = tmp_path / "trunc.pdb"
    # The file holds residues 3-5 of a 6-residue query.
    _write_pdb(path, _backbone_atoms("DEF", chain_id="A"))
    atom_array = _model_atom_array([("A", "ACDEFG")], with_oxt=False)

    _, mask, info = build_score_coordinates(_job(path, "ACDEFG"), atom_array)

    assert info["chains"][0]["n_mapped_residue"] == 3
    assert mask[np.isin(atom_array.res_id, [3, 4, 5])].all()
    assert not mask[np.isin(atom_array.res_id, [1, 2, 6])].any()


def test_chain_count_mismatch_is_reported(tmp_path):
    path = tmp_path / "monomer.pdb"
    _write_pdb(path, _backbone_atoms("ACDE", chain_id="A"))
    atom_array = _model_atom_array([("A", "ACDE"), ("B", "ACDE")])

    with pytest.raises(ValueError, match="2 model chain"):
        build_score_coordinates(_job(path, "ACDE", "ACDE"), atom_array)


def test_no_overlap_is_reported(tmp_path):
    path = tmp_path / "other.pdb"
    _write_pdb(path, _backbone_atoms("GGG", chain_id="A"))
    # The query shares no residue position with the file after alignment.
    atom_array = _model_atom_array([("A", "K")], with_oxt=False)
    atom_array.atom_name[:] = "ZZ"  # no atom name is in atom37

    with pytest.raises(ValueError, match="no model atoms found"):
        build_score_coordinates(_job(path, "K"), atom_array)


# --- Filling missing atoms from a diffusion sample ----------------------------


def _random_rotation(generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_overlay_full_mask_returns_given_coordinates():
    generator = torch.Generator().manual_seed(0)
    target = torch.randn(20, 3, generator=generator)
    sampled = torch.randn(2, 20, 3, generator=generator)

    out = overlay_score_coordinates(sampled, target, torch.ones(20, dtype=torch.bool))

    assert out.shape == (2, 20, 3)
    torch.testing.assert_close(out, target.expand(2, -1, -1))


def test_overlay_places_missing_atoms_in_given_frame():
    generator = torch.Generator().manual_seed(1)
    target = torch.randn(30, 3, generator=generator, dtype=torch.float64) * 5
    rotation = _random_rotation(generator)
    # The sample is the true structure in another frame.
    sampled = (
        target @ rotation + torch.tensor([10.0, -3.0, 7.0], dtype=torch.float64)
    )[None]
    mask = torch.ones(30, dtype=torch.bool)
    mask[[3, 17, 29]] = False

    out = overlay_score_coordinates(sampled.float(), target.float(), mask)

    torch.testing.assert_close(out[0], target.float(), atol=1e-4, rtol=0)


def test_overlay_never_leaves_missing_atoms_at_the_origin():
    generator = torch.Generator().manual_seed(2)
    target = torch.randn(12, 3, generator=generator) * 4 + 100.0
    sampled = torch.randn(1, 12, 3, generator=generator)
    mask = torch.ones(12, dtype=torch.bool)
    mask[5] = False

    out = overlay_score_coordinates(sampled, target, mask)

    # The filled atom sits near the given structure, not at (0, 0, 0).
    assert out[0, 5].norm() > 50.0


# --- Model plumbing ------------------------------------------------------------


def test_pop_score_coordinates_strips_keys_and_detects_full_coverage():
    feats = {
        "residue_index": torch.zeros(3),
        SCORE_COORDINATE_KEY: torch.zeros(5, 3),
        SCORE_COORDINATE_MASK_KEY: torch.ones(5, dtype=torch.bool),
    }

    stripped, coords, mask, skip = pop_score_coordinates(feats)

    assert set(stripped) == {"residue_index"}
    assert skip is True
    assert coords.shape == (5, 3) and mask.shape == (5,)
    # Popped in place and the same dict returned, so the inference loop's
    # later `del` of the MSA/template features still frees that memory.
    assert stripped is feats
    assert SCORE_COORDINATE_KEY not in feats


def test_pop_score_coordinates_detects_partial_coverage():
    mask = torch.ones(5, dtype=torch.bool)
    mask[2] = False
    feats = {
        SCORE_COORDINATE_KEY: torch.zeros(5, 3),
        SCORE_COORDINATE_MASK_KEY: mask,
    }

    assert pop_score_coordinates(feats)[3] is False


def test_normal_path_is_untouched_without_scoring_input():
    feats = {"residue_index": torch.zeros(3)}

    stripped, coords, mask, skip = pop_score_coordinates(feats)

    # Same object, nothing added or removed, and no scoring behaviour.
    assert stripped is feats
    assert (coords, mask, skip) == (None, None, False)


def test_expand_score_coordinates_shape_and_dtype():
    coords = torch.zeros(7, 3, dtype=torch.float64)

    out = expand_score_coordinates(coords, 3)

    assert out.shape == (3, 7, 3)
    assert out.dtype == torch.float32
    # A real tensor, not a view: the model may write into it.
    assert out.is_contiguous()


# --- Validation ----------------------------------------------------------------


def test_validation_accepts_a_well_formed_job():
    validate_score_structure(
        _job("/x/dimer.cif", "ACDE", "GHIK", chain_ids=["A", "B"]),
        location="scoreStructure",
    )
    # chainIds is optional.
    validate_score_structure(_job("/x/dimer.cif", "ACDE"), location="scoreStructure")


def test_validation_rejects_unknown_keys():
    job = _job("/x/dimer.cif", "ACDE")
    job["scoreStructure"] = {"structurePath": "/x.cif", "chain": "A"}
    with pytest.raises(ValueError, match="unknown keys"):
        validate_score_structure(job, location="scoreStructure")


def test_validation_rejects_a_bad_path_and_bad_chain_ids():
    job = _job("/x/dimer.cif", "ACDE")
    job["scoreStructure"] = {"structurePath": "  "}
    with pytest.raises(ValueError, match="non-empty path"):
        validate_score_structure(job, location="scoreStructure")

    job["scoreStructure"] = {"structurePath": "/x.cif", "chainIds": []}
    with pytest.raises(ValueError, match="non-empty list"):
        validate_score_structure(job, location="scoreStructure")

    job["scoreStructure"] = {"structurePath": "/x.cif", "chainIds": ["A", "A"]}
    with pytest.raises(ValueError, match="must not repeat"):
        validate_score_structure(job, location="scoreStructure")


def test_validation_rejects_a_chain_count_mismatch():
    wrong_count = _job("/x/dimer.cif", "ACDE", "GHIK", chain_ids=["A"])
    with pytest.raises(ValueError, match="1 entries for 2 model chain"):
        validate_score_structure(wrong_count, location="scoreStructure")


def test_validation_rejects_non_protein_entities():
    ligand = _job("/x/dimer.cif", "ACDE")
    ligand["sequences"].append({"ligand": {"ligand": "CCD_ATP", "count": 1}})
    with pytest.raises(ValueError, match="only proteinChain"):
        validate_score_structure(ligand, location="scoreStructure")


def test_validation_rejects_modified_residues():
    job = _job("/x/mono.cif", "ACDE")
    job["sequences"][0]["proteinChain"]["modifications"] = [
        {"ptmType": "CCD_SEP", "ptmPosition": 2}
    ]
    with pytest.raises(ValueError, match="modifications"):
        validate_score_structure(job, location="scoreStructure")


# --- The layout assumption the mapper rests on ---------------------------------


def test_model_atom_layout_matches_the_mapper_assumptions():
    """`asym_id_int` follows JSON entity-then-copy order and `res_id` is 1..L.

    `build_score_coordinates` indexes model atoms by (asym_id_int, res_id,
    atom_name), so this is the assumption the whole mapping rests on. Needs the
    CCD components file, so it is skipped on a partial install.
    """
    import os

    from configs.configs_data import data_configs

    if not os.path.isfile(data_configs["ccd_components_file"]):
        pytest.skip("CCD components file not available")

    from protenix.data.inference.json_to_feature import SampleDictToFeatures

    job = {
        "name": "layout",
        "sequences": [
            {"proteinChain": {"sequence": "ACDE", "count": 2}},
            {"proteinChain": {"sequence": "GHIK", "count": 1}},
        ],
    }
    atom_array = SampleDictToFeatures(job).get_atom_array()

    asym_ids = np.asarray(atom_array.asym_id_int)
    # Three chains, numbered 0..2 in entity-then-copy order.
    assert sorted(set(asym_ids.tolist())) == [0, 1, 2]
    # Each chain is one contiguous block, so chain order is atom order.
    starts = np.flatnonzero(np.diff(asym_ids) != 0) + 1
    assert asym_ids[np.concatenate([[0], starts])].tolist() == [0, 1, 2]
    expected_lengths = {0: 4, 1: 4, 2: 4}
    for asym_id, n_res in expected_lengths.items():
        chain = atom_array[asym_ids == asym_id]
        # res_id runs 1..L with no gaps.
        assert sorted(set(chain.res_id.tolist())) == list(range(1, n_res + 1))
        # Only the last residue carries an OXT (leaving atoms are removed).
        oxt_res = set(chain.res_id[chain.atom_name == "OXT"].tolist())
        assert oxt_res == {n_res}
    # The third chain is the second entity, so it holds that sequence's residues.
    third = atom_array[asym_ids == 2]
    assert third.res_name[third.res_id == 1][0] == "GLY"
