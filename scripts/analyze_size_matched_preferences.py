#!/usr/bin/env python3
"""Separate size from attachment chemistry on held-out selections.

The marginal within-parent contrast mixes fragment size with attachment
element and hybridization. Fluorine, chlorine, bromine, and iodine pendants
are single heavy atoms, so a preference for nitrogen over fluorine can be a
preference for the larger fragment.

This script keeps only sibling actions with the same heavy-atom count and
counts which attachment is selected. It reads archived test selections and
does not refit models. Fragment descriptors use the same induced-fragment
construction as schema v2.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_v2" / "feature_analysis" / "size_matched_preferences.json"

HYBRID = {
    Chem.HybridizationType.SP: "SP",
    Chem.HybridizationType.SP2: "SP2",
    Chem.HybridizationType.SP3: "SP3",
}


def induced_fragment(parent: Chem.Mol, atom_indices: list[int]) -> Chem.Mol:
    smiles = Chem.MolFragmentToSmiles(
        parent,
        atomsToUse=list(atom_indices),
        canonical=True,
        isomericSmiles=False,
    )
    fragment = Chem.MolFromSmiles(smiles)
    if fragment is None:
        raise ValueError("failed to construct fragment")
    return fragment


def load_selections() -> dict[str, dict[str, str]]:
    with (ROOT / "results_v2" / "test" / "selections.json").open(encoding="utf-8") as handle:
        primary = json.load(handle)
    with (ROOT / "results_v2" / "ablation" / "shallow_mlp" / "test_selections.json").open(
        encoding="utf-8"
    ) as handle:
        mlp = json.load(handle)
    with (ROOT / "results_v2" / "ablation" / "lightgbm_regressor" / "test_selections.json").open(
        encoding="utf-8"
    ) as handle:
        lightgbm = json.load(handle)
    return {
        "gatv2": {
            row["parent_id"]: row["action_id"]
            for row in primary
            if row["policy"] == "learned"
        },
        "shallow_mlp": {row["parent_id"]: row["action_id"] for row in mlp},
        "lightgbm": {row["parent_id"]: row["action_id"] for row in lightgbm},
    }


def load_parents(selections: dict[str, dict[str, str]]) -> list[dict]:
    parents = []
    with (ROOT / "manifests_v2" / "test.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            mol = Chem.MolFromSmiles(raw["smiles"])
            if mol is None:
                raise ValueError(f"invalid SMILES for {raw['parent_id']}")
            actions = []
            for action in raw["actions"]:
                atom = mol.GetAtomWithIdx(action["fragment_attachment_atom"])
                fragment = induced_fragment(mol, action["fragment_atom_indices"])
                actions.append(
                    {
                        "id": action["action_id"],
                        "heavy_atoms": int(fragment.GetNumHeavyAtoms()),
                        "tpsa": float(Descriptors.TPSA(fragment)),
                        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(fragment)),
                        "element": atom.GetSymbol(),
                        "hybridization": HYBRID.get(atom.GetHybridization(), "other"),
                    }
                )
            parents.append(
                {
                    "actions": actions,
                    "chosen": {
                        model: chosen[raw["parent_id"]] for model, chosen in selections.items()
                    },
                }
            )
    return parents


def pair_counts(parents: list[dict], model: str, field: str) -> dict[str, int]:
    """Count equal-heavy-atom sibling pairs in which one side was selected."""
    counts: Counter[str] = Counter()
    for parent in parents:
        selected = next(
            action for action in parent["actions"] if action["id"] == parent["chosen"][model]
        )
        for other in parent["actions"]:
            if other["id"] == selected["id"] or other["heavy_atoms"] != selected["heavy_atoms"]:
                continue
            if other[field] == selected[field]:
                continue
            counts[f"{selected[field]}>{other[field]}"] += 1
    return dict(counts)


def matched_physchem(parents: list[dict], model: str) -> dict[str, float]:
    tpsa_deltas = []
    rot_deltas = []
    for parent in parents:
        selected = next(
            action for action in parent["actions"] if action["id"] == parent["chosen"][model]
        )
        siblings = [
            action
            for action in parent["actions"]
            if action["id"] != selected["id"] and action["heavy_atoms"] == selected["heavy_atoms"]
        ]
        if not siblings:
            continue
        tpsa_deltas.append(
            selected["tpsa"] - sum(action["tpsa"] for action in siblings) / len(siblings)
        )
        rot_deltas.append(
            selected["rotatable_bonds"]
            - sum(action["rotatable_bonds"] for action in siblings) / len(siblings)
        )
    tpsa_deltas.sort()
    rot_deltas.sort()
    mid = len(tpsa_deltas) // 2
    return {
        "n_parents": len(tpsa_deltas),
        "median_tpsa_difference": tpsa_deltas[mid],
        "fraction_higher_tpsa": sum(value > 0 for value in tpsa_deltas) / len(tpsa_deltas),
        "median_rotatable_bond_difference": rot_deltas[mid],
        "fraction_more_rotatable_bonds": sum(value > 0 for value in rot_deltas) / len(rot_deltas),
    }


def element_size_gap(parents: list[dict]) -> dict[str, dict[str, float]]:
    summary = {}
    for element in ("F", "Cl", "Br", "I", "N", "O"):
        gaps = []
        single = 0
        for parent in parents:
            hits = [action for action in parent["actions"] if action["element"] == element]
            for hit in hits:
                single += int(hit["heavy_atoms"] == 1)
                siblings = [action for action in parent["actions"] if action["id"] != hit["id"]]
                gaps.append(
                    hit["heavy_atoms"] - sum(action["heavy_atoms"] for action in siblings) / len(siblings)
                )
        summary[element] = {
            "n_actions": len(gaps),
            "fraction_one_heavy_atom": single / len(gaps),
            "mean_heavy_atoms_minus_siblings": sum(gaps) / len(gaps),
        }
    return summary


def main() -> None:
    parents = load_parents(load_selections())
    report = {
        "split": "test",
        "n_parents": len(parents),
        "definition": (
            "Equal-heavy-atom sibling comparisons: the selected action versus each "
            "unselected sibling with the same fragment heavy-atom count."
        ),
        "element_size_relative_to_siblings": element_size_gap(parents),
        "models": {},
    }
    for model in ("gatv2", "shallow_mlp", "lightgbm"):
        report["models"][model] = {
            "equal_heavy_atom_element": pair_counts(parents, model, "element"),
            "equal_heavy_atom_hybridization": pair_counts(parents, model, "hybridization"),
            "equal_heavy_atom_physchem": matched_physchem(parents, model),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
