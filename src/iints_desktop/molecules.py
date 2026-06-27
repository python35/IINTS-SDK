from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import shlex


@dataclass(frozen=True)
class MoleculeAsset:
    """One structural-biology asset shown in the desktop deep-dive tab."""

    key: str
    title: str
    uniprot_id: str
    image_path: Path
    structure_path: Path
    explanation: str
    sdk_link: str
    pae_target: str | None
    pae_note: str


@dataclass(frozen=True)
class BackboneAtom:
    """One C-alpha atom used for the lightweight interactive backbone view."""

    chain_id: str
    residue_index: int
    residue_name: str
    x: float
    y: float
    z: float
    confidence: float | None


@dataclass(frozen=True)
class MoleculeBackbone:
    """A compact, deterministic representation of a protein backbone."""

    atoms: tuple[BackboneAtom, ...]
    center: tuple[float, float, float]
    radius: float

    @property
    def chain_count(self) -> int:
        return len({atom.chain_id for atom in self.atoms})


class MoleculeStructureError(ValueError):
    """Raised when a bundled mmCIF structure cannot be rendered safely."""


def pae_html_path(target: str, output_dir: Path = Path("results") / "structural") -> Path:
    """Return the default interactive PAE HTML path for a structural target."""

    return output_dir / f"{target}_pae.html"


def load_molecule_backbone(path: Path) -> MoleculeBackbone:
    """Read C-alpha coordinates from a compact AlphaFold mmCIF structure.

    The desktop app deliberately renders only the C-alpha backbone.  It is fast,
    works without a web view or GPU-only dependency, and is sufficient for an
    interactive explanation of a protein chain.  It is not used by the
    physiological simulator or any decision logic.
    """

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MoleculeStructureError(f"Could not read structure: {path}") from exc

    if not lines or not lines[0].startswith("data_"):
        raise MoleculeStructureError(
            f"{path.name} is not a valid AlphaFold mmCIF structure."
        )

    headers: list[str] = []
    records: list[list[str]] = []
    start_index: int | None = None
    for index, line in enumerate(lines):
        if line.startswith("_atom_site.group_PDB"):
            start_index = index
            break
    if start_index is None:
        raise MoleculeStructureError(f"{path.name} has no atom coordinates.")

    index = start_index
    while index < len(lines) and lines[index].startswith("_atom_site."):
        headers.append(lines[index].strip().removeprefix("_atom_site."))
        index += 1

    for line in lines[index:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped == "loop_":
            break
        if stripped.startswith("_"):
            break
        fields = shlex.split(stripped)
        if len(fields) == len(headers):
            records.append(fields)

    if not headers or not records:
        raise MoleculeStructureError(f"{path.name} contains no readable atom records.")

    header_index = {name: offset for offset, name in enumerate(headers)}

    def value(record: list[str], *names: str) -> str | None:
        for name in names:
            offset = header_index.get(name)
            if offset is not None:
                item = record[offset]
                if item not in {"?", "."}:
                    return item
        return None

    atoms: list[BackboneAtom] = []
    for record in records:
        if value(record, "label_atom_id", "auth_atom_id") != "CA":
            continue
        try:
            sequence = int(value(record, "label_seq_id", "auth_seq_id") or "0")
            atom = BackboneAtom(
                chain_id=value(record, "label_asym_id", "auth_asym_id") or "?",
                residue_index=sequence,
                residue_name=value(record, "label_comp_id", "auth_comp_id") or "UNK",
                x=float(value(record, "Cartn_x") or "nan"),
                y=float(value(record, "Cartn_y") or "nan"),
                z=float(value(record, "Cartn_z") or "nan"),
                confidence=_optional_float(value(record, "B_iso_or_equiv")),
            )
        except ValueError:
            continue
        if all(coordinate == coordinate for coordinate in (atom.x, atom.y, atom.z)):
            atoms.append(atom)

    if len(atoms) < 2:
        raise MoleculeStructureError(
            f"{path.name} has too few C-alpha atoms for an interactive chain view."
        )

    atoms.sort(key=lambda atom: (atom.chain_id, atom.residue_index))
    center = (
        sum(atom.x for atom in atoms) / len(atoms),
        sum(atom.y for atom in atoms) / len(atoms),
        sum(atom.z for atom in atoms) / len(atoms),
    )
    radius = max(
        ((atom.x - center[0]) ** 2 + (atom.y - center[1]) ** 2 + (atom.z - center[2]) ** 2) ** 0.5
        for atom in atoms
    )
    return MoleculeBackbone(atoms=tuple(atoms), center=center, radius=max(radius, 1.0))


def _optional_float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def list_molecule_assets() -> list[MoleculeAsset]:
    """Return bundled AlphaFold assets with SDK-facing interpretation text."""

    base = Path(str(resources.files("iints_desktop") / "assets" / "alphafold"))
    return [
        MoleculeAsset(
            key="insulin",
            title="Insulin",
            uniprot_id="P01308",
            image_path=base / "insulin_3D.png",
            structure_path=base / "AF-P01308-F1-model_v4.cif",
            explanation=(
                "Insulin does not act instantly after a pump delivers it. The SDK's IOB and "
                "subcutaneous PK/PD delays are a mathematical abstraction of slow absorption, "
                "diffusion, and receptor-level biology."
            ),
            sdk_link="Connects to: insulin-on-board, subcutaneous absorption, Hovorka/Bergman insulin action.",
            pae_target="insulin-mutation",
            pae_note=(
                "PAE heatmap for UniProt P01308. Low PAE values indicate more confident relative "
                "placement between residue positions in the AlphaFold structure."
            ),
        ),
        MoleculeAsset(
            key="glucagon",
            title="Glucagon",
            uniprot_id="P01275",
            image_path=base / "glucagon_3D.png",
            structure_path=base / "AF-P01275-F1-model_v4.cif",
            explanation=(
                "Glucagon is the counter-regulatory rescue hormone. In the SDK it links to "
                "hypoglycemia defense, hepatic glucose production, and dual-hormone research demos."
            ),
            sdk_link="Connects to: hypo defense layer, HAAF experiments, glucagon rescue dynamics.",
            pae_target="glucagon",
            pae_note=(
                "PAE heatmap for UniProt P01275. The matrix is structural prediction evidence only; "
                "it is not a glucose, dosing, or treatment metric."
            ),
        ),
    ]
