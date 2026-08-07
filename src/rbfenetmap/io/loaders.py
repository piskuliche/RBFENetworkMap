"""Load ligands from SDF, mol2, SMILES, or a directory.

Every loader converges on the same guarantees, because
:class:`~rbfenetmap.core.models.Ligand` demands them: explicit hydrogens, exactly one 3D
conformer, and a filesystem-safe unique name. Enforcing that here rather than downstream
means a malformed input fails while the user still knows which file it came from.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem

from rbfenetmap.core.models import Ligand

__all__ = ("load_ligands", "load_sdf", "load_smiles", "sanitize_name")

logger = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9_.+-]+")

#: Extensions recognised when a directory is given.
_MOLECULE_SUFFIXES = (".sdf", ".mol", ".mol2", ".smi")


def sanitize_name(raw: str, fallback: str) -> str:
    """Return a filesystem-safe ligand name derived from *raw*.

    Ligand names become directory and file names in exports and are parsed back out of
    ``"source~target"`` edge keys, so characters outside ``[A-Za-z0-9_.+-]`` are replaced
    with underscores rather than passed through.
    """
    cleaned = _UNSAFE.sub("_", (raw or "").strip()).strip("_")
    return cleaned or fallback


def _prepare(mol: Chem.Mol, name: str, source: Path, *, embed_if_missing: bool) -> Ligand | None:
    """Ensure a 3D conformer and wrap in a :class:`Ligand`.

    A structure read from a file is taken exactly as written: its atoms are the ligand's
    atoms. Calling ``AddHs`` on it would not be neutral, because RDKit derives the
    missing-hydrogen count from valence, and valence comes from the file's bond orders.
    An Amber mol2 that records a carbonyl as ``C-O`` single rather than ``C=O`` leaves
    the oxygen looking one bond short, so ``AddHs`` materialises a hydroxyl hydrogen that
    exists nowhere in the input -- silently, and on every ligand in the set, which then
    carries into the mapping and every downstream export.

    The trade-off is deliberate: a genuinely hydrogen-less 3D input -- a heavy-atom-only
    PDB, say -- is now taken at face value rather than protonated for you. Prepare
    structures fully before planning against them.
    """
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except (Chem.AtomValenceException, Chem.KekulizeException) as exc:
        logger.warning("Skipping %s from %s: sanitization failed: %s", name, source, exc)
        return None

    has_3d = mol.GetNumConformers() > 0 and mol.GetConformer().Is3D()

    if has_3d:
        # The file's atom list is the ligand. Declaring every atom hydrogen-complete stops
        # RDKit imputing hydrogens from valence, which is what turned a mis-typed C-O
        # single bond into a hydroxyl. Bond orders and explicit atoms are untouched.
        for atom in mol.GetAtoms():
            atom.SetNoImplicit(True)
        mol.UpdatePropertyCache(strict=False)
    else:
        if not embed_if_missing:
            logger.warning("Skipping %s from %s: no 3D conformer.", name, source)
            return None
        # Only an input with no structure gets its hydrogens built here. There is nothing
        # to be faithful to in a SMILES string, and Ligand rejects implicit hydrogens.
        mol = Chem.AddHs(mol)
        # Independently embedded ligands share no frame, so the in-place core RMSD is
        # meaningless across them and every edge will be rejected for geometry. Embedding
        # is a convenience for SMILES input, not a substitute for co-posed structures.
        logger.warning(
            "%s has no 3D conformer; embedding it independently. Ligands embedded this way are not "
            "posed in a common frame -- expect core_geometry_mismatch rejections. Supply co-posed "
            "structures, or raise core_rmsd_threshold.",
            name,
        )
        if AllChem.EmbedMolecule(mol, randomSeed=0xF00D) != 0:
            logger.warning("Skipping %s from %s: embedding failed.", name, source)
            return None
        AllChem.MMFFOptimizeMolecule(mol)

    while mol.GetNumConformers() > 1:
        mol.RemoveConformer(mol.GetConformers()[-1].GetId())

    return Ligand.from_mol(mol, name, source=source)


def load_sdf(path: Path, *, name_property: str = "_Name") -> list[Ligand]:
    """Load every record from an SDF or MOL file.

    Parameters
    ----------
    path : pathlib.Path
    name_property : str, optional
        Molecule property to read the name from. Falls back to ``stem_index``.

    Returns
    -------
    list[Ligand]
    """
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    ligands: list[Ligand] = []
    for index, mol in enumerate(supplier):
        if mol is None:
            logger.warning("Skipping unreadable record %d in %s", index, path)
            continue
        raw = mol.GetProp(name_property) if mol.HasProp(name_property) else ""
        ligand = _prepare(mol, sanitize_name(raw, f"{path.stem}_{index}"), path, embed_if_missing=False)
        if ligand is not None:
            ligands.append(ligand)
    return ligands


def load_smiles(path: Path) -> list[Ligand]:
    """Load a whitespace-delimited ``SMILES name`` file, embedding each molecule.

    See the warning in :func:`_prepare`: independently embedded ligands are not co-posed.
    """
    ligands: list[Ligand] = []
    for index, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        smiles, _, raw = line.partition(" ")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Skipping unparsable SMILES on line %d of %s", index + 1, path)
            continue
        ligand = _prepare(mol, sanitize_name(raw, f"{path.stem}_{index}"), path, embed_if_missing=True)
        if ligand is not None:
            ligands.append(ligand)
    return ligands


def load_mol2(path: Path) -> list[Ligand]:
    """Load a mol2 file.

    Tries stock RDKit first, then ``rdk_amber`` if installed, which handles GAFF/AMBER
    atom types that RDKit's SYBYL-only parser rejects.
    """
    mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
    if mol is None:
        try:
            from rdk_amber.mol2 import load_amber_mol2
        except ImportError:
            logger.warning(
                "Could not read %s with RDKit. If it carries GAFF/AMBER atom types, install the "
                "optional extra: pip install rbfe-network-map[amber-mol2]",
                path,
            )
            return []
        mol = load_amber_mol2(str(path))
    ligand = _prepare(mol, sanitize_name(path.stem, path.stem), path, embed_if_missing=False)
    return [ligand] if ligand is not None else []


def load_ligands(paths: Iterable[Path] | Sequence[str], *, name_property: str = "_Name") -> list[Ligand]:
    """Load ligands from any mixture of files and directories.

    Parameters
    ----------
    paths : Iterable[pathlib.Path] or Sequence[str]
        Files or directories. Directories are scanned (non-recursively) for the
        recognised molecule suffixes.
    name_property : str, optional
        Passed to :func:`load_sdf`.

    Returns
    -------
    list[Ligand]
        In discovery order.

    Raises
    ------
    FileNotFoundError
        If a named path does not exist.
    ValueError
        If two ligands end up with the same name, which would silently collapse two
        network vertices into one.
    """
    resolved: list[Path] = []
    for entry in paths:
        path = Path(entry)
        if not path.exists():
            raise FileNotFoundError(f"No such file or directory: {path}")
        if path.is_dir():
            resolved.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in _MOLECULE_SUFFIXES))
        else:
            resolved.append(path)

    ligands: list[Ligand] = []
    for path in resolved:
        suffix = path.suffix.lower()
        if suffix in (".sdf", ".mol"):
            ligands.extend(load_sdf(path, name_property=name_property))
        elif suffix == ".mol2":
            ligands.extend(load_mol2(path))
        elif suffix == ".smi":
            ligands.extend(load_smiles(path))
        else:
            logger.warning("Ignoring %s: unrecognised suffix %r", path, suffix)

    seen: dict[str, Path | None] = {}
    for ligand in ligands:
        if ligand.name in seen:
            raise ValueError(
                f"Duplicate ligand name {ligand.name!r} (from {seen[ligand.name]} and {ligand.source}). "
                "Names identify network vertices, so they must be unique."
            )
        seen[ligand.name] = ligand.source
    return ligands
