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

from rbfenetmap.core.models import EDGE_SEPARATOR, Ligand

__all__ = ("load_fepplus_network", "load_ligands", "load_orion_network", "load_sdf", "load_smiles", "sanitize_name")

logger = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9_.+-]+")

#: One run of two or more ``>`` characters, which is how both foreign edge formats write
#: their arrow. Matched as a whole run rather than split on so that an FEP+ ``>>>`` line
#: cannot be read by the Orion parser as ``>>`` plus a stray ``>`` on the ligand name.
_ARROW = re.compile(r">{2,}")

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
            "structures, align them with rbfenetmap.core.align.align_ligands (rbfenet --align), or "
            "raise core_rmsd_threshold.",
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


def _parse_edge_file(path: Path, arrow: str, tool: str) -> tuple[str, ...]:
    """Parse a foreign edge list into this package's ``"a~b"`` specifications.

    Parameters
    ----------
    path : pathlib.Path
    arrow : str
        The exact arrow the format writes between two ligand names, ``">>>"`` or ``">>"``.
    tool : str
        Named in error messages, so a user who handed an Orion file to the FEP+ loader is
        told which parser rejected it and why.

    Returns
    -------
    tuple[str, ...]
        In file order, duplicates collapsed. Order is preserved rather than sorted because
        an imported map is a decision someone already made, and the order they wrote it in
        is the only trace of their reasoning that survives the import.

    Raises
    ------
    ValueError
        If a non-comment line does not hold exactly one arrow of the expected length, names
        an empty ligand, or names the same ligand twice; or if the file yields no edges at
        all. Skipping a bad line silently would import a *different* network from the one on
        disk, which is worse than importing none.

    Notes
    -----
    The arrow is matched as a whole run of ``>``. Splitting on the literal would let the
    Orion parser read an FEP+ ``A >>> B`` line as ``A`` to ``> B`` -- a plausible-looking
    edge to a ligand that does not exist -- rather than refusing the file.

    Both formats optionally prefix a name with an internal identifier and a colon
    (``h1a2b3:ligand_7``); only the part after the last colon is a ligand name.

    Names go through :func:`sanitize_name`, because that is what every ligand loader here
    does to the names it reads. An imported edge that kept a character the ligand loader
    would have replaced would name a vertex that does not exist, and the ``explicit``
    planner would refuse the whole network over it.
    """
    specs: list[str] = []
    seen: set[tuple[str, str]] = set()
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        arrows = _ARROW.findall(line)
        if len(arrows) != 1 or arrows[0] != arrow:
            found = arrows[0] if len(arrows) == 1 else f"{len(arrows)} arrows"
            raise ValueError(
                f"{path}:{number}: a {tool} edge line is '<ligand> {arrow} <ligand>', but this line has "
                f"{found}. Offending line: {raw!r}"
            )
        left, right = (field.strip().rpartition(":")[2].strip() for field in line.split(arrow))
        if not left or not right:
            raise ValueError(f"{path}:{number}: a {tool} edge line names an empty ligand. Offending line: {raw!r}")
        source, target = sanitize_name(left, left), sanitize_name(right, right)
        if source == target:
            raise ValueError(f"{path}:{number}: a {tool} edge names {source!r} on both sides.")
        key = (min(source, target), max(source, target))
        if key in seen:
            continue
        seen.add(key)
        specs.append(f"{source}{EDGE_SEPARATOR}{target}")
    if not specs:
        raise ValueError(f"{path} contains no {tool} edges. Check that it is the edge list and not a log file.")
    return tuple(specs)


def load_fepplus_network(path: Path) -> tuple[str, ...]:
    """Read a Schrodinger FEP+ edge list into ``explicit_pairs`` specifications.

    FEP+ writes its planned map as a text file of ``A >>> B`` lines, each name optionally
    prefixed by an internal hash and a colon. openfe ships the equivalent as
    ``load_fepplus_network``; this is the same job in this package's vocabulary, and it
    exists so that a group with an existing map can evaluate this tool against it rather
    than having to start from nothing.

    Parameters
    ----------
    path : pathlib.Path

    Returns
    -------
    tuple[str, ...]
        ``"a~b"`` specifications, ready to hand to
        :class:`~rbfenetmap.core.options.NetworkOptions` as ``explicit_pairs`` alongside
        ``pair_strategy="explicit"`` and the ``explicit`` planner.

    Raises
    ------
    ValueError
        If a line is unparsable or the file holds no edges.

    Notes
    -----
    Only the *topology* is imported; FEP+'s atom mappings are not read, and this package
    maps every imported pair itself. That is the point rather than a limitation -- the main
    reason to import a foreign map is to put it through this package's own feasibility
    rules -- but it does mean an edge FEP+ was happy with can come back rejected. The
    ``explicit`` planner says so loudly instead of dropping it.
    """
    return _parse_edge_file(Path(path), ">>>", "FEP+")


def load_orion_network(path: Path) -> tuple[str, ...]:
    """Read an OpenEye Orion NES edge list into ``explicit_pairs`` specifications.

    Orion writes ``A >> B`` lines, with ``#`` comments. Two arrows rather than three is the
    only structural difference from the FEP+ format, which is exactly why these are two
    entry points over one parser rather than one loader that sniffs: a guess that read an
    FEP+ line as an Orion edge would import a network nobody planned.

    Parameters
    ----------
    path : pathlib.Path

    Returns
    -------
    tuple[str, ...]
        ``"a~b"`` specifications. See :func:`load_fepplus_network` for how to use them and
        for what is deliberately not imported.

    Raises
    ------
    ValueError
        If a line is unparsable or the file holds no edges.
    """
    return _parse_edge_file(Path(path), ">>", "Orion")
