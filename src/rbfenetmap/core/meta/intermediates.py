"""The intermediate-generator contract: invent a molecule to bridge a gap."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping

from rbfenetmap.core.intermediates import IntermediateOptions, IntermediateProposal
from rbfenetmap.core.models import Ligand
from rbfenetmap.core.options import MappingOptions

__all__ = ("AbstractIntermediateGenerator",)


class AbstractIntermediateGenerator(ABC):
    """Propose molecules that split one hard transformation into two easier ones.

    The fifth plugin kind, and the only one whose output changes the *ligand set* rather
    than the network over it. That is why the contract is narrow: a generator proposes,
    and nothing else. It does not pose its molecules -- :mod:`rbfenetmap.core.posing`
    does, once, for everyone -- it does not decide whether the resulting edges are
    feasible, and it does not price them. Each of those already has an owner, and a
    generator that took any of them over would make an intermediate's quality depend on
    which generator happened to invent it.

    Attributes
    ----------
    name : str
        The registered plugin name, recorded on every ligand the generator's proposals
        become.

    Notes
    -----
    A generator that cannot help with a pair returns an
    :class:`~rbfenetmap.core.intermediates.IntermediateProposal` with no molecules and a
    ``rejection`` string. It does **not** raise: an intermediate is an optimisation, and
    one gap that cannot be bridged is an ordinary outcome rather than an impossible
    request. This is the same rule that keeps a rejected edge out of
    :mod:`rbfenetmap.core.exceptions`.
    """

    name: ClassVar[str] = "abstract"

    @abstractmethod
    def propose(
        self, source: Ligand, target: Ligand, options: IntermediateOptions, mapping_options: MappingOptions
    ) -> IntermediateProposal:
        """Suggest molecules bridging the gap between *source* and *target*.

        Parameters
        ----------
        source, target : Ligand
            The two real ligands, each with explicit hydrogens and one 3D conformer.
        options : IntermediateOptions
            How many molecules may be proposed, and the posing budget that will be spent
            on them.
        mapping_options : MappingOptions
            The same settings the mappers run under, so a generator that needs an MCS
            finds the one the pipeline would have found.

        Returns
        -------
        IntermediateProposal
            Possibly empty, with ``rejection`` set to say why. Every proposed molecule
            must carry no conformer; a
            :class:`~rbfenetmap.core.intermediates.ProposedMolecule` strips any it is
            given, so this is enforced rather than merely asked for.
        """

    def supports_pair(self, source: Ligand, target: Ligand) -> bool:
        """Whether this generator can attempt the pair at all.

        Cheap pre-check, called before :meth:`propose`. The default accepts everything.
        """
        del source, target
        return True

    def describe_parameters(self) -> Mapping[str, Any]:
        """Return the generator's own settings, for the run record.

        Returns
        -------
        Mapping[str, Any]
            JSON-friendly values. The default is empty.

        Notes
        -----
        Generators are the plugin kind most likely to carry knobs of their own -- how far
        to search, which transformations to consider -- and those knobs change what
        molecules a run invents. Reporting them alongside the network is what makes an
        invented ligand reproducible by someone who was not there when it was invented.
        """
        return {}

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r})"
