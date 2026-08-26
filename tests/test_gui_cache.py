"""The GUI's mapping cache, its cancellation path, and the progress hook it needs.

The cache exists because mapping is where a plan spends its time and the one stage that
does not depend on the planner. What these tests protect is not the speed but the thing
speed must not cost: a cached run has to produce exactly the network an uncached one does.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from rbfenetmap.core.exceptions import MappingError
from rbfenetmap.core.meta.mappers import AbstractMapper
from rbfenetmap.core.models import AtomMapping
from rbfenetmap.core.options import CorePruningPolicy, MappingOptions, NetworkOptions
from rbfenetmap.core.pipeline import build_network
from rbfenetmap.gui.cache import CachingMapper, MappingCache, RunCancelled
from rbfenetmap.io.networkio import network_to_dict

from .conftest import DummyMapper


class CountingMapper(DummyMapper):
    """A DummyMapper that records how often it was actually asked to do the work."""

    name: ClassVar[str] = "counting"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0

    def map_pair(self, source, target, options):
        self.calls += 1
        return super().map_pair(source, target, options)


class FailingMapper(AbstractMapper):
    """A mapper that always fails, counting its attempts."""

    name: ClassVar[str] = "failing"

    def __init__(self) -> None:
        self.calls = 0

    def map_pair(self, source, target, options):
        self.calls += 1
        raise MappingError(f"no common core between {source.name} and {target.name}")


@pytest.fixture
def pair(benzamides):
    """Two co-posed ligands to map."""
    return benzamides["bza_H"], benzamides["bza_F"]


class TestMemoization:
    def test_a_repeated_pair_is_mapped_once(self, pair):
        """The whole point: the second lookup must not reach the MCS search."""
        source, target = pair
        inner = CountingMapper()
        mapper = CachingMapper(inner)
        options = MappingOptions()

        first = mapper.map_pair(source, target, options)
        second = mapper.map_pair(source, target, options)

        assert inner.calls == 1
        assert first == second
        assert (mapper.cache.hits, mapper.cache.misses) == (1, 1)

    def test_the_reverse_pair_is_a_different_entry(self, pair):
        """``map_pair(a, b)`` and ``map_pair(b, a)`` are not the same question.

        ``cc1`` indexes the source, so sharing one entry between the two directions would
        return a correspondence pointing at the wrong molecule.
        """
        source, target = pair
        inner = CountingMapper()
        mapper = CachingMapper(inner)
        mapper.map_pair(source, target, MappingOptions())
        mapper.map_pair(target, source, MappingOptions())
        assert inner.calls == 2

    def test_changed_mapping_options_miss(self, pair):
        """A cache that ignored the options would answer a question it was not asked."""
        source, target = pair
        inner = CountingMapper()
        mapper = CachingMapper(inner)
        mapper.map_pair(source, target, MappingOptions())
        mapper.map_pair(source, target, MappingOptions(match_selection="first"))
        assert inner.calls == 2

    def test_a_changed_pruning_policy_misses(self, pair):
        """The key recurses into core_pruning rather than reading the top-level fields.

        A nested policy compared only by identity would let a pruning change reuse a
        mapping built under the old one.
        """
        source, target = pair
        inner = CountingMapper()
        mapper = CachingMapper(inner)
        mapper.map_pair(source, target, MappingOptions())
        mapper.map_pair(source, target, MappingOptions(core_pruning=CorePruningPolicy.preset("mcss-e2")))
        assert inner.calls == 2

    def test_a_different_ligand_misses(self, benzamides):
        """Keyed on the atom block, so a different molecule cannot collide with this one."""
        inner = CountingMapper()
        mapper = CachingMapper(inner)
        mapper.map_pair(benzamides["bza_H"], benzamides["bza_F"], MappingOptions())
        mapper.map_pair(benzamides["bza_H"], benzamides["bza_Cl"], MappingOptions())
        assert inner.calls == 2

    def test_the_cache_is_shared_between_mappers(self, pair):
        """A session keeps one cache across runs; two wrappers over it must both see it."""
        source, target = pair
        cache = MappingCache()
        first, second = CountingMapper(), CountingMapper()
        CachingMapper(first, cache).map_pair(source, target, MappingOptions())
        CachingMapper(second, cache).map_pair(source, target, MappingOptions())
        assert (first.calls, second.calls) == (1, 0)


class TestFailuresAreCachedToo:
    def test_a_failing_pair_is_attempted_once(self, pair):
        """The pair that costs a full --mcs-timeout to fail is the one worth remembering.

        Its answer cannot change when a planner knob moves, and re-searching it on every
        knob change would make the unmappable pairs dominate the interactive loop.
        """
        source, target = pair
        inner = FailingMapper()
        mapper = CachingMapper(inner)

        for _ in range(2):
            with pytest.raises(MappingError, match="no common core"):
                mapper.map_pair(source, target, MappingOptions())

        assert inner.calls == 1

    def test_the_remembered_failure_carries_its_message(self, pair):
        """A cached rejection has to explain itself as well as the fresh one did."""
        source, target = pair
        mapper = CachingMapper(FailingMapper())
        with pytest.raises(MappingError) as first:
            mapper.map_pair(source, target, MappingOptions())
        with pytest.raises(MappingError) as second:
            mapper.map_pair(source, target, MappingOptions())
        assert str(first.value) == str(second.value)


class TestDisk:
    def test_a_saved_cache_is_reused_by_the_next_session(self, pair, tmp_path):
        """What makes the second launch of the GUI fast, not merely the second run."""
        source, target = pair
        path = tmp_path / "mappings.json"

        first = CountingMapper()
        warm = CachingMapper(first, MappingCache(path))
        expected = warm.map_pair(source, target, MappingOptions())
        warm.cache.save()

        second = CountingMapper()
        cold = CachingMapper(second, MappingCache(path))
        assert cold.map_pair(source, target, MappingOptions()) == expected
        assert second.calls == 0

    def test_the_round_trip_preserves_the_mapping_exactly(self, pair, tmp_path):
        """Indices, counts and method all survive, or the mapping means something else."""
        source, target = pair
        path = tmp_path / "mappings.json"
        mapper = CachingMapper(CountingMapper(), MappingCache(path))
        original = mapper.map_pair(source, target, MappingOptions())
        mapper.cache.save()

        restored = CachingMapper(CountingMapper(), MappingCache(path)).map_pair(source, target, MappingOptions())
        for field in ("cc1", "cc2", "sc1", "sc2", "n_atoms_1", "n_atoms_2", "method"):
            assert getattr(restored, field) == getattr(original, field), field

    def test_a_cached_failure_survives_the_round_trip(self, pair, tmp_path):
        """Otherwise a restart quietly re-runs every expensive failure."""
        source, target = pair
        path = tmp_path / "mappings.json"
        warm = CachingMapper(FailingMapper(), MappingCache(path))
        with pytest.raises(MappingError):
            warm.map_pair(source, target, MappingOptions())
        warm.cache.save()

        inner = FailingMapper()
        with pytest.raises(MappingError, match="no common core"):
            CachingMapper(inner, MappingCache(path)).map_pair(source, target, MappingOptions())
        assert inner.calls == 0

    def test_a_corrupt_cache_is_ignored_rather_than_fatal(self, tmp_path):
        """Everything in the cache can be recomputed, so it is never worth failing over.

        A tool that refuses to open because a scratch file is damaged is a worse outcome
        than one that is briefly slow.
        """
        path = tmp_path / "mappings.json"
        path.write_text("{ this is not json")
        assert len(MappingCache(path)) == 0

    def test_a_cache_from_another_version_is_discarded(self, tmp_path):
        """There is nothing to migrate, so an old shape is dropped, not read."""
        path = tmp_path / "mappings.json"
        path.write_text(json.dumps({"version": 0, "entries": {"k": {"error": "x"}}}))
        assert len(MappingCache(path)) == 0

    def test_saving_leaves_no_temporary_behind(self, pair, tmp_path):
        """The write is a rename, so an interrupted save cannot corrupt the cache."""
        source, target = pair
        path = tmp_path / "mappings.json"
        mapper = CachingMapper(CountingMapper(), MappingCache(path))
        mapper.map_pair(source, target, MappingOptions())
        mapper.cache.save()
        assert [p.name for p in tmp_path.iterdir()] == ["mappings.json"]

    def test_an_in_memory_cache_saves_nothing(self, tmp_path):
        """No path means no file, not a file in the working directory."""
        MappingCache().save()
        assert list(tmp_path.iterdir()) == []


class TestIdentity:
    def test_the_wrapper_reports_the_wrapped_mapper_name(self, pair):
        """Recorded on every mapping as its ``method`` and serialized into the network.

        A network claiming it was mapped by "caching" would name something that is not a
        mapping algorithm, and would not round-trip to the run that produced it.
        """
        source, target = pair
        mapper = CachingMapper(CountingMapper())
        assert mapper.name == "counting"
        assert mapper.map_pair(source, target, MappingOptions()).method == "counting"

    def test_supports_pair_is_delegated(self, pair):
        """The cheap pre-check is the wrapped mapper's opinion, not the wrapper's."""

        class Picky(CountingMapper):
            def supports_pair(self, source, target):
                return False

        source, target = pair
        assert CachingMapper(Picky()).supports_pair(source, target) is False


class TestAgainstThePipeline:
    def test_a_cached_run_produces_the_same_network(self, benzamides):
        """The assertion the whole module has to earn.

        Compared through ``network_to_dict``, which is this package's own notion of what
        makes two networks the same -- it is what the JSON round-trip and the checked-in
        golden baselines are built on. A cache that changed a single selected edge would
        be worse than no cache.
        """
        options = NetworkOptions()
        plain = build_network(benzamides, mapper="mcss-e2", network_options=options)

        # Cold cache, then the same cache warm: both must match the uncached network.
        cache = MappingCache()
        cold = build_network(benzamides, mapper=CachingMapper(_mcss(), cache), network_options=options)
        warm = build_network(benzamides, mapper=CachingMapper(_mcss(), cache), network_options=options)

        assert network_to_dict(cold) == network_to_dict(plain)
        assert network_to_dict(warm) == network_to_dict(plain)

    def test_the_second_run_maps_nothing(self, benzamides):
        """A knob moved between two runs must not re-run the MCS searches."""
        cache = MappingCache()
        inner = _mcss()
        build_network(benzamides, mapper=CachingMapper(inner, cache), network_options=NetworkOptions())
        before = cache.hits

        # A pure selection knob: same pairs, same mappings, different network.
        build_network(
            benzamides, mapper=CachingMapper(inner, cache), network_options=NetworkOptions(edges_per_ligand=3)
        )
        assert cache.hits > before
        assert cache.misses == len(cache)


class TestCancellation:
    def test_a_cancelled_run_raises_out_of_build_network(self, benzamides):
        """Cancellation has to escape the pipeline, not be absorbed as a rejection.

        build_candidate catches MappingError and turns it into a mapper_failed rejection,
        so a cancellation spelled that way would produce a plausible-looking network in
        which every unreached pair appears infeasible.
        """
        mapper = CachingMapper(_mcss(), should_cancel=lambda: True)
        with pytest.raises(RunCancelled):
            build_network(benzamides, mapper=mapper, network_options=NetworkOptions())

    def test_a_cancellation_is_not_a_mapping_error(self):
        """Stated directly, because the whole design rests on it."""
        assert not issubclass(RunCancelled, MappingError)

    def test_an_uncancelled_run_is_untouched(self, benzamides):
        """The flag is polled, not merely present."""
        mapper = CachingMapper(_mcss(), should_cancel=lambda: False)
        network = build_network(benzamides, mapper=mapper, network_options=NetworkOptions())
        assert network.edges


class TestProgressCallback:
    def test_the_increments_sum_to_the_pair_count(self, benzamides):
        """A caller divides by the pair count, so the increments have to reach it."""
        seen: list[int] = []
        build_network(benzamides, mapper="mcss-e2", network_options=NetworkOptions(), progress_callback=seen.append)
        n = len(benzamides)
        assert sum(seen) == n * (n - 1) // 2

    def test_adaptive_evaluation_reports_at_most_the_pair_count(self, benzamides):
        """The adaptive loop stops when the targets are met, so it is a ceiling.

        Documented on build_network rather than left for a caller to discover by watching
        a progress bar stall at 60%.
        """
        seen: list[int] = []
        build_network(
            benzamides,
            mapper="mcss-e2",
            network_options=NetworkOptions(pair_evaluation="adaptive"),
            progress_callback=seen.append,
        )
        n = len(benzamides)
        assert 0 < sum(seen) <= n * (n - 1) // 2

    def test_omitting_the_callback_changes_nothing(self, benzamides):
        """The parameter is additive; the default path must be exactly as it was."""
        options = NetworkOptions()
        with_callback = build_network(benzamides, network_options=options, progress_callback=lambda _: None)
        without = build_network(benzamides, network_options=options)
        assert network_to_dict(with_callback) == network_to_dict(without)


def _mcss():
    """A real mapper instance, so the pipeline tests exercise genuine MCS work."""
    from rbfenetmap.plugins.mappers import create_mapper

    return create_mapper("mcss-e2")


def test_every_atom_mapping_field_is_stored(pair):
    """Guards the encode/decode pair against a field being added to AtomMapping.

    A new field would otherwise be silently dropped on the way to disk and silently
    defaulted on the way back, which is the kind of loss that shows up as a wrong
    soft-core months later.
    """
    source, target = pair
    mapper = CachingMapper(CountingMapper())
    mapping = mapper.map_pair(source, target, MappingOptions())
    assert isinstance(mapping, AtomMapping)

    (entry,) = mapper.cache._entries.values()  # noqa: SLF001 - the stored shape is the point
    assert set(entry) == set(AtomMapping.__dataclass_fields__)
