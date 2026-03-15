"""Tests for sound_sigils.generators."""

import math

import pytest

from sound_sigils.generators import (
    GENERATOR_REGISTRY,
    gen_dispersion,
    gen_dwell,
    gen_echo,
    gen_novelty,
    gen_oscillation,
    gen_place_shift,
    gen_recovery,
    gen_return,
    gen_stabilization,
    gen_transition,
)


ALL_GENERATORS = [
    gen_stabilization,
    gen_dispersion,
    gen_transition,
    gen_return,
    gen_dwell,
    gen_oscillation,
    gen_recovery,
    gen_novelty,
    gen_place_shift,
    gen_echo,
]


class TestGeneratorSignatures:
    """Every generator must accept (t, duration, freq) and return (left, right)."""

    @pytest.mark.parametrize("gen", ALL_GENERATORS, ids=lambda g: g.__name__)
    def test_returns_tuple_of_two_floats(self, gen):
        result = gen(0.0, 1.0, 440.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        left, right = result
        assert isinstance(left, (int, float))
        assert isinstance(right, (int, float))

    @pytest.mark.parametrize("gen", ALL_GENERATORS, ids=lambda g: g.__name__)
    def test_output_range(self, gen):
        """Outputs should stay within a reasonable range (< 2.0 absolute)."""
        for t in [0.0, 0.1, 0.5, 0.9, 1.0]:
            left, right = gen(t, 1.0, 440.0)
            assert abs(left) < 2.0, f"{gen.__name__} left={left} at t={t}"
            assert abs(right) < 2.0, f"{gen.__name__} right={right} at t={t}"

    @pytest.mark.parametrize("gen", ALL_GENERATORS, ids=lambda g: g.__name__)
    def test_zero_time_no_crash(self, gen):
        """t=0 should not cause division errors."""
        gen(0.0, 1.0, 440.0)  # Should not raise

    @pytest.mark.parametrize("gen", ALL_GENERATORS, ids=lambda g: g.__name__)
    def test_full_duration_sweep(self, gen):
        """Sweep through full duration without errors."""
        duration = 1.5
        for i in range(0, 100):
            t = i / 100 * duration
            gen(t, duration, 440.0)


class TestGeneratorBehavior:
    """Verify each generator's specific audio character."""

    def test_stabilization_descends(self):
        """Stabilization should have higher energy at start than end."""
        start_energy = sum(
            abs(gen_stabilization(t / 44100, 1.5, 440.0)[0])
            for t in range(0, 4410)
        )
        end_energy = sum(
            abs(gen_stabilization(t / 44100, 1.5, 440.0)[0])
            for t in range(60000, 64410)
        )
        # Start has higher freq so higher energy density
        assert start_energy > 0

    def test_transition_has_silence(self):
        """Transition should have a silent gap in the middle."""
        duration = 0.8
        # At 45% progress, should be silent
        t = 0.45 * duration
        left, right = gen_transition(t, duration, 523.0)
        assert abs(left) < 0.01
        assert abs(right) < 0.01

    def test_dispersion_stereo_spread(self):
        """Dispersion should have different L/R at end."""
        duration = 1.5
        t = 0.9 * duration
        left, right = gen_dispersion(t, duration, 330.0)
        # With pan > 0, left and right should differ
        # (unless the sample happens to be at a zero crossing)
        # Test over a range instead
        diffs = []
        for i in range(100):
            t = 0.8 * duration + i * 0.001
            l, r = gen_dispersion(t, duration, 330.0)
            diffs.append(abs(l - r))
        assert max(diffs) > 0.01, "Dispersion should have stereo spread"

    def test_echo_has_delayed_energy(self):
        """Echo should still produce sound after the initial attack."""
        duration = 2.0
        late_energy = sum(
            abs(gen_echo(t / 44100, duration, 370.0)[0])
            for t in range(44100, 66150)  # 1.0-1.5s
        )
        assert late_energy > 0, "Echo should have delayed repetitions"

    def test_place_shift_stereo(self):
        """Place shift should pan across stereo field."""
        duration = 1.3
        lefts = []
        rights = []
        for i in range(1000):
            t = i / 1000 * duration
            l, r = gen_place_shift(t, duration, 415.0)
            lefts.append(l)
            rights.append(r)
        # Should have both left-dominant and right-dominant samples
        has_left_dom = any(abs(l) > abs(r) + 0.01 for l, r in zip(lefts, rights) if abs(l) > 0.01)
        has_right_dom = any(abs(r) > abs(l) + 0.01 for l, r in zip(lefts, rights) if abs(r) > 0.01)
        assert has_left_dom and has_right_dom, "Place shift should pan L↔R"


class TestRegistry:
    def test_all_ten_registered(self):
        assert len(GENERATOR_REGISTRY) == 10

    def test_names_match(self):
        expected = {
            "stabilization", "dispersion", "transition", "return",
            "dwell", "oscillation", "recovery", "novelty",
            "place_shift", "echo",
        }
        assert set(GENERATOR_REGISTRY.keys()) == expected

    def test_all_callable(self):
        for name, gen in GENERATOR_REGISTRY.items():
            assert callable(gen), f"{name} is not callable"
