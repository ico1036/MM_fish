# backend/tests/test_services/test_rwa_personas.py
import pytest
from app.services.rwa_personas import RWA_COMPOSITION, RWA_ARCHETYPES, REASSESS_INTERVALS, generate_rwa_profiles


class TestRWAComposition:
    def test_composition_sums_to_one(self):
        total = sum(RWA_COMPOSITION.values())
        assert total == pytest.approx(1.0)

    def test_all_types_have_archetypes(self):
        for trader_type in RWA_COMPOSITION:
            assert trader_type in RWA_ARCHETYPES

    def test_all_types_have_reassess_interval(self):
        for trader_type in RWA_COMPOSITION:
            assert trader_type in REASSESS_INTERVALS


class TestGenerateRWAProfiles:
    def test_generates_correct_count(self):
        profiles = generate_rwa_profiles(total_agents=50, seed=42)
        assert len(profiles) == 50

    def test_profiles_have_rwa_personas(self):
        profiles = generate_rwa_profiles(total_agents=10, seed=42)
        for p in profiles:
            assert p.trader_type in RWA_COMPOSITION
            assert len(p.persona) > 20

    def test_deterministic(self):
        p1 = generate_rwa_profiles(total_agents=10, seed=42)
        p2 = generate_rwa_profiles(total_agents=10, seed=42)
        assert [p.agent_id for p in p1] == [p.agent_id for p in p2]
