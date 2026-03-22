"""Tests for persona generator."""

import pytest

from app.models.agent_profile import TraderProfile, AGENT_COMPOSITION, ARCHETYPE_PERSONAS
from app.services.persona_generator import generate_profiles_deterministic, _parse_persona_response


class TestGenerateProfilesDeterministic:
    def test_generates_correct_count(self):
        profiles = generate_profiles_deterministic(total_agents=50)
        assert len(profiles) == 50

    def test_generates_small_count(self):
        profiles = generate_profiles_deterministic(total_agents=10)
        assert len(profiles) == 10

    def test_all_profiles_are_trader_profiles(self):
        profiles = generate_profiles_deterministic(total_agents=20)
        assert all(isinstance(p, TraderProfile) for p in profiles)

    def test_unique_agent_ids(self):
        profiles = generate_profiles_deterministic(total_agents=50)
        ids = [p.agent_id for p in profiles]
        assert len(ids) == len(set(ids))

    def test_all_trader_types_represented(self):
        profiles = generate_profiles_deterministic(total_agents=50)
        types = {p.trader_type for p in profiles}
        assert "informed" in types
        assert "noise" in types
        assert "momentum" in types
        assert "hft" in types

    def test_risk_appetite_bounded(self):
        profiles = generate_profiles_deterministic(total_agents=50)
        assert all(0 <= p.risk_appetite <= 1 for p in profiles)

    def test_capital_positive(self):
        profiles = generate_profiles_deterministic(total_agents=50)
        assert all(p.capital > 0 for p in profiles)

    def test_deterministic_with_same_seed(self):
        p1 = generate_profiles_deterministic(total_agents=20, seed=123)
        p2 = generate_profiles_deterministic(total_agents=20, seed=123)
        for a, b in zip(p1, p2):
            assert a.agent_id == b.agent_id
            assert a.risk_appetite == b.risk_appetite

    def test_different_seeds_different_profiles(self):
        p1 = generate_profiles_deterministic(total_agents=20, seed=1)
        p2 = generate_profiles_deterministic(total_agents=20, seed=2)
        risks1 = [p.risk_appetite for p in p1]
        risks2 = [p.risk_appetite for p in p2]
        assert risks1 != risks2

    def test_custom_composition(self):
        comp = {"informed": 0.5, "noise": 0.5}
        profiles = generate_profiles_deterministic(total_agents=20, composition=comp)
        types = [p.trader_type for p in profiles]
        assert set(types) == {"informed", "noise"}

    def test_persona_is_non_empty(self):
        profiles = generate_profiles_deterministic(total_agents=10)
        assert all(len(p.persona) > 10 for p in profiles)

    def test_hft_is_scalper(self):
        profiles = generate_profiles_deterministic(total_agents=50)
        hft_profiles = [p for p in profiles if p.trader_type == "hft"]
        assert all(p.time_horizon == "scalper" for p in hft_profiles)

    def test_institutional_is_swing(self):
        profiles = generate_profiles_deterministic(total_agents=50)
        inst_profiles = [p for p in profiles if p.trader_type == "institutional"]
        assert all(p.time_horizon == "swing" for p in inst_profiles)


class TestTraderProfile:
    def test_short_description(self):
        profile = TraderProfile(
            agent_id="test_001",
            trader_type="momentum",
            persona="A trader",
            risk_appetite=0.7,
        )
        desc = profile.short_description
        assert "test_001" in desc
        assert "momentum" in desc

    def test_default_values(self):
        profile = TraderProfile(
            agent_id="test",
            trader_type="noise",
            persona="A noise trader",
        )
        assert profile.risk_appetite == 0.5
        assert profile.capital == 10000.0
        assert profile.time_horizon == "intraday"


class TestParsePersonaResponse:
    def test_parse_json_array(self):
        text = '["Persona 1", "Persona 2", "Persona 3"]'
        result = _parse_persona_response(text, 3)
        assert len(result) == 3
        assert result[0] == "Persona 1"

    def test_parse_code_block(self):
        text = '```json\n["P1", "P2"]\n```'
        result = _parse_persona_response(text, 2)
        assert len(result) == 2

    def test_parse_embedded_array(self):
        text = 'Here are the personas: ["P1", "P2"] done.'
        result = _parse_persona_response(text, 2)
        assert len(result) == 2

    def test_fallback_on_bad_input(self):
        result = _parse_persona_response("Not valid", 3)
        assert len(result) == 3  # Returns defaults


class TestArchetypePersonas:
    def test_all_types_have_personas(self):
        for trader_type in AGENT_COMPOSITION:
            assert trader_type in ARCHETYPE_PERSONAS
            assert len(ARCHETYPE_PERSONAS[trader_type]) > 20
