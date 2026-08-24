"""Tests for the configurable sampling temperature (#178/#168).

Temperature is a cross-provider knob: when set it must reach the underlying
chat client; when unset the provider keeps its own default.
"""

import importlib

import pytest

from tradingagents.llm_clients.factory import create_llm_client


@pytest.mark.unit
class TestTemperatureForwarding:
    @pytest.mark.parametrize(
        "provider,model",
        [
            # gpt-4.1 is intentionally a non-reasoning model: the GPT-5 family
            # are reasoning models and correctly drop temperature (see
            # test_openai_reasoning_effort), so forwarding is tested on gpt-4.1.
            ("openai", "gpt-4.1"),
            ("anthropic", "claude-sonnet-5"),
            ("google", "gemini-3.5-flash"),
            ("deepseek", "deepseek-chat"),
        ],
    )
    def test_temperature_reaches_client_when_set(self, provider, model):
        llm = create_llm_client(
            provider=provider, model=model, temperature=0.0, api_key="placeholder"
        ).get_llm()
        assert llm.temperature == 0.0

    def test_temperature_omitted_leaves_provider_default(self):
        # Not passing temperature must not force it to a value.
        llm = create_llm_client(
            provider="openai", model="gpt-4.1", api_key="placeholder"
        ).get_llm()
        # langchain's default is unset/None, not 0.0
        assert llm.temperature is None


@pytest.mark.unit
class TestTemperatureEnvOverlay:
    def test_env_sets_temperature(self, monkeypatch):
        import tradingagents.default_config as dc
        monkeypatch.setenv("TRADINGAGENTS_TEMPERATURE", "0.2")
        importlib.reload(dc)
        # Stored on config (string from env is fine; consumed via float()).
        assert dc.DEFAULT_CONFIG["temperature"] in ("0.2", 0.2)
        assert float(dc.DEFAULT_CONFIG["temperature"]) == 0.2
        monkeypatch.delenv("TRADINGAGENTS_TEMPERATURE", raising=False)
        importlib.reload(dc)

    def test_default_temperature_is_none(self, monkeypatch):
        import tradingagents.default_config as dc
        monkeypatch.delenv("TRADINGAGENTS_TEMPERATURE", raising=False)
        importlib.reload(dc)
        assert dc.DEFAULT_CONFIG["temperature"] is None


@pytest.mark.unit
class TestProviderKwargsTemperature:
    """_get_provider_kwargs float-coerces and forwards temperature, or omits it."""

    def _kwargs_for(self, temperature):
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        # Call the method without constructing the full graph.
        graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
        graph.config = {"llm_provider": "openai", "temperature": temperature}
        return TradingAgentsGraph._get_provider_kwargs(graph)

    def test_float_string_coerced(self):
        assert self._kwargs_for("0.3")["temperature"] == 0.3

    def test_float_passthrough(self):
        assert self._kwargs_for(0.0)["temperature"] == 0.0

    def test_none_omitted(self):
        assert "temperature" not in self._kwargs_for(None)

    def test_empty_string_omitted(self):
        assert "temperature" not in self._kwargs_for("")


@pytest.mark.unit
class TestResearcherTemperature:
    """The bull/bear researchers may run hotter than the rest of the graph.

    At high ``max_debate_rounds`` a low global temperature can collapse the
    investment debate into agreement, but the Portfolio Manager's rating is
    consumed downstream and wants to stay reproducible.
    """

    def _graph_setup(self, researcher_llm=None):
        from tradingagents.graph.conditional_logic import ConditionalLogic
        from tradingagents.graph.setup import GraphSetup

        return GraphSetup(
            "quick-llm",
            "deep-llm",
            {},
            ConditionalLogic(),
            researcher_thinking_llm=researcher_llm,
        )

    def test_researchers_default_to_the_quick_client(self):
        # Callers that never separate the two must be unaffected.
        assert self._graph_setup().researcher_thinking_llm == "quick-llm"

    def test_researchers_use_their_own_client_when_given(self):
        setup = self._graph_setup(researcher_llm="hot-llm")
        assert setup.researcher_thinking_llm == "hot-llm"
        # The Portfolio Manager stays on the deep client either way.
        assert setup.deep_thinking_llm == "deep-llm"

    def test_default_researcher_temperature_is_none(self, monkeypatch):
        import tradingagents.default_config as dc

        monkeypatch.delenv("TRADINGAGENTS_RESEARCHER_TEMPERATURE", raising=False)
        importlib.reload(dc)
        assert dc.DEFAULT_CONFIG["researcher_temperature"] is None

    def test_env_sets_researcher_temperature(self, monkeypatch):
        import tradingagents.default_config as dc

        monkeypatch.setenv("TRADINGAGENTS_RESEARCHER_TEMPERATURE", "0.7")
        importlib.reload(dc)
        assert float(dc.DEFAULT_CONFIG["researcher_temperature"]) == 0.7
        monkeypatch.delenv("TRADINGAGENTS_RESEARCHER_TEMPERATURE", raising=False)
        importlib.reload(dc)
