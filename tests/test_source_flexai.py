from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from prices import source_flexai
from prices.prices_types import ModelPrice, StartDateConstraint
from prices.update import ProviderYaml

MODELS_RESPONSE: dict[str, Any] = {
    'object': 'list',
    'data': [
        {
            'id': 'test-chat-a',
            'name': 'Test Chat A',
            'context_length': 131072,
            'pricing': {'input_per_mtok': 0.06, 'cached_input_per_mtok': 0.009, 'output_per_mtok': 0.18},
        },
        {
            'id': 'test-chat-b',
            'name': 'Test Chat B',
            'context_length': 262144,
            'pricing': {'input_per_mtok': 0.14, 'cached_input_per_mtok': None, 'output_per_mtok': 0.8},
        },
        {
            'id': 'test-embed',
            'name': 'Test Embed',
            'context_length': 8192,
            'pricing': {'input_per_mtok': 0.01, 'output_per_mtok': 0.0},
        },
        # Billed per image: no per-token price, so it is skipped.
        {'id': 'test-image', 'name': 'Test Image', 'pricing': None, 'media_pricing': {'per_image': 0.003}},
    ],
}


def test_parse_models_extracts_token_priced_models() -> None:
    models = source_flexai.parse_models(MODELS_RESPONSE)

    assert [model.id for model in models] == ['test-chat-a', 'test-chat-b', 'test-embed']
    assert models[0].name == 'Test Chat A'
    assert models[0].context_window == 131072
    assert models[0].prices == ModelPrice(
        input_mtok=Decimal('0.06'), cache_read_mtok=Decimal('0.009'), output_mtok=Decimal('0.18')
    )
    assert models[1].prices == ModelPrice(input_mtok=Decimal('0.14'), output_mtok=Decimal('0.8'))
    assert models[2].prices == ModelPrice(input_mtok=Decimal('0.01'))
    assert all(model.prices_checked == date.today() for model in models)


def test_parse_models_rejects_duplicates() -> None:
    duplicate = {'data': [MODELS_RESPONSE['data'][0], MODELS_RESPONSE['data'][0]]}

    with pytest.raises(RuntimeError, match='Duplicate FlexAI model in pricing data: test-chat-a'):
        source_flexai.parse_models(duplicate)


def test_updater_preserves_existing_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider_path = tmp_path / 'flexai.yml'
    provider_path.write_text(
        """\
id: flexai
name: FlexAI
api_pattern: flexai
models:
  - id: test-chat-a
    name: Curated name
    match: {equals: test-chat-a}
    context_window: 123456
    prices: {input_mtok: 0.05, output_mtok: 0.15}
"""
    )
    monkeypatch.setattr(source_flexai, 'MIN_MODEL_COUNT', 1)
    [model] = source_flexai.parse_models(MODELS_RESPONSE)[:1]

    assert source_flexai.update_flexai_provider(ProviderYaml(provider_path), [model]) == (0, 1)

    updated = ProviderYaml(provider_path).provider.find_model('test-chat-a')
    assert updated is not None
    assert updated.name == 'Curated name'
    assert isinstance(updated.prices, list)
    constraint = updated.prices[-1].constraint
    assert isinstance(constraint, StartDateConstraint)
    assert constraint.start_date == date.today()
    assert updated.prices[-1].prices == model.prices


def test_updater_adds_new_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider_path = tmp_path / 'flexai.yml'
    provider_path.write_text('id: flexai\nname: FlexAI\napi_pattern: flexai\nmodels: []\n')
    monkeypatch.setattr(source_flexai, 'MIN_MODEL_COUNT', 1)
    models = source_flexai.parse_models(MODELS_RESPONSE)

    assert source_flexai.update_flexai_provider(ProviderYaml(provider_path), models) == (3, 0)
    assert ProviderYaml(provider_path).provider.find_model('test-embed') is not None


def test_updater_rejects_suspiciously_small_results(tmp_path: Path) -> None:
    provider_path = tmp_path / 'flexai.yml'
    provider_path.write_text('id: flexai\nname: FlexAI\napi_pattern: flexai\nmodels: []\n')

    with pytest.raises(RuntimeError, match='FlexAI pricing returned only 0 models; expected at least 15'):
        source_flexai.update_flexai_provider(ProviderYaml(provider_path), [])


def test_updater_rejects_sharp_catalog_drop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider_path = tmp_path / 'flexai.yml'
    provider_path.write_text(
        'id: flexai\nname: FlexAI\napi_pattern: flexai\nmodels:\n'
        + ''.join(
            f'  - id: model-{index}\n    name: Model {index}\n    match: {{equals: model-{index}}}\n    prices: {{input_mtok: 1}}\n'
            for index in range(3)
        )
    )
    monkeypatch.setattr(source_flexai, 'MIN_MODEL_COUNT', 1)
    [model] = source_flexai.parse_models(MODELS_RESPONSE)[:1]

    with pytest.raises(RuntimeError, match='FlexAI pricing returned only 1 models; tracked provider has 3'):
        source_flexai.update_flexai_provider(ProviderYaml(provider_path), [model])


def test_main_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('FLEXAI_API_KEY', raising=False)

    with pytest.raises(SystemExit, match='FLEXAI_API_KEY is not set'):
        source_flexai.main()


def test_main_fetches_with_bearer_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider_path = tmp_path / 'flexai.yml'
    provider_path.write_text('id: flexai\nname: FlexAI\napi_pattern: flexai\nmodels: []\n')
    monkeypatch.setenv('FLEXAI_API_KEY', 'test-key')
    monkeypatch.setattr(source_flexai, 'MIN_MODEL_COUNT', 1)
    calls: list[dict[str, Any]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return MODELS_RESPONSE

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({'url': url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(source_flexai.httpx2, 'get', fake_get)

    source_flexai.main(provider_path)

    assert calls[0]['url'] == 'https://api.flex.ai/v1/models'
    assert calls[0]['headers'] == {'Authorization': 'Bearer test-key'}
    assert len(ProviderYaml(provider_path).provider.models) == 3


def test_main_exits_on_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('FLEXAI_API_KEY', 'test-key')

    def failing_get(_url: str, **_kwargs: Any) -> Any:
        raise RuntimeError('boom')

    monkeypatch.setattr(source_flexai.httpx2, 'get', failing_get)

    with pytest.raises(SystemExit, match='Error fetching FlexAI models: boom'):
        source_flexai.main()
