from __future__ import annotations

import os
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx2

from prices.prices_types import ClauseEquals, ModelInfo, ModelPrice
from prices.update import ProviderYaml
from prices.utils import package_dir

MODELS_URL = 'https://api.flex.ai/v1/models'
API_KEY_ENV = 'FLEXAI_API_KEY'
MIN_MODEL_COUNT = 15


def parse_models(data: dict[str, Any]) -> list[ModelInfo]:
    """Extract token-priced models from a FlexAI `GET /v1/models` response.

    FlexAI publishes per-million-token prices in each model's `pricing` block. Models
    billed per image or per minute carry no `input_per_mtok` and are skipped.
    """
    models: list[ModelInfo] = []
    seen_ids: set[str] = set()
    entries: list[dict[str, Any]] = data.get('data', [])
    for model in entries:
        model_id = model['id']
        pricing: dict[str, Any] = model.get('pricing') or {}
        if pricing.get('input_per_mtok') is None:
            continue
        if model_id in seen_ids:
            raise RuntimeError(f'Duplicate FlexAI model in pricing data: {model_id}')
        seen_ids.add(model_id)
        models.append(
            ModelInfo(
                id=model_id,
                name=model.get('name') or model_id,
                match=ClauseEquals(equals=model_id),
                prices=ModelPrice(
                    input_mtok=_price(pricing.get('input_per_mtok')),
                    cache_read_mtok=_price(pricing.get('cached_input_per_mtok')),
                    output_mtok=_price(pricing.get('output_per_mtok')),
                ),
                context_window=model.get('context_length'),
                prices_checked=date.today(),
            )
        )
    return sorted(models, key=lambda m: m.id)


def update_flexai_provider(provider_yaml: ProviderYaml, models: list[ModelInfo]) -> tuple[int, int]:
    if len(models) < MIN_MODEL_COUNT:
        raise RuntimeError(f'FlexAI pricing returned only {len(models)} models; expected at least {MIN_MODEL_COUNT}')
    existing_count = len(provider_yaml.provider.models)
    if len(models) * 2 < existing_count:
        raise RuntimeError(f'FlexAI pricing returned only {len(models)} models; tracked provider has {existing_count}')

    models_added = 0
    models_updated = 0
    for model in models:
        matching_model = provider_yaml.provider.find_model(model.id)
        if matching_model is None:
            models_added += provider_yaml.add_model(model)
        else:
            provider_yaml.update_model(matching_model.id, model, set_prices=True, preserve_price_history=True)
            models_updated += 1
    provider_yaml.save()
    return models_added, models_updated


def main(provider_path: Path | None = None) -> None:
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        # The catalog endpoint is authenticated. A key with a zero spend budget can read it.
        raise SystemExit(f'{API_KEY_ENV} is not set; FlexAI model pricing requires an API key to read')
    try:
        response = httpx2.get(MODELS_URL, headers={'Authorization': f'Bearer {api_key}'}, timeout=30.0)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        # Exit non-zero so a scheduled or scripted refresh can't report success on stale data.
        raise SystemExit(f'Error fetching FlexAI models: {e}') from e
    models = parse_models(data)
    provider_yaml = ProviderYaml(provider_path or package_dir / 'providers/flexai.yml')
    models_added, models_updated = update_flexai_provider(provider_yaml, models)
    print(f'FlexAI prices updated: {models_added} added, {models_updated} updated')


def get_flexai_prices() -> None:  # pragma: no cover - thin CLI alias for main
    """Download and update FlexAI model prices (requires FLEXAI_API_KEY)."""
    main()


def _price(value: Any) -> Decimal | None:
    if value is None:
        return None
    price = Decimal(str(value))
    return price if price > 0 else None


if __name__ == '__main__':
    main()
