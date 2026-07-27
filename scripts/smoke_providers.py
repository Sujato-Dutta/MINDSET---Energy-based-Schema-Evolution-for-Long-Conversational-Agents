from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from mindset.api.backend import APIBackend
from mindset.api.models import CompletionRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/models.yaml')
    parser.add_argument(
        '--models',
        nargs='+',
        default=['gpt_oss_120b', 'glm_47', 'gemma_4_31b'],
    )
    args = parser.parse_args()
    backend = APIBackend.from_file(args.config)
    results = []
    for model_name in args.models:
        result = backend.complete(
            CompletionRequest(
                model_name=model_name,
                system_prompt='You are a connectivity test. Follow the user instruction exactly.',
                user_prompt='Reply with exactly: OK',
                dataset_example_id='provider-smoke-v1',
                method='provider_smoke',
                code_version='provider-smoke-v1',
                temperature=0.0,
                max_output_tokens=256,
            ),
            max_retries=3,
        )
        results.append(
            {
                'model_name': model_name,
                'provider': result.provider,
                'model_id': result.model_id,
                'key_alias': result.key_alias,
                'request_id': result.request_id,
                'input_tokens': result.usage.input_tokens,
                'output_tokens': result.usage.output_tokens,
                'latency_seconds': result.latency_seconds,
                'retry_count': result.retry_count,
                'cache_hit': result.cache_hit,
                'response_ok': result.text.strip().upper() == 'OK',
            }
        )
    timestamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = ROOT / 'outputs' / 'logs' / f'provider_smoke_{timestamp}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

