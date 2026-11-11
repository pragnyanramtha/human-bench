"""API trials resume only current successful records; transport errors are not answers."""
import asyncio
import json
from types import SimpleNamespace

from humanbench.harness import run_model as runner
from humanbench.harness.cards import fingerprint, load_tasks, split_dir


def test_run_resume_stale_and_http_failure(tmp_path, monkeypatch):
    pkg = next(iter(load_tasks([split_dir('dev')]).values()))
    monkeypatch.setattr(runner, 'TRIALS', tmp_path)
    monkeypatch.setattr(runner, 'load_tasks', lambda _: {pkg.card.id: pkg})
    monkeypatch.setenv('TEST_MODEL_KEY', 'local-test')
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({'choices': [{'message': {'content': '<think>plan</think>Hello'},
                                           'finish_reason': 'stop'}],
                               'usage': {'prompt_tokens': 10, 'completion_tokens': 8}}).encode()
    def request(req, timeout):
        calls.append(json.loads(req.data))
        assert req.full_url == 'http://local/v1/chat/completions'
        return Response()
    monkeypatch.setattr(runner.urllib.request, 'urlopen', request)
    args = SimpleNamespace(run_id='example', model='vendor/model', base_url='http://local/v1',
                           api_key_env='TEST_MODEL_KEY', samples=1, concurrency=1, max_tokens=100,
                           track='instructed', split='dev', tasks=None, effort='low', timeout=3)
    def run(): asyncio.run(runner.run(args))
    run()
    path = tmp_path / 'example/raw/model' / pkg.card.id / 's1.json'
    record = json.loads(path.read_text())
    assert record['output'] == 'Hello' and not record['infra_failure']
    assert record['card_fingerprint'] == fingerprint(pkg)
    assert calls[0]['messages'][1]['content'] == pkg.model_prompt()
    assert calls[0]['reasoning_effort'] == 'low'
    run()
    assert len(calls) == 1
    record['card_fingerprint'] = 'stale'
    path.write_text(json.dumps(record))
    run()
    assert len(calls) == 2
    record['card_fingerprint'] = 'stale'
    path.write_text(json.dumps(record))
    def fail(*args, **kwargs):
        import urllib.error
        raise urllib.error.HTTPError('http://local', 503, 'unavailable', {}, None)
    monkeypatch.setattr(runner.urllib.request, 'urlopen', fail)
    run()
    record = json.loads(path.read_text())
    assert record['infra_failure'] and record['output'] == ''
    assert 'deterministic' not in record
    assert '503' in record['error_tail']
    monkeypatch.setattr(runner.urllib.request, 'urlopen', request)
    run()
    assert not json.loads(path.read_text())['infra_failure']


def test_api_judge_route(monkeypatch):
    from humanbench.harness import floor_run
    monkeypatch.setattr(runner, 'configured_call', lambda *a: {'ok': True, 'text': 'vote'})
    assert asyncio.run(floor_run._call('api:example', 'prompt')) == 'vote'
