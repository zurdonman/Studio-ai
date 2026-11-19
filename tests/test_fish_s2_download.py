"""CPU-only tests for the preparatory manifest; no model inference."""
import hashlib

import pytest

from studio.engines import fish_s2_download as fish


def blob(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def test_manifest_has_pinned_model_codec_tokenizer_and_license():
    assert len(fish.REVISION) == 40
    assert {'LICENSE.md', 'codec.pth', 'config.json', 'tokenizer.json',
            'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors',
            'model.safetensors.index.json'} <= fish.FILES.keys()
    assert sum(item[0] for item in fish.FILES.values()) > 10_000_000_000
    for size, kind, digest in fish.FILES.values():
        assert size > 0
        assert (kind, len(digest)) in {('git', 40), ('sha256', 64)}
        int(digest, 16)


def test_pinned_download_and_reuse(monkeypatch, tmp_path):
    contents = {'config.json': b'config', 'weight.bin': b'weights'}
    monkeypatch.setattr(fish, 'FILES', {
        'config.json': (6, 'git', blob(contents['config.json'])),
        'weight.bin': (7, 'sha256', hashlib.sha256(contents['weight.bin']).hexdigest()),
    })
    calls = []

    def download(repo_id, filename, revision, local_dir):
        calls.append((repo_id, filename, revision))
        (tmp_path / filename).write_bytes(contents[filename])

    monkeypatch.setattr(fish, 'hf_hub_download', download)
    fish.main(tmp_path)
    fish.main(tmp_path)
    assert calls == [(fish.REPO, name, fish.REVISION) for name in contents]


def test_mismatch_and_symlink_are_not_overwritten(monkeypatch, tmp_path):
    expected = b'good'
    monkeypatch.setattr(fish, 'FILES', {'file': (4, 'git', blob(expected))})
    def no_download(*args, **kwargs):
        pytest.fail('an existing file must not be replaced')
    monkeypatch.setattr(fish, 'hf_hub_download', no_download)
    path = tmp_path / 'file'
    path.write_bytes(b'evil')
    with pytest.raises(RuntimeError, match='FISH_FILE_MISMATCH'):
        fish.main(tmp_path)
    assert path.read_bytes() == b'evil'
    path.unlink()
    target = tmp_path / 'target'
    target.write_bytes(expected)
    path.symlink_to(target)
    with pytest.raises(RuntimeError, match='FISH_FILE_MISMATCH'):
        fish.main(tmp_path)
    assert target.read_bytes() == expected
