"""H3 prebuilt acquisition contracts; no network or GPU needed."""
import io
import zipfile

import pytest


def _zip(path, entries):
    with zipfile.ZipFile(path, 'w') as z:
        for name, data in entries:
            z.writestr(name, data)


def test_archive_requires_flat_safe_files_and_revision(tmp_path):
    from studio.engines import minimax_h3_prebuilt as p
    stem = 'release'
    archive = tmp_path / 'runtime.zip'
    valid = [(stem + '/sd-cli', b'exe'), (stem + '/UNSLOTH_BUILD.txt', f'source_commit: {p.REVISION}')]
    _zip(archive, valid)
    p.unpack(archive, tmp_path / 'out', stem)
    assert (tmp_path / 'out' / 'sd-cli').stat().st_mode & 0o100
    _zip(archive, valid + [(stem + '/../escape', b'bad')])
    with pytest.raises(ValueError, match='ruta no segura'):
        p.unpack(archive, tmp_path / 'bad', stem)
    assert not (tmp_path / 'escape').exists()
    symlink = zipfile.ZipInfo(stem + '/link')
    symlink.create_system = 3
    symlink.external_attr = 0o120777 << 16
    _zip(archive, valid + [(symlink, '../escape')])
    with pytest.raises(ValueError, match='ruta no segura'):
        p.unpack(archive, tmp_path / 'bad2', stem)
    _zip(archive, [(stem + '/sd-cli', b'exe'), (stem + '/UNSLOTH_BUILD.txt', 'wrong')])
    with pytest.raises(ValueError, match='revisión'):
        p.unpack(archive, tmp_path / 'bad3', stem)


def test_download_checks_and_extracts_valid_bundle(tmp_path, monkeypatch):
    import hashlib
    from studio.engines import minimax_h3_prebuilt as p
    stem = f'sd-{p.TAG}-bin-Linux-Ubuntu-22.04-x86_64-cuda12'
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as z:
        z.writestr(stem + '/sd-cli', b'exe')
        z.writestr(stem + '/UNSLOTH_BUILD.txt', f'source_commit: {p.REVISION}')
    content = data.getvalue()
    monkeypatch.setitem(p.ASSETS, '12', (len(content), hashlib.sha256(content).hexdigest()))
    monkeypatch.setattr(p.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(content))
    p.download('12', tmp_path)
    assert (tmp_path / 'bundle' / 'sd-cli').read_bytes() == b'exe'
    assert not (tmp_path / 'runtime.zip').exists()


def test_download_rejects_wrong_hash_before_extracting(tmp_path, monkeypatch):
    from studio.engines import minimax_h3_prebuilt as p
    data = b'wrong'
    monkeypatch.setitem(p.ASSETS, '12', (len(data), '0' * 64))
    monkeypatch.setattr(p.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(data))
    with pytest.raises(ValueError, match='SHA-256'):
        p.download('12', tmp_path)
    assert not (tmp_path / 'bundle').exists()


def test_prebuilt_installs_then_reuses_without_recompiling(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import minimax_h3 as h3
    from studio.jobs import JobRecord
    monkeypatch.setattr(h3, 'ROOT', tmp_path)
    monkeypatch.setattr(h3, '_cuda_major', lambda env: '12')
    calls = []
    def command(args, job, **kwargs):
        calls.append(args)
        if '-m' in args:
            bundle = __import__('pathlib').Path(args[-1]) / 'bundle'
            bundle.mkdir()
            (bundle / 'sd-cli').write_text('fake')
    monkeypatch.setattr(h3.common, '_command', command)
    job = JobRecord('download', 'test')
    first = h3._install(job)
    assert first == tmp_path / 'prebuilt-cuda12' / 'sd-cli'
    assert len(calls) == 2
    assert h3._install(job) == first
    assert len(calls) == 3 and '-m' not in calls[-1]


def test_prebuilt_failure_falls_back_but_cancellation_never_compiles(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import minimax_h3 as h3
    from studio.jobs import JobCancelledError, JobRecord
    from studio.memory_policy import MemoryPressureError
    monkeypatch.setattr(h3, 'ROOT', tmp_path)
    monkeypatch.setattr(h3, '_cuda_major', lambda env: '12')
    job = JobRecord('download', 'test')
    for error in (RuntimeError('broken download'), JobCancelledError(), MemoryPressureError('pressure')):
        def fail(*args, **kwargs):
            raise error
        monkeypatch.setattr(h3.common, '_command', fail)
        if isinstance(error, RuntimeError) and not isinstance(error, MemoryPressureError):
            assert h3._install_prebuilt(job) is None
            assert not (tmp_path / 'prebuilt-cuda12').exists()
        else:
            with pytest.raises(type(error)):
                h3._install_prebuilt(job)
        assert not list(tmp_path.glob('.h3-prebuilt-*'))


def test_existing_compiled_runtime_does_not_redownload(temp_studio_env, monkeypatch, tmp_path):
    from studio.engines import minimax_h3 as h3
    from studio.jobs import JobRecord
    import json
    binary = tmp_path / 'build' / 'bin' / 'sd-cli'
    binary.parent.mkdir(parents=True)
    binary.write_text('original')
    (tmp_path / 'ready.json').write_text(json.dumps({'revision': h3.RUNTIME_REV, 'cuda': True}))
    monkeypatch.setattr(h3, 'ROOT', tmp_path)
    monkeypatch.setattr(h3, '_install_prebuilt', lambda job: pytest.fail('must reuse compilation'))
    assert h3._install(JobRecord('download', 'test')) == binary
