import wave
from pathlib import Path

import pytest


@pytest.fixture
def ref():
    # Do not import studio.config during collection: other tests configure it first.
    from studio.engines import voice_clone_reference
    return voice_clone_reference


def write_wav(path: Path, seconds: float = 5.0) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\0\0" * int(8000 * seconds))


def test_transcript_requirements(ref):
    assert ref.validate_clone_request("qwen3-tts-0.6b-base", "hola", None) == ""
    assert ref.validate_clone_request("fish-speech-1.5", "hola", "  muestra  ") == "muestra"
    with pytest.raises(ValueError, match="transcripción"):
        ref.validate_clone_request("fish-speech-1.5", "hola", " ")
    with pytest.raises(ValueError, match="texto"):
        ref.validate_clone_request("qwen3-tts-0.6b-base", " ", "")
    with pytest.raises(ValueError, match="desconocido"):
        ref.validate_clone_request("other", "hola", "muestra")
    with pytest.raises(ValueError, match="demasiado larga"):
        ref.validate_clone_request("qwen3-tts-0.6b-base", "hola", "x" * 2001)


def test_conversion_and_cleanup(tmp_path, monkeypatch, ref):
    source = tmp_path / "sample.wav"
    write_wav(source)
    monkeypatch.setattr(ref, "TEMP_DIR", tmp_path / "temp")
    with ref.prepared_reference_audio(source) as output:
        assert output.exists()
        with wave.open(str(output), "rb") as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (24000, 1, 2)
        generated = output
    assert not generated.exists()
    assert source.exists()


def test_cleanup_on_failure_and_duration_limits(tmp_path, monkeypatch, ref):
    monkeypatch.setattr(ref, "TEMP_DIR", tmp_path / "temp")
    source = tmp_path / "sample.wav"
    write_wav(source)
    with pytest.raises(RuntimeError, match="cancelled"):
        with ref.prepared_reference_audio(source) as output:
            generated = output
            raise RuntimeError("cancelled")
    assert not generated.exists()
    write_wav(source, 1)
    with pytest.raises(ValueError, match="3 y 15"):
        with ref.prepared_reference_audio(source):
            pass
    write_wav(source, 16)
    with pytest.raises(ValueError, match="3 y 15"):
        with ref.prepared_reference_audio(source):
            pass


def test_invalid_or_too_large_file(tmp_path, monkeypatch, ref):
    source = tmp_path / "missing.wav"
    with pytest.raises(ValueError, match="válido"):
        with ref.prepared_reference_audio(source):
            pass
    source.write_bytes(b"bad audio")
    monkeypatch.setattr(ref, "MAX_REFERENCE_BYTES", 4)
    with pytest.raises(ValueError, match="50 MiB"):
        with ref.prepared_reference_audio(source):
            pass
