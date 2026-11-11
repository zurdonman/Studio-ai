"""CPU integration tests, using tones rather than AI-generated music."""
import json
import subprocess

import numpy as np


def test_music_mix_through_mcp(temp_studio_env):
    import studio.mcp_server as mcp
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info
    _, root = temp_studio_env
    for index, frequency in [(1, 440), (2, 880)]:
        subprocess.run([get_ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i",
                        f"sine=frequency={frequency}:sample_rate=48000:duration=1", "-y",
                        str(root / "uploads" / f"music_tone_{index}.wav")], check=True)
    reply = mcp._execute_mcp_tool("studio_mix_music_tracks", {"tracks": [
        {"audio_rel": "uploads/music_tone_1.wav", "start_sec": 0, "volume": .5},
        {"audio_rel": "uploads/music_tone_2.wav", "start_sec": 1, "volume": .5}], "wait": True})
    data = json.loads(reply["content"][0]["text"])
    assert data["status"] == "completed", data
    output = root / data["result"]["rel_path"]
    assert 1.99 <= probe_media_info(output)["duration_seconds"] <= 2.01
    raw = subprocess.run([get_ffmpeg_bin(), "-v", "error", "-i", str(output), "-ac", "1", "-ar", "48000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, dtype="<f4")
    for t, expected in [(.2, 440), (1.2, 880)]:
        chunk = arr[int(t*48000):int((t+.5)*48000)]
        freq = np.fft.rfftfreq(len(chunk), 1/48000)[np.argmax(abs(np.fft.rfft(chunk)))]
        assert abs(freq - expected) < 5
    assert output.with_suffix('.json').exists()


def test_music_mix_validation_and_separate_views(temp_studio_env):
    client, _ = temp_studio_env
    assert client.post('/api/audio/music-mix', json={'tracks': []}).status_code == 422
    assert client.post('/api/audio/music-mix', json={'tracks': [{'audio_rel': '', 'volume': 1}]}).status_code == 422
    assert client.post('/api/audio/music-mix', json={'tracks': [{'audio_rel': '../outside', 'volume': 1}]}).status_code in (400, 404)
    assert client.post('/api/audio/music-mix', json={'tracks': [{'audio_rel': 'a', 'volume': 2.1}]}).status_code == 422
    html = client.get('/').text
    assert 'id="tab-music"' in html and 'id="tab-tts"' in html
    assert html.count('id="bgm-model-select"') == 1


def make_midi():
    import io
    import mido
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    track.append(mido.Message('note_on', note=69, velocity=100, time=0))
    track.append(mido.Message('note_off', note=69, time=480))
    track.append(mido.MetaMessage('set_tempo', tempo=1000000, time=0))
    track.append(mido.Message('note_on', note=72, velocity=100, time=0))
    track.append(mido.Message('note_off', note=72, time=480))
    buff = io.BytesIO()
    midi.save(file=buff)
    return buff.getvalue()


def test_midi_render_via_mcp_and_tempo(temp_studio_env):
    import base64
    import studio.mcp_server as mcp
    from studio.engines.ffmpeg_utils import probe_media_info, get_ffmpeg_bin
    _, root = temp_studio_env
    reply = mcp._execute_mcp_tool('studio_render_midi_synth', {'midi_base64': base64.b64encode(make_midi()).decode(), 'instrument': 'piano', 'wait': True})
    result = json.loads(reply['content'][0]['text'])
    assert result['status'] == 'completed', result
    meta = result['result']
    assert meta['notes'] == 2
    output = root / meta['rel_path']
    assert 1.64 <= probe_media_info(output)['duration_seconds'] <= 1.66
    raw = subprocess.run([get_ffmpeg_bin(), '-v', 'error', '-i', str(output), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    samples = np.frombuffer(raw, dtype='<f4')
    for when, freq in [(.1, 440), (.7, 523.25)]:
        chunk = samples[int(when*44100):int((when+.2)*44100)]
        peak = np.fft.rfftfreq(len(chunk), 1/44100)[np.argmax(abs(np.fft.rfft(chunk)))]
        assert abs(peak-freq) < 6


def test_midi_api_and_bad_requests(temp_studio_env):
    client, _ = temp_studio_env
    assert client.post('/api/audio/midi-render', files={'file': ('test.txt', b'x')}).status_code == 400
    assert client.post('/api/audio/midi-render', files={'file': ('test.mid', b'x')}, data={'instrument': 'unknown'}).status_code == 400
    response = client.post('/api/audio/midi-render', files={'file': ('test.mid', make_midi())}, data={'instrument': 'strings'})
    assert response.status_code == 200
    import time
    for _ in range(100):
        data = client.get('/api/jobs/' + response.json()['job']['id']).json()
        job = data.get('job', data)
        if job['status'] in ('completed', 'failed', 'cancelled'):
            break
        time.sleep(.05)
    assert job['status'] == 'completed', job


def test_trim_and_silence_volume(temp_studio_env):
    import studio.mcp_server as mcp
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info
    _, root = temp_studio_env
    src = root / 'uploads' / 'trim_tone.wav'
    subprocess.run([get_ffmpeg_bin(), '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2', '-y', str(src)], check=True)
    reply = mcp._execute_mcp_tool('studio_mix_music_tracks', {'tracks': [{'audio_rel': 'uploads/trim_tone.wav', 'start_sec': .5, 'trim_start': .2, 'trim_end': 1.2, 'volume': 0}], 'wait': True})
    result = json.loads(reply['content'][0]['text'])
    assert result['status'] == 'completed', result
    out = root / result['result']['rel_path']
    assert 1.49 <= probe_media_info(out)['duration_seconds'] <= 1.51
    data = subprocess.run([get_ffmpeg_bin(), '-v', 'error', '-i', str(out), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    assert not np.any(np.frombuffer(data, dtype='<f4'))
