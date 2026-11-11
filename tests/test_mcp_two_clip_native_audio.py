"""Exercise the existing MCP montage with real synthetic A/V, no models or GPU."""
import json
import subprocess

import numpy as np


def test_two_video_clips_keep_both_audio_tracks_through_mcp(temp_studio_env):
    import studio.mcp_server as mcp
    from studio.engines.ffmpeg_utils import get_ffmpeg_bin, probe_media_info
    _, root = temp_studio_env
    ffmpeg = get_ffmpeg_bin()
    for index, tone in [(1, 440), (2, 880)]:
        target = root / "uploads" / f"mcp_tone_{index}.mp4"
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=c=blue:s=256x256:r=24:d=1", "-f", "lavfi", "-i",
                        f"sine=frequency={tone}:sample_rate=48000:duration=1", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", "-y", str(target)], check=True)
    result = mcp._execute_mcp_tool("studio_render_montage", {
        "items": [{"media_rel": "uploads/mcp_tone_1.mp4"}, {"media_rel": "uploads/mcp_tone_2.mp4"}],
        "width": 256, "height": 256, "fps": 24, "fade_seconds": 0, "wait": True,
    })
    data = json.loads(result["content"][0]["text"])
    assert data["status"] == "completed", data
    output = root / data["result"]["rel_path"]
    info = probe_media_info(output)
    assert info["has_video"] and info["has_audio"]
    assert 1.9 <= info["duration_seconds"] <= 2.3
    raw = subprocess.run([ffmpeg, "-v", "error", "-i", str(output), "-vn", "-ac", "1", "-ar", "48000",
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    audio = np.frombuffer(raw, dtype="<f4")
    def peak(start):
        samples = audio[int(start * 48000):int((start + 0.4) * 48000)]
        assert samples.size == 19200
        return np.fft.rfftfreq(samples.size, 1/48000)[np.argmax(abs(np.fft.rfft(samples)))]
    assert abs(peak(.3) - 440) < 10
    assert abs(peak(1.3) - 880) < 10
