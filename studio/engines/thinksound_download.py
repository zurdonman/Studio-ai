"""Pinned ThinkSound light V2A weights and feature encoders."""
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download
from studio.engines.qwen_clone_download import verify

FILES = (
    ('model', 'FunAudioLLM/ThinkSound', '7f152d1a088f91b21307337060216f0dec8d4a8f', 'thinksound_light.ckpt', 5733246918, 'sha256', '5232db0ca3f1bca319958ea81181267e525228c226f62015d2b0fc624ce1df82'),
    ('model', 'FunAudioLLM/ThinkSound', '7f152d1a088f91b21307337060216f0dec8d4a8f', 'vae.ckpt', 2521255758, 'sha256', '8132c32ec65bbba2f1457f2659136c5f2cb31f9902af1c659d78c9f7d3aa619e'),
    ('model', 'FunAudioLLM/ThinkSound', '7f152d1a088f91b21307337060216f0dec8d4a8f', 'synchformer_state_dict.pth', 950058171, 'sha256', '8aff082f2df5c3bc52759db0c865c7ee772ae6400b860d1b7e90413f2defb67c'),
    ('clip', 'facebook/metaclip-h14-fullcc2.5b', 'ff60fc0029c2f7f7010c9a5b4869163e79903c08', 'added_tokens.json', 57, 'git', '73baf615fb9bb9f6135141084dedf0ea506d951c'),
    ('clip', 'facebook/metaclip-h14-fullcc2.5b', 'ff60fc0029c2f7f7010c9a5b4869163e79903c08', 'config.json', 602, 'git', '6fabb2355824f834724694f7c9fef556c51c357b'),
    ('clip', 'facebook/metaclip-h14-fullcc2.5b', 'ff60fc0029c2f7f7010c9a5b4869163e79903c08', 'merges.txt', 524619, 'git', '76e821f1b6f0a9709293c3b6b51ed90980b3166b'),
    ('clip', 'facebook/metaclip-h14-fullcc2.5b', 'ff60fc0029c2f7f7010c9a5b4869163e79903c08', 'model.safetensors', 3944549372, 'sha256', '6173e1a29c7b449ce5ebe788936625c909b4d9059eb89bc4ec80bbdc00bc6571'),
    ('clip', 'facebook/metaclip-h14-fullcc2.5b', 'ff60fc0029c2f7f7010c9a5b4869163e79903c08', 'preprocessor_config.json', 504, 'git', '6b87f9bc7511073f32189607eaf7d534529b025d'),
    ('clip', 'facebook/metaclip-h14-fullcc2.5b', 'ff60fc0029c2f7f7010c9a5b4869163e79903c08', 'special_tokens_map.json', 133, 'git', 'd073e26c77c9d2dcb62c109b4488565af35fb5bb'),
    ('clip', 'facebook/metaclip-h14-fullcc2.5b', 'ff60fc0029c2f7f7010c9a5b4869163e79903c08', 'tokenizer_config.json', 950, 'git', '081766a98204773e1064c69e0e0e880ec4c13e06'),
    ('clip', 'facebook/metaclip-h14-fullcc2.5b', 'ff60fc0029c2f7f7010c9a5b4869163e79903c08', 'vocab.json', 1059962, 'git', '469be27c5c010538f845f518c4f5e8574c78f7c8'),
    ('t5', 'google/t5-v1_1-xl', 'e827710dfd12562d3ef5ea28853b2d0b31859bc6', 'config.json', 591, 'git', 'd6dd3022afa58ac49aebae60e01368633e8ee9fa'),
    ('t5', 'google/t5-v1_1-xl', 'e827710dfd12562d3ef5ea28853b2d0b31859bc6', 'pytorch_model.bin', 11399285345, 'sha256', '72be4ab4cb05720c91e72995ad26fb5e2785fc1648c80ad0561c45ce432447fc'),
    ('t5', 'google/t5-v1_1-xl', 'e827710dfd12562d3ef5ea28853b2d0b31859bc6', 'special_tokens_map.json', 1786, 'git', '881bdbffc06e471924ecea57f962bc5f8e2a9f21'),
    ('t5', 'google/t5-v1_1-xl', 'e827710dfd12562d3ef5ea28853b2d0b31859bc6', 'spiece.model', 791656, 'git', '4e28ff6ebdf584f5372d9de68867399142435d9a'),
    ('t5', 'google/t5-v1_1-xl', 'e827710dfd12562d3ef5ea28853b2d0b31859bc6', 'tokenizer_config.json', 1857, 'git', 'b114c318caf72f6e89ea92e0755c41327a453198'),
)


def main(root: Path):
    for i, (group, repo, revision, name, size, kind, digest) in enumerate(FILES, 1):
        folder = root/group
        folder.mkdir(parents=True, exist_ok=True)
        target = folder/name
        print(f'[{i}/{len(FILES)}] Preparando {group}/{name} ({size/1024**3:.2f} GiB)', flush=True)
        if target.is_symlink(): raise RuntimeError(f'THINKSOUND_FILE_MISMATCH: {name} es un enlace')
        if not target.exists(): hf_hub_download(repo, filename=name, revision=revision, local_dir=str(folder))
        if target.is_symlink() or not verify(target, size, kind, digest):
            raise RuntimeError(f'THINKSOUND_FILE_MISMATCH: {name}; no se sobrescribe')
        print(f'[{i}/{len(FILES)}] Verificado {group}/{name}', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
