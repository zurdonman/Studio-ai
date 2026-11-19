"""Pinned Wan 2.2 I2V two-expert files. Sizes are bytes, hashes SHA-256."""
from studio.engines.motion_manifest import COMMON as MOTION_COMMON

GGUF_REPO = 'QuantStack/Wan2.2-I2V-A14B-GGUF'
GGUF_REV = '6c6717459277b9cd1f72579d78a0fd62a79e57dc'
COMFY_REPO = 'Comfy-Org/Wan-2.2'
COMFY_REV = '5df3a4a3ed0628002308022198249dbbbd045d09'
FP4_REPO = 'GitMylo/Wan_2.2_nvfp4'
FP4_REV = '5735fdd42b5e0f2c3a73462937a7c25314023497'
# Quant: filename suffix, bytes per expert, high SHA, low SHA.
GGUF = {
    'q2': ('Q2_K', 5300957696, '8426c0c02a2878f4f20c28a620a1516787045a7a56301f5b6a1d91198c15a8c1', '384b33bfebf1e3277abc2373b9d55d9672e49f3cfa92887161a6c72dd479ec5d'),
    'q3s': ('Q3_K_S', 6515012096, '2708962c357537c9f517fa49edd8397f3024057b059c3e8df827c774271e1161', '3352289be6021c783df4716686fb3bb8ec09bf8e1230350145294c78d1ce55b0'),
    'q3': ('Q3_K_M', 7176106496, 'ac3a72d74330e030fe2ded047b0ef9afc21dd90afc4f300c7aa79e1b0f9dd940', '4472e4e30a9729da0e0de36c708c45344a70799d431b17371b0c513d20a1351b'),
    'q4s': ('Q4_K_S', 8748151296, '8fc5f726ccf1214e43c6a599bdef1e63daf7fab06e71fb74081416c4b9ce200b', '54e2313ef7fbb169e72ab1d7cecdc90a04db01097c86f9f39097b8caacea50fb'),
    'q4': ('Q4_K_M', 9651728896, '836250abfaa3411694e2c9cf3a0cc18265329d5156d81aa116d5366b0f8f02e7', 'e2f98d834af009d035c6b0918268f2eba0aa8a63025ce942277e2384d40b0866'),
    'q5s': ('Q5_K_S', 10137514496, 'db0809162bbfd4f32e58247ccb680c70ac913edc918b33307b84ddacdfcbc1f5', 'd637b62d754aeb1b6b964e3e59763aa05eac7727f196681bc9e884230ad6b59c'),
    'q5_0': ('Q5_0', 10314461696, '7856fbfbeacd110993fb400547b51a5acb22fe27633bb3c49da88a057cf14e1c', '5826e1245f66d5e795a14867d8f52cee8cc64123721ea6a0da342cffb4cc75e8'),
    'q5_1': ('Q5_1', 11015696896, '69aaa6525af03cca3e868c817393540a97484e7b3c7370d95169414553510edd', '44e8c32953bdbef99f83e9a8d89efde742859f121f32038f39365676a7fc206b'),
    'q5': ('Q5_K_M', 10792055296, '163e9e5ae7ff83a4598d55242c767384be8909749dc07240d388a24838e8bac6', 'c5affcba15576959fa6d63c18261275b6407c3edd5ad3d8ab1c420e96f9d05d0'),
    'q6': ('Q6_K', 12003652096, '33ab8d99b5be745d2c939055f541d09108befe3467802fdbbf22ba8e4e62383c', '962d9c937a6e882562cd234ba87c5cfc19d6e99efbf0fc31b4dd9e43eb493ae6'),
    'q8': ('Q8_0', 15406608896, '619a66032c28e1b27882dfccc0bf93e51edb1491e8d4e4c6f291726abe4de8aa', '029c7adc74de4f7804905c5e4fb9335d0862cd2fc37191df526aeac13b64425e'),
}
SAFE = {
    'fp8': (COMFY_REPO, COMFY_REV, 'split_files/diffusion_models/wan2.2_i2v_{stage}_noise_14B_fp8_scaled.safetensors', 14294742832,
            '6122e79d55e0f235698d11d657f3b196c5273c830da00b2b013c5a048d5e6a42', '5471a457b6ac404202a5fbe6c11595a3d5641fc766b00f38763f72303fffc21e'),
    'nvfp4': (FP4_REPO, FP4_REV, 'wan2.2_i2v_{stage}_noise_14B_nvfp4_mixed.safetensors', 10535921496,
              '5378a9ae4858d5d6f83086af0a92effa8944b7f01a308d281056204cbe42b0c6', '31b07082bd14ddc4300cf04813fa7af72c28510b29067cf1301431131630cced'),
}


def files_for(quant):
    if quant in GGUF:
        suffix, size, high, low = GGUF[quant]
        experts = tuple((stage.lower(), GGUF_REPO, GGUF_REV, f'{stage}Noise/Wan2.2-I2V-A14B-{stage}Noise-{suffix}.gguf', size, digest)
                        for stage, digest in (('High', high), ('Low', low)))
    elif quant in SAFE:
        repo, revision, template, size, high, low = SAFE[quant]
        experts = tuple((stage, repo, revision, template.format(stage=stage), size, digest)
                        for stage, digest in (('high', high), ('low', low)))
    else:
        raise ValueError('WAN22_QUANT_INVALID')
    # Wan 2.1 VAE and UMT5 as used in the official Comfy I2V template.
    return (*experts, MOTION_COMMON[0], MOTION_COMMON[2])
