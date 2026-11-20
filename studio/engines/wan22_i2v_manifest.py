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


# Wan2.2-Fun-A14B-InP: a distinct 14B-per-expert model supporting text,
# image, and first/last frames. QuantStack exports are NOT I2V-A14B weights.
FUN_REPO = 'QuantStack/Wan2.2-Fun-A14B-InP-GGUF'
FUN_REV = '6a24c79dc08b02363a67239c7091fffb365efd0d'
FUN_GGUF = {
 'q2': ('Q2_K',5310798464,'828b1e659f3e46beb4388984e58d824f0a7aaca1f71b995c5cc1ded409d4a6f5','7c1aa823290dbd6378fe7dfc5a94bf37838b899021ecaabe8eb1e2501086f236'),
 'q3s': ('Q3_K_S',6524852864,'96b2a936a5d2abd5ca894b0f82e3300133762359e9cd43c64cadd9e8e5157f17','dbac606a191ef4841883aa0ea01860fbd28d2ea44f6ceed394f4cdeccf2de8ec'),
 'q3': ('Q3_K_M',7185947264,'348e499a7b189543850e6ba0fa8c346b0ebf268acd2bc207ad0b370ec608bc89','f310d4a872b4a503e858cbd97268947d0d4ab63597305a5e2357e95175737c11'),
 'q4_0': ('Q4_0',8567937664,'46f197ef83d9e5d33b26ae49a6dfc369081629e1d4a94d25d162bb81f207824a','fb424345830b5d045e43261bc15f53b1a72e6a6b8e0d1cc22476b296cd9767e8'),
 'q4_1': ('Q4_1',9269172864,'d683748908ca5d7ee1e1ce5ba13aed5fee558ddc1876a659f6c7386b06da6703','e9d5938aedeb30463ac5640e03c614c667df7672a0d7f1d3c94237a16297a064'),
 'q4s': ('Q4_K_S',8757992064,'6e0703a4ef02cba81677ac7d6cde1d229ddd77b32f338714d30640e6f102c429','181f6a8684d763403a5cdb04a7704ae6b6b94e2db92d201d46cb7d2da0da9dca'),
 'q4': ('Q4_K_M',9661569664,'a7f1a6a377d37d9453efcae7de03f720067bf3e25f52081818d0526cd60d9d46','5354d1e4655647064ac86c6fcdecd032254c2c59a87ccda63ce1cf20834aebd1'),
 'q5_0': ('Q5_0',10324302464,'3cd3f50b5cc9bf997beec65ec7c370fcd22edbb20d7cc6d8934f3f3eea6515fc','dde40d406450f464db87c3371db3d7c2fc0d98327a2f79c03a5ac43b6400c9b6'),
 'q5_1': ('Q5_1',11025537664,'1f3cc77bbbb541b18ed51de3e97b24d4d7579f3e0a15b92d44e58e44110b0a3d','7910eb494f3155ab132203721d5c7a4586c8290046d59c323f7cfdef78d2ffaa'),
 'q5s': ('Q5_K_S',10147355264,'2c92e5433a90d4874c1d73934b5f098469b60c99bdaac067c004dcf96d3807fe','e16aae643a561551a68fe4bac2ec9691d00e5d2286ac2775c1d05d0e6eb5c1e1'),
 'q5': ('Q5_K_M',10801896064,'04417746fab36d72fe7fe025b11aca8404c9c52bc045181be58147d60f5cb8ab','3e2cfdc87fc170a87fb43be0b7554b4e644225bdd84a2a37d93ecd6fa39f6885'),
 'q6': ('Q6_K',12013492864,'07274b920e7b0610f911c45ea860572a44616552768a7518682419f2d2633e6d','28b1e179f5237549c5eab03ab0e460879298e76480570c0ce40a5e92a8209468'),
 'q8': ('Q8_0',15416449664,'83a8f323170067afbe03cde982d3cfbf89aa2ab03846b96e7786294d98ed9b6a','64cd19132b76acfa6e5afcd4a0dceb9c43e9f33a2757a7791030090788b3e077'),
}
FUN_FP8 = (COMFY_REPO, COMFY_REV, 'split_files/diffusion_models/wan2.2_fun_inpaint_{stage}_noise_14B_fp8_scaled.safetensors',14294743520,
           '6106c137d8a921bc17d579c3e9deaa4fc4fadbc77fbe733f43ff19bb33ec3995','a3981f3e0f79c59fb3cffe1240ee34656bbc57e599101fac2b7140461037ab9c')


def files_for(quant, recipe='i2v'):
    if recipe not in ('i2v', 'fun'): raise ValueError('WAN22_RECIPE_INVALID')
    gguf = FUN_GGUF if recipe == 'fun' else GGUF
    safe = {'fp8': FUN_FP8} if recipe == 'fun' else SAFE
    if quant in gguf:
        suffix, size, high, low = gguf[quant]
        repo, rev = (FUN_REPO, FUN_REV) if recipe == 'fun' else (GGUF_REPO, GGUF_REV)
        template = '{stage}Noise/Wan2.2-Fun-A14B-InP_{stage}Noise-{suffix}.gguf' if recipe == 'fun' else '{stage}Noise/Wan2.2-I2V-A14B-{stage}Noise-{suffix}.gguf'
        experts = tuple((stage.lower(),repo,rev,template.format(stage=stage,suffix=suffix),size,digest)
                        for stage,digest in (('High',high),('Low',low)))
    elif quant in safe:
        repo,revision,template,size,high,low = safe[quant]
        experts = tuple((stage,repo,revision,template.format(stage=stage),size,digest)
                        for stage,digest in (('high',high),('low',low)))
    else: raise ValueError('WAN22_QUANT_INVALID: variante no disponible para esta receta')
    return (*experts, MOTION_COMMON[0], MOTION_COMMON[2])


# Comfy's documented 4-step pair for Wan2.2 Fun InP FP8, high and low
# matched to their expert. SHA-256 and sizes from the pinned Comfy revision.
FUN_LIGHTX2V = (
    ('lora_high', COMFY_REPO, COMFY_REV, 'split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors', 1226977424, 'd176c808d6fc461999b68e321efcb7501b20b8c3797523ed0df14f7d1deff11e'),
    ('lora_low', COMFY_REPO, COMFY_REV, 'split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors', 1226977424, '024f21de095bc8fad9809ded3e9e49a2e170dcf27075da8145ba7d60d8aab7f9'),
)

def recipe_files(quant, recipe='i2v', speed='none'):
    files = files_for(quant, recipe)
    if speed == 'none': return files
    if speed == 'lightx2v' and recipe == 'fun' and quant == 'fp8': return (*files, *FUN_LIGHTX2V)
    raise ValueError('WAN22_LORA_INCOMPATIBLE: LightX2V solo para Fun InP FP8 inicio-final en esta receta.')
