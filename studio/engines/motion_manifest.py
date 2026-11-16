"""Pinned motion recipes. Tuples: role, repo, revision, path, bytes, sha256."""
REBEL_WAN = 'realrebelai/Wan-Animate-2_GGUFs'
WAN_REV = '46b287218ffe4223cfedba0c06e36db356d34ff9'
REBEL_SCAIL = 'realrebelai/SCAIL-2_GGUF'
SCAIL_REV = '16e572157fbc2ffa76782dd5134a9262dc1a40cf'
COMFY = 'Comfy-Org/Wan-Animate-2'
COMFY_REV = 'e7181fe1896b7f2ce34120d1a3663413548240d7'
SC_COMFY = 'Comfy-Org/SCAIL-2'
SC_REV = 'fe3c728bc793ba21ca674688f822afb709ad44fb'
SAM = ('sam', 'Comfy-Org/sam3.1', '7bb8374780a725b4353ed31f3a9395c9742b5621', 'checkpoints/sam3.1_multiplex_fp16.safetensors', 1745546848, '9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03')
COMMON = (
    ('umt5', COMFY, COMFY_REV, 'text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors', 6735906897, 'c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68'),
    ('vision', COMFY, COMFY_REV, 'clip_vision/clip_vision_h.safetensors', 1264219396, '64a7ef761bfccbadbaa3da77366aac4185a6c58fa5de5f589b42a65bcc21f161'),
    ('vae', COMFY, COMFY_REV, 'vae/Wan2_1_VAE_bf16.safetensors', 253806278, '1ab9a32cc2c740f6e39d80d367ce5dcc28db8c71b79b28670546b8973e9d75f9'),
)
WAN = {
 'q2': ('Distilled/Wan-Animate-2-14B-TURBO-Q2_K.gguf',6496443360,'0c668e40b56aa246f2025b15c15ac41274f639b8cced2b6f3415f2e589d25300'),
 'q3': ('Distilled/Wan-Animate-2-14B-TURBO-Q3_K_M.gguf',8584584160,'0d212bcaacbb50c352f1a21e4868a3922dfc0c53fe08bd1ba317bae16f432624'),
 'q4': ('Distilled/Wan-Animate-2-14B-TURBO-Q4_K_M.gguf',11338734560,'e533882d59bc01e0efa5d820a3d95a3fd01cec9012a692e335cb2ebd04ecb5e1'),
 'q5': ('Distilled/Wan-Animate-2-14B-TURBO-Q5_K_M.gguf',12741204960,'c43e655ea8e76420630526291c75359c7fd4114efb592b07f67c0d99ab51296f'),
 'q8': ('Distilled/Wan-Animate-2-14B-TURBO-Q8_0.gguf',18142190560,'ee8c65e23ea69c7a29515ac171b1e6ebed5ced0ecdbda34cb973927b0d5f6a01'),
}
SCAIL = {
 'q2': ('SCAIL-2-Q2_K.gguf',6020521632,'29ca1600879e2a2796d5fb4ae86fd057924c7d2b5c49141b48ea12efb06bfad6'),
 'q3': ('SCAIL-2-Q3_K_M.gguf',8108662432,'86c718053716c217f30ff887c01f0f3e247c2f4319944e32f59632ee54976b3d'),
 'q4': ('SCAIL-2-Q4_K_M.gguf',10862812832,'ced59d192230b56990c7cb2ce905ae72cac08f564639f6198edff57e035f2646'),
 'q5': ('SCAIL-2-Q5_K_M.gguf',12265283232,'cf61b07081f01bdc7b81afeb0bb4a2264f3b2677f43923bcfce362d21e2525a9'),
 'q8': ('SCAIL-2-Q8_0.gguf',17666268832,'ba4e6ef95eb071b8df357bb0d171a406b9b0acd501483be46e95046d4e3112dd'),
}
SAFE = {
 ('wan','int8'): (COMFY, COMFY_REV, 'diffusion_models/wan_animate_2_distill_int8_convrot.safetensors', 16653175528, 'd2e566ecac3f164cb1a6b63dc7d868d6678c9affa49fbceb3abe4f9458f89c8a'),
 ('wan','bf16'): (COMFY, COMFY_REV, 'diffusion_models/wan_animate_2_distill_bf16.safetensors', 32789894104, 'eb05383cb4b71228bbeb959f4468431a93ab25d0399054b317f399be114d08f9'),
 ('scail','fp8'): (SC_COMFY, SC_REV, 'diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors', 17694586857, '11513b4697ecf566de0cb74660c478f301fb6699a62b10369e91a6ed0fd6b083'),
 ('scail','fp16'): (SC_COMFY, SC_REV, 'diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors', 32794040328, 'e3f9cd52184949ee8ffdc01ba14de4c619e0ddfa09ae105cb7efce5c8dc9fe43'),
}


def files_for(engine, quant):
    table = WAN if engine == 'wan' else SCAIL if engine == 'scail' else None
    if table is None:
        raise ValueError('Motor de movimiento desconocido')
    if quant in table:
        filename, size, digest = table[quant]
        item = ('dit', REBEL_WAN if engine == 'wan' else REBEL_SCAIL, WAN_REV if engine == 'wan' else SCAIL_REV, filename, size, digest)
    elif (engine, quant) in SAFE:
        item = ('dit', *SAFE[(engine, quant)])
    else:
        raise ValueError('Cuantización no disponible para este motor')
    return (item, *COMMON, *((SAM,) if engine == 'scail' else ()))
