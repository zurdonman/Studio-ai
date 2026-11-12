"""Pinned, explicit LTX-2.3/Gemma artifacts for the experimental GGUF graph.

The Gemma model is separately licensed. File sizes are bytes, hashes are LFS
SHA-256 hashes published by the corresponding Hugging Face model metadata.
"""
LTX_REPO = "unsloth/LTX-2.3-GGUF"
LTX_REV = "96e8ed4925ead3db9ff4d0084f165ef6a74f28d0"
GEMMA_REPO = "unsloth/gemma-3-12b-it-qat-GGUF"
GEMMA_REV = "858acec7ec0541a46c39985c95d3b52d8f3ab183"

FILES = (
    ("dit", LTX_REPO, LTX_REV, "distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf", 14326856736, "4e4459bee04199bf93187ba385729f6b7d8e874d754b72d26e751fe2066f4358"),
    ("video_vae", LTX_REPO, LTX_REV, "vae/ltx-2.3-22b-distilled_video_vae.safetensors", 1452256522, "e68d6d8f8a42942ac9b862cc315beb3bc30805a8876c7ad63ba5bf7a2b8e168a"),
    ("audio_vae", LTX_REPO, LTX_REV, "vae/ltx-2.3-22b-distilled_audio_vae.safetensors", 364853140, "3cd6a6eb8cb28f5ecc12f1f3126952b2a3d2b0b42ad3270e63cefafafe0d9b57"),
    ("connectors", LTX_REPO, LTX_REV, "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors", 2312144712, "c61cbb396e2a8175d8b2da51f0fdac885a4ccd22c9f64dafa5aa2c455dc8a507"),
    ("text_encoder", GEMMA_REPO, GEMMA_REV, "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf", 7432229248, "da98f81c86916ed1c76b3eeda56b25cb7b8352b01093e2edb8028110fe2cb53b"),
)
