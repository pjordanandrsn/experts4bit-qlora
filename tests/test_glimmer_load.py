"""End-to-end GGUF -> model load, on a SYNTHETIC mini-Glimmer.

The real checkpoint is 18 GB; the load PATH is what needs testing, not the
weights. So this builds a tiny Glimmer (2 layers, small dims) with the real
architecture, writes a real GGUF carrying the real tensor names via gguf-py's
writer, loads it through the production path, and asserts:

  * every parameter the keymap promises is materialized (nothing left on meta);
  * the values arrive with the right ARITHMETIC — a centered norm comes back
    ``gguf - 1``, the final norm comes back unchanged, the untied head is NOT
    the embedding;
  * the qk-norm scalars are dropped, and a non-uniform one raises;
  * a GGUF missing a tensor is refused in strict mode rather than loading a
    model with a hole in it.

That last pair is the point of the whole lane: the failure mode being
engineered out is a load that succeeds and computes nonsense.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")

from experts4bit_qlora.glimmer import GlimmerKeymapError  # noqa: E402
from experts4bit_qlora.glimmer_load import load_glimmer_text_tower  # noqa: E402

pytest.importorskip("kquant_ref", reason="needs grouped-nf4-gemm k-quant lane")

QK = 3.87
N_LAYERS = 2
HID = 64
INTER = 128
HEADS = 4
KV_HEADS = 1
HEAD_DIM = 16
VOCAB = 100


def _mini_config():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("meta-models/Muse-Glimmer-30B")
    t = cfg.text_config
    t.num_hidden_layers = N_LAYERS
    t.hidden_size = HID
    t.intermediate_size = INTER
    t.num_attention_heads = HEADS
    t.num_key_value_heads = KV_HEADS
    t.head_dim = HEAD_DIM
    t.vocab_size = VOCAB
    t.layer_types = (["sliding_attention"] * 3 + ["full_attention"])[:N_LAYERS]
    t.layer_rope_theta = [500000.0] * N_LAYERS
    # Shrink the vision half too so from_config stays cheap.
    v = cfg.vision_config
    v.num_hidden_layers = 1
    v.hidden_size = 32
    v.intermediate_size = 64
    v.num_attention_heads = 2
    return cfg


def _write_mini_gguf(path, *, omit=None, bad_qnorm=False):
    """Write a real GGUF with the real Glimmer text-tower tensor names."""
    w = gguf.GGUFWriter(str(path), "muse_glimmer")
    q_out = HEADS * HEAD_DIM
    kv_out = KV_HEADS * HEAD_DIM
    tensors = {
        "token_embd.weight": (VOCAB, HID),
        "output.weight": (VOCAB, HID),
        "output_norm.weight": (HID,),
    }
    for i in range(N_LAYERS):
        tensors.update({
            f"blk.{i}.attn_norm.weight": (HID,),
            f"blk.{i}.post_attention_norm.weight": (HID,),
            f"blk.{i}.ffn_norm.weight": (HID,),
            f"blk.{i}.post_ffw_norm.weight": (HID,),
            f"blk.{i}.attn_q.weight": (q_out, HID),
            f"blk.{i}.attn_k.weight": (kv_out, HID),
            f"blk.{i}.attn_v.weight": (kv_out, HID),
            f"blk.{i}.attn_output.weight": (HID, q_out),
            f"blk.{i}.attn_gate.weight": (q_out, HID),
            f"blk.{i}.ffn_gate.weight": (INTER, HID),
            f"blk.{i}.ffn_up.weight": (INTER, HID),
            f"blk.{i}.ffn_down.weight": (HID, INTER),
            f"blk.{i}.attn_q_norm.weight": (HEAD_DIM,),
            f"blk.{i}.attn_k_norm.weight": (HEAD_DIM,),
        })
    rng = np.random.default_rng(0)
    written = {}
    for name, shape in tensors.items():
        if omit and name == omit:
            continue
        if name.endswith("attn_q_norm.weight"):
            arr = (np.linspace(0.5, 4.0, shape[0]).astype(np.float32) if bad_qnorm
                   else np.full(shape, QK, dtype=np.float32))
        elif name.endswith("attn_k_norm.weight"):
            arr = np.full(shape, 1.0, dtype=np.float32)
        elif name.endswith("norm.weight"):
            # Centered norms ship with the +1 baked in; final norm centers on 0.
            base = 0.0 if name == "output_norm.weight" else 1.0
            arr = (base + rng.normal(0, 0.05, shape)).astype(np.float32)
        else:
            arr = rng.normal(0, 0.02, shape).astype(np.float32)
        # GGUF ne is reversed vs torch shape; the writer takes numpy as-is.
        w.add_tensor(name, arr)
        written[name] = arr
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return written


def _build_meta_model(cfg):
    from transformers import AutoModelForImageTextToText
    with torch.device("meta"):
        return AutoModelForImageTextToText.from_config(cfg)


def _try_setup(tmp_path, **kw):
    try:
        cfg = _mini_config()
    except Exception as e:
        pytest.skip(f"cannot obtain Glimmer config (hermetic CI): {e}")
    path = tmp_path / "mini.gguf"
    written = _write_mini_gguf(path, **kw)
    return cfg, path, written


def test_full_load_fills_every_promised_param(tmp_path):
    cfg, path, written = _try_setup(tmp_path)
    model = _build_meta_model(cfg)
    report = load_glimmer_text_tower(
        str(path), model, qk_scale_factor=QK, dtype=torch.float32)
    assert report["unfilled"] == []
    assert report["dropped"] == 2 * N_LAYERS          # the qk-norm scalars
    assert report["assigned"] == 3 + 12 * N_LAYERS
    assert report["text_only"] is True
    # Nothing promised is left on meta.
    sd = dict(model.named_parameters())
    for name in ["lm_head.weight", "model.language_model.embed_tokens.weight",
                 "model.language_model.layers.0.self_attn.q_proj.weight"]:
        assert not sd[name].is_meta, name


def test_norm_arithmetic_and_untied_head(tmp_path):
    cfg, path, written = _try_setup(tmp_path)
    model = _build_meta_model(cfg)
    load_glimmer_text_tower(str(path), model, qk_scale_factor=QK,
                            dtype=torch.float32)
    sd = dict(model.named_parameters())
    # Centered norm: model param == gguf - 1
    got = sd["model.language_model.layers.0.input_layernorm.weight"].float()
    want = torch.from_numpy(written["blk.0.attn_norm.weight"]) - 1.0
    assert torch.allclose(got, want, atol=1e-5)
    # Final norm: unchanged
    got_f = sd["model.language_model.norm.weight"].float()
    assert torch.allclose(got_f, torch.from_numpy(written["output_norm.weight"]),
                          atol=1e-5)
    # Untied head: lm_head is `output`, NOT a view of the embedding.
    head = sd["lm_head.weight"].float()
    emb = sd["model.language_model.embed_tokens.weight"].float()
    assert torch.allclose(head, torch.from_numpy(written["output.weight"]), atol=1e-5)
    assert not torch.allclose(head, emb), "untied head was aliased to the embedding"


def test_missing_tensor_is_refused_not_absorbed(tmp_path):
    cfg, path, _ = _try_setup(tmp_path, omit="blk.1.ffn_down.weight")
    model = _build_meta_model(cfg)
    with pytest.raises(GlimmerKeymapError, match="unfilled"):
        load_glimmer_text_tower(str(path), model, qk_scale_factor=QK,
                                dtype=torch.float32)


def test_learned_qk_norm_raises(tmp_path):
    cfg, path, _ = _try_setup(tmp_path, bad_qnorm=True)
    model = _build_meta_model(cfg)
    with pytest.raises(GlimmerKeymapError, match="learned qk-norm"):
        load_glimmer_text_tower(str(path), model, qk_scale_factor=QK,
                                dtype=torch.float32)


def test_wrong_num_layers_is_refused(tmp_path):
    """A caller-supplied depth that disagrees with the model must raise, not
    quietly narrow the coverage check (Bugbot #92)."""
    cfg, path, _ = _try_setup(tmp_path)
    model = _build_meta_model(cfg)
    with pytest.raises(GlimmerKeymapError, match="disagrees with the model"):
        load_glimmer_text_tower(str(path), model, qk_scale_factor=QK,
                                num_layers=N_LAYERS - 1, dtype=torch.float32)


def test_device_argument_places_weights(tmp_path):
    """Weights land on the requested device as they decode (streaming keeps
    peak host RAM at one tensor for a 30B-scale load)."""
    cfg, path, _ = _try_setup(tmp_path)
    model = _build_meta_model(cfg)
    load_glimmer_text_tower(str(path), model, qk_scale_factor=QK,
                            dtype=torch.float32, device="cpu")
    sd = dict(model.named_parameters())
    assert sd["lm_head.weight"].device.type == "cpu"
