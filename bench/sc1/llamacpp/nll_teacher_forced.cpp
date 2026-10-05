// bench/sc1/llamacpp/nll_teacher_forced.cpp -- SC1 (experts4bit-qlora#846): teacher-forced NLL on a FIXED token-id
// window through libllama, in the two shapes the lane scores every engine in.
//
//   decode  : llama_decode the first prompt_len ids as one batch (prefill), then for t in 0..steps-1 decode EXACTLY ONE
//             token (llama_batch_get_one of the TRUE id ids[prompt_len+t]) and accumulate -log_softmax(logits)[ids[prompt_len+t+1]].
//             This is the single-token path llama-server's decode takes (MMVQ for quantized weights at n_tokens == 1).
//   prefill : llama_decode ids[:prompt_len+steps] as ONE batch (n_batch >= that; logits requested for the scored positions)
//             and score positions prompt_len .. prompt_len+steps-1 -- the logit at position i predicts token i+1, so these
//             are the SAME targets decode mode scores (the prompt's own last logit, which predicts ids[prompt_len], is not
//             scored in either shape: step_decomp feeds cont[0] = ids[prompt_len] as the first teacher-forced input).
//
// Both shapes score the same `steps` targets ids[prompt_len+1 .. prompt_len+steps], so their mean NLLs are directly
// comparable; mirrors bench/p39/step_decomp.py's K8 loop (prompt 512, 2048 steps, nll += -log_softmax(logits[-1].float())[next],
// mean, ppl = exp). NLL is accumulated in double precision from the float logits with a max-shifted log-sum-exp.
// (Calibration on the Mac caught the one-off alternative -- scoring positions prompt_len-1..prompt_len+steps-2 -- as a
// 0.013-nat "prefill vs decode" gap that was simply a different 256-token window; the two shapes must share the window.)
//
// text_sha256 = sha256 of the first prompt_len+steps+1 ids serialised as little-endian int64 -- byte-identical to
// step_decomp's `ids[:prompt_len+steps+1].numpy().tobytes()` digest (`tok(...).input_ids[0]` is torch.int64), so a K8
// receipt and an SC1 llama.cpp receipt can REFUSE each other when they scored different text (k8_gate's rule).
//
// SC1g A4 (#846): --named <bin> (int32 little-endian [steps x --named-k], the reference's named token ids per scored
// position, sc1g_kl.py's artifact) writes --named-out <bin> (float64 little-endian [steps x (k+1)]: the log-probs of those
// ids at that position, then the target's), from the same float logits and the same max-shifted log-sum-exp as the NLL.
//
// Needs the tokens file to hold at least prompt_len+steps+1 ids (refuses, exit 3). No dependency outside the pinned
// llama.cpp tree: llama.h + ggml headers + the vendored nlohmann/json; SHA-256 is implemented here.
//
// Build: bench/sc1/llamacpp/build_harness.sh <llama.cpp src> [<build dir>] -- links libllama + libggml from <build>/bin.

#include "llama.h"
#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

#ifndef SC1_LLAMA_COMMIT
#define SC1_LLAMA_COMMIT "unknown"
#endif
#ifndef SC1_LLAMA_TAG
#define SC1_LLAMA_TAG "unknown"
#endif

// ---------------------------------------------------------------------------------------------------------------------
// SHA-256 (FIPS 180-4) -- the receipt hashes the GGUF file and the id window without an OpenSSL dependency.
// ---------------------------------------------------------------------------------------------------------------------
namespace {

struct Sha256 {
    uint32_t h[8];
    uint64_t total = 0;
    uint8_t  buf[64];
    size_t   buf_len = 0;

    Sha256() {
        static const uint32_t init[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                         0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        std::memcpy(h, init, sizeof(h));
    }

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t * p) {
        static const uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) | (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
        }
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const void * data, size_t n) {
        const uint8_t * p = static_cast<const uint8_t *>(data);
        total += n;
        if (buf_len) {
            size_t take = std::min(n, 64 - buf_len);
            std::memcpy(buf + buf_len, p, take);
            buf_len += take; p += take; n -= take;
            if (buf_len == 64) { block(buf); buf_len = 0; }
        }
        while (n >= 64) { block(p); p += 64; n -= 64; }
        if (n) { std::memcpy(buf, p, n); buf_len = n; }
    }

    std::string hexdigest() {
        uint64_t bits = total * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t zero = 0;
        while (buf_len != 56) { update(&zero, 1); }
        uint8_t len[8];
        for (int i = 0; i < 8; i++) { len[i] = uint8_t(bits >> (56 - 8 * i)); }
        update(len, 8);
        char out[65];
        for (int i = 0; i < 8; i++) { std::snprintf(out + 8 * i, 9, "%08x", h[i]); }
        return std::string(out, 64);
    }
};

std::string sha256_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { return std::string(); }
    Sha256 s;
    std::vector<char> buf(1 << 20);
    while (f) {
        f.read(buf.data(), (std::streamsize) buf.size());
        std::streamsize got = f.gcount();
        if (got > 0) { s.update(buf.data(), (size_t) got); }
    }
    return s.hexdigest();
}

// ---------------------------------------------------------------------------------------------------------------------
// llama log capture -- the receipt carries the lines the box script asserts on (offload count, flash_attn, n_ubatch).
// ---------------------------------------------------------------------------------------------------------------------
struct LogState {
    std::mutex m;
    std::vector<std::string> lines;
    std::string partial;
    bool echo = true;
};
LogState g_log;

void log_cb(ggml_log_level level, const char * text, void * ud) {
    (void) level; (void) ud;
    std::lock_guard<std::mutex> lk(g_log.m);
    if (g_log.echo) { std::fputs(text, stderr); std::fflush(stderr); }
    g_log.partial += text;
    size_t pos;
    while ((pos = g_log.partial.find('\n')) != std::string::npos) {
        g_log.lines.push_back(g_log.partial.substr(0, pos));
        g_log.partial.erase(0, pos + 1);
    }
}

std::vector<std::string> log_grep(const std::regex & re) {
    std::lock_guard<std::mutex> lk(g_log.m);
    std::vector<std::string> out;
    for (const auto & l : g_log.lines) { if (std::regex_search(l, re)) { out.push_back(l); } }
    return out;
}

struct Args {
    std::string model, tokens, out;
    std::string mode = "decode";
    std::string flash_attn = "on";
    std::string type_kv = "f16";     // KV cache type for both K and V (llama-server's default f16)
    int prompt_len = 512;
    int steps = 2048;
    int n_gpu_layers = 99;
    int n_batch = 0;      // 0 = derived from the mode
    int n_ubatch = 512;   // llama-server's default physical batch
    int threads = 0;      // 0 = hardware_concurrency/2 clamped to [1, 8] (10 spinning threads on an 8P+2E M1 ran 200x slower)
    bool quiet = false;
    std::string named, named_out;   // SC1g A4: named token ids in, their log-probs out
    int named_k = 64;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <gguf> --tokens <json with key \"ids\"> --out <json>\n"
        "          [--prompt-len 512] [--steps 2048] [--mode decode|prefill] [--n-gpu-layers 99]\n"
        "          [--flash-attn on|off|auto] [--type-kv f16|f32|bf16|q8_0] [--n-batch N] [--n-ubatch 512] [--threads N] [--quiet]\n"
        "          [--named <int32 bin [steps x k]> --named-out <float64 bin [steps x (k+1)]> [--named-k 64]]\n"
        "exit codes: 2 usage, 3 refusal (too few ids / id out of vocab), 4 load failure, 5 decode failure\n", argv0);
}

Args parse_args(int argc, char ** argv) {
    Args a;
    auto need = [&](int & i) -> const char * {
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); usage(argv[0]); std::exit(2); }
        return argv[++i];
    };
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        if      (k == "--model")        { a.model = need(i); }
        else if (k == "--tokens")       { a.tokens = need(i); }
        else if (k == "--out")          { a.out = need(i); }
        else if (k == "--mode")         { a.mode = need(i); }
        else if (k == "--flash-attn")   { a.flash_attn = need(i); }
        else if (k == "--type-kv")      { a.type_kv = need(i); }
        else if (k == "--prompt-len")   { a.prompt_len = std::atoi(need(i)); }
        else if (k == "--steps")        { a.steps = std::atoi(need(i)); }
        else if (k == "--n-gpu-layers") { a.n_gpu_layers = std::atoi(need(i)); }
        else if (k == "--n-batch")      { a.n_batch = std::atoi(need(i)); }
        else if (k == "--n-ubatch")     { a.n_ubatch = std::atoi(need(i)); }
        else if (k == "--threads")      { a.threads = std::atoi(need(i)); }
        else if (k == "--quiet")        { a.quiet = true; }
        else if (k == "--named")        { a.named = need(i); }
        else if (k == "--named-out")    { a.named_out = need(i); }
        else if (k == "--named-k")      { a.named_k = std::atoi(need(i)); }
        else if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); usage(argv[0]); std::exit(2); }
    }
    if (a.model.empty() || a.tokens.empty() || a.out.empty()) { usage(argv[0]); std::exit(2); }
    if (a.mode != "decode" && a.mode != "prefill") { std::fprintf(stderr, "--mode must be decode or prefill\n"); std::exit(2); }
    if (a.flash_attn != "on" && a.flash_attn != "off" && a.flash_attn != "auto") { std::fprintf(stderr, "--flash-attn must be on|off|auto\n"); std::exit(2); }
    if (a.type_kv != "f16" && a.type_kv != "f32" && a.type_kv != "bf16" && a.type_kv != "q8_0") { std::fprintf(stderr, "--type-kv must be f16|f32|bf16|q8_0\n"); std::exit(2); }
    if (a.prompt_len < 1 || a.steps < 1) { std::fprintf(stderr, "--prompt-len and --steps must be >= 1\n"); std::exit(2); }
    if (a.named.empty() != a.named_out.empty() || a.named_k < 1) { std::fprintf(stderr, "--named and --named-out go together; --named-k >= 1\n"); std::exit(2); }
    return a;
}

// -log_softmax(logits)[target] in double from float logits, max-shifted; also reports the argmax (top-1 agreement).
double nll_of(const float * lg, int n_vocab, int target, int * argmax) {
    float m = lg[0];
    int am = 0;
    for (int i = 1; i < n_vocab; i++) { if (lg[i] > m) { m = lg[i]; am = i; } }
    double s = 0.0;
    for (int i = 0; i < n_vocab; i++) { s += std::exp((double) lg[i] - (double) m); }
    *argmax = am;
    return (double) m + std::log(s) - (double) lg[target];
}

// SC1g A4: log_softmax(logits) at the k named ids, then at the target, into out[0..k]; same arithmetic as nll_of.
void named_lps(const float * lg, int n_vocab, const int32_t * idx, int k, int target, double * out) {
    float m = lg[0];
    for (int i = 1; i < n_vocab; i++) { if (lg[i] > m) { m = lg[i]; } }
    double s = 0.0;
    for (int i = 0; i < n_vocab; i++) { s += std::exp((double) lg[i] - (double) m); }
    const double lse = (double) m + std::log(s);
    for (int j = 0; j < k; j++) { out[j] = (double) lg[idx[j]] - lse; }
    out[k] = (double) lg[target] - lse;
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

int main(int argc, char ** argv) {
    const double t_proc0 = now_s();
    Args a = parse_args(argc, argv);

    // ---- the id window --------------------------------------------------------------------------------------------
    std::vector<llama_token> ids;
    {
        std::ifstream f(a.tokens);
        if (!f) { std::fprintf(stderr, "cannot open tokens file %s\n", a.tokens.c_str()); return 2; }
        json jt;
        try { jt = json::parse(f); } catch (const std::exception & e) { std::fprintf(stderr, "tokens file is not JSON: %s\n", e.what()); return 2; }
        if (!jt.contains("ids") || !jt["ids"].is_array()) { std::fprintf(stderr, "tokens file needs key \"ids\" = list of ints\n"); return 2; }
        try { ids = jt["ids"].get<std::vector<llama_token>>(); } catch (const std::exception & e) { std::fprintf(stderr, "\"ids\" must be ints: %s\n", e.what()); return 2; }
    }
    const size_t need = (size_t) a.prompt_len + (size_t) a.steps + 1;
    if (ids.size() < need) {
        std::fprintf(stderr, "REFUSE: tokens file holds %zu ids but --prompt-len %d + --steps %d needs %zu (prompt + steps + 1 target); "
                             "widen the window, do not shorten the steps\n", ids.size(), a.prompt_len, a.steps, need);
        return 3;
    }
    std::string text_sha;
    {
        Sha256 s;
        for (size_t i = 0; i < need; i++) {
            const int64_t v = (int64_t) ids[i];
            uint8_t le[8];
            for (int k = 0; k < 8; k++) { le[k] = (uint8_t) ((uint64_t) v >> (8 * k)); }
            s.update(le, 8);
        }
        text_sha = s.hexdigest();
    }

    // ---- GGUF digest in the background while the model loads ----------------------------------------------------
    std::string model_sha;
    std::thread sha_thread([&] { model_sha = sha256_file(a.model); });

    g_log.echo = !a.quiet;
    llama_log_set(log_cb, nullptr);
    ggml_backend_load_all();      // only does work when the backends were built as dynamic libraries (GGML_BACKEND_DL)
    llama_backend_init();

    // ---- model ------------------------------------------------------------------------------------------------------
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = a.n_gpu_layers;
    const double t_load0 = now_s();
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    if (!model) { std::fprintf(stderr, "failed to load model %s\n", a.model.c_str()); sha_thread.join(); return 4; }
    const double load_s = now_s() - t_load0;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (size_t i = 0; i < need; i++) {
        if (ids[i] < 0 || ids[i] >= n_vocab) {
            std::fprintf(stderr, "REFUSE: id %d at index %zu is outside the model vocab (n_vocab %d)\n", ids[i], i, n_vocab);
            llama_model_free(model); sha_thread.join(); return 3;
        }
    }

    // ---- context ----------------------------------------------------------------------------------------------------
    const uint32_t n_tok_prefill = (uint32_t) (a.prompt_len + a.steps);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = (uint32_t) need;                       // libllama pads to a multiple of 256
    cp.n_seq_max = 1;
    if (a.mode == "decode") {
        cp.n_batch = (uint32_t) std::max(a.n_batch, a.prompt_len);            // one batch for the prompt, then 1-token batches
    } else {
        cp.n_batch = (uint32_t) (a.n_batch > 0 ? a.n_batch : (int) n_tok_prefill);
        if (cp.n_batch < n_tok_prefill) {
            std::fprintf(stderr, "prefill mode needs --n-batch >= prompt_len+steps (%u), got %u\n", n_tok_prefill, cp.n_batch);
            llama_model_free(model); sha_thread.join(); return 2;
        }
    }
    cp.n_ubatch = (uint32_t) std::min<int>(a.n_ubatch, (int) cp.n_batch);
    cp.flash_attn_type = a.flash_attn == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                       : a.flash_attn == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    const int threads = a.threads > 0 ? a.threads : (int) std::min(8u, std::max(1u, std::thread::hardware_concurrency() / 2));
    cp.n_threads = cp.n_threads_batch = threads;
    cp.type_k = cp.type_v = a.type_kv == "f16" ? GGML_TYPE_F16 : a.type_kv == "f32" ? GGML_TYPE_F32
                          : a.type_kv == "bf16" ? GGML_TYPE_BF16 : GGML_TYPE_Q8_0;
    cp.no_perf = false;
    cp.embeddings = false;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "failed to create context\n"); llama_model_free(model); sha_thread.join(); return 4; }

    // ---- score ------------------------------------------------------------------------------------------------------
    double nll = 0.0;
    int top1 = 0;
    double prefill_s = 0.0, loop_s = 0.0;
    int rc = 0;

    // SC1g A4: the named ids, read and range-checked before any decode
    std::vector<int32_t> named_ids;
    std::vector<double> named_buf;
    const int K = a.named_k;
    if (!a.named.empty()) {
        std::ifstream nf(a.named, std::ios::binary);
        named_ids.resize((size_t) a.steps * K);
        if (!nf.read(reinterpret_cast<char *>(named_ids.data()), (std::streamsize) (named_ids.size() * sizeof(int32_t))) || nf.peek() != EOF) {
            std::fprintf(stderr, "--named %s is not exactly %d x %d int32\n", a.named.c_str(), a.steps, K);
            llama_free(ctx); llama_model_free(model); sha_thread.join(); return 3;
        }
        for (int32_t v : named_ids) { if (v < 0 || v >= n_vocab) { std::fprintf(stderr, "--named id %d out of vocab\n", v); llama_free(ctx); llama_model_free(model); sha_thread.join(); return 3; } }
        named_buf.assign((size_t) a.steps * (K + 1), 0.0);
    }

    if (a.mode == "decode") {
        llama_batch b = llama_batch_init(a.prompt_len, 0, 1);
        for (int i = 0; i < a.prompt_len; i++) {
            b.token[i] = ids[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
            b.logits[i] = (i == a.prompt_len - 1) ? 1 : 0;           // the prompt's last logit is not scored (step_decomp scores from cont[0])
        }
        b.n_tokens = a.prompt_len;
        double t0 = now_s();
        rc = llama_decode(ctx, b);
        prefill_s = now_s() - t0;
        llama_batch_free(b);
        if (rc != 0) { std::fprintf(stderr, "prompt llama_decode failed rc=%d\n", rc); llama_free(ctx); llama_model_free(model); sha_thread.join(); return 5; }

        double t1 = now_s();
        for (int t = 0; t < a.steps; t++) {
            llama_token tok = ids[a.prompt_len + t];                 // TEACHER-FORCED: the true token, never the argmax
            llama_batch one = llama_batch_get_one(&tok, 1);          // pos/seq tracked by the context; logits = last token only
            rc = llama_decode(ctx, one);
            if (rc != 0) { std::fprintf(stderr, "step %d llama_decode failed rc=%d\n", t, rc); llama_free(ctx); llama_model_free(model); sha_thread.join(); return 5; }
            const float * lg = llama_get_logits_ith(ctx, -1);
            int am = 0;
            const int target = ids[a.prompt_len + t + 1];
            nll += nll_of(lg, n_vocab, target, &am);
            top1 += (am == target);
            if (!named_ids.empty()) { named_lps(lg, n_vocab, &named_ids[(size_t) t * K], K, target, &named_buf[(size_t) t * (K + 1)]); }
        }
        loop_s = now_s() - t1;
    } else {
        llama_batch b = llama_batch_init((int) n_tok_prefill, 0, 1);
        for (uint32_t i = 0; i < n_tok_prefill; i++) {
            b.token[i] = ids[i]; b.pos[i] = (llama_pos) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
            b.logits[i] = (i >= (uint32_t) a.prompt_len) ? 1 : 0;   // positions prompt_len .. n-1: their logits predict ids[prompt_len+1 .. n]
        }
        b.n_tokens = (int) n_tok_prefill;
        double t0 = now_s();
        rc = llama_decode(ctx, b);
        prefill_s = now_s() - t0;
        if (rc != 0) { std::fprintf(stderr, "batch llama_decode failed rc=%d\n", rc); llama_batch_free(b); llama_free(ctx); llama_model_free(model); sha_thread.join(); return 5; }
        double t1 = now_s();
        for (int i = a.prompt_len; i <= (int) n_tok_prefill - 1; i++) {      // same targets as decode mode: ids[i+1]
            const float * lg = llama_get_logits_ith(ctx, i);
            if (!lg) { std::fprintf(stderr, "no logits for batch index %d\n", i); llama_batch_free(b); llama_free(ctx); llama_model_free(model); sha_thread.join(); return 5; }
            int am = 0;
            const int target = ids[i + 1];
            nll += nll_of(lg, n_vocab, target, &am);
            top1 += (am == target);
            if (!named_ids.empty()) {
                const int t = i - a.prompt_len;
                named_lps(lg, n_vocab, &named_ids[(size_t) t * K], K, target, &named_buf[(size_t) t * (K + 1)]);
            }
        }
        loop_s = now_s() - t1;     // the scoring pass only; the forward is prefill_s
        llama_batch_free(b);
    }

    const double mean_nll = nll / a.steps;
    const llama_perf_context_data perf = llama_perf_context(ctx);
    sha_thread.join();

    // ---- receipt ----------------------------------------------------------------------------------------------------
    json out;
    out["harness"] = "bench/sc1/llamacpp/nll_teacher_forced";
    out["mode"] = a.mode;
    out["prompt_len"] = a.prompt_len;
    out["steps"] = a.steps;
    out["tokens_scored"] = a.steps;
    out["mean_nll"] = mean_nll;
    out["ppl"] = std::exp(mean_nll);
    out["top1_agree"] = top1;
    out["top1_agree_frac"] = (double) top1 / a.steps;
    out["text_sha256"] = text_sha;
    out["ids_in_file"] = ids.size();
    out["ids_used"] = need;
    out["targets"] = json{{"first_index", a.prompt_len + 1}, {"last_index", a.prompt_len + a.steps}};
    if (!named_ids.empty()) {
        std::ofstream of(a.named_out, std::ios::binary);
        of.write(reinterpret_cast<const char *>(named_buf.data()), (std::streamsize) (named_buf.size() * sizeof(double)));
        out["named"] = json{{"in", a.named}, {"out", a.named_out}, {"k", K}, {"ok", (bool) of},
                            {"layout", "float64 [steps x (k+1)]: the named ids' log-probs, then the target's"}};
    }

    char desc[256] = {0};
    llama_model_desc(model, desc, sizeof(desc));
    char name[512] = {0};
    const int nlen = llama_model_meta_val_str(model, "general.name", name, sizeof(name));
    out["model"] = a.model;
    out["model_sha256"] = model_sha;
    out["model_desc"] = desc;
    out["model_general_name"] = nlen >= 0 ? json(std::string(name)) : json(nullptr);
    out["model_size_bytes"] = llama_model_size(model);
    out["model_n_params"] = llama_model_n_params(model);
    out["n_layer"] = llama_model_n_layer(model);
    out["n_vocab"] = n_vocab;
    out["n_ctx_train"] = llama_model_n_ctx_train(model);

    out["llama_commit"] = SC1_LLAMA_COMMIT;
    out["llama_tag"] = SC1_LLAMA_TAG;
    out["system_info"] = llama_print_system_info();

    out["n_gpu_layers_requested"] = a.n_gpu_layers;
    {
        auto lines = log_grep(std::regex("offloaded ([0-9]+)/([0-9]+) layers to GPU"));
        if (!lines.empty()) {
            std::smatch m;
            std::regex_search(lines.back(), m, std::regex("offloaded ([0-9]+)/([0-9]+) layers to GPU"));
            out["gpu_layers_offloaded"] = json{{"n", std::stoi(m[1])}, {"of", std::stoi(m[2])}, {"all", std::stoi(m[1]) == std::stoi(m[2])}};
            out["offload_line"] = lines.back();
        } else {
            out["gpu_layers_offloaded"] = nullptr;
            out["offload_line"] = nullptr;
        }
    }
    out["flash_attn_requested"] = a.flash_attn;
    out["flash_attn_log"] = log_grep(std::regex("flash_attn|[Ff]lash [Aa]ttention"));
    out["n_ctx"] = llama_n_ctx(ctx);
    out["n_ctx_seq"] = llama_n_ctx_seq(ctx);
    out["n_batch"] = llama_n_batch(ctx);
    out["n_ubatch"] = llama_n_ubatch(ctx);
    out["n_threads"] = threads;
    out["type_k"] = ggml_type_name(cp.type_k);
    out["type_v"] = ggml_type_name(cp.type_v);

    out["load_s"] = load_s;
    out["prefill_s"] = prefill_s;
    out["loop_wall_s"] = loop_s;
    if (a.mode == "decode") {
        out["scored_tok_s"] = a.steps / loop_s;                       // one llama_decode per scored token
        out["decode_ms_per_step"] = loop_s / a.steps * 1e3;
    } else {
        out["scored_tok_s"] = a.steps / prefill_s;                    // scored positions over the single-batch forward
        out["prefill_tok_s"] = n_tok_prefill / prefill_s;             // every token of the batch over the same forward
        out["tokens_in_batch"] = n_tok_prefill;
    }
    out["wall_s"] = now_s() - t_proc0;
    out["perf"] = json{{"t_load_ms", perf.t_load_ms}, {"t_p_eval_ms", perf.t_p_eval_ms}, {"n_p_eval", perf.n_p_eval},
                       {"t_eval_ms", perf.t_eval_ms}, {"n_eval", perf.n_eval}, {"n_reused", perf.n_reused}};
    out["basis"] = a.mode == "decode"
        ? "teacher-forced through llama_decode one token at a time after a single prompt batch; targets ids[prompt_len+1..prompt_len+steps]; double-precision log-sum-exp from float logits"
        : "one llama_decode over prompt_len+steps tokens with logits at positions prompt_len..prompt_len+steps-1; same targets as decode mode (ids[prompt_len+1..prompt_len+steps])";
    {
        std::lock_guard<std::mutex> lk(g_log.m);
        const size_t n = g_log.lines.size();
        const size_t from = n > 40 ? n - 40 : 0;
        out["log_tail"] = std::vector<std::string>(g_log.lines.begin() + from, g_log.lines.end());
    }

    {
        std::ofstream f(a.out);
        if (!f) { std::fprintf(stderr, "cannot write %s\n", a.out.c_str()); llama_free(ctx); llama_model_free(model); return 2; }
        f << out.dump(1) << "\n";
    }
    std::printf("SC1_LLAMACPP_NLL mode=%s steps=%d prompt_len=%d nll=%.6f ppl=%.5f top1=%.4f sha=%s tok_s=%.2f out=%s\n",
                a.mode.c_str(), a.steps, a.prompt_len, mean_nll, std::exp(mean_nll), (double) top1 / a.steps,
                text_sha.substr(0, 12).c_str(), a.mode == "decode" ? a.steps / loop_s : a.steps / prefill_s, a.out.c_str());
    std::fflush(stdout);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
