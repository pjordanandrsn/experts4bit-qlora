# SC2 (request-level serving, lane of #846): claimed

The serving-campaign session holds lane SC2. This file is the claim, and the pre-registration follows on this branch
before any rental. The subject is request-level serving behind an OpenAI-compatible endpoint:
- TTFT and ITL at low load;
- a Poisson load curve with a goodput SLO;
- the capacity ceiling;
- a prompt-length sweep.

The engines are e4b's `serve_paged` against vLLM, SGLang and llama.cpp's server, on one RTX 5090, through one shared
request driver.
