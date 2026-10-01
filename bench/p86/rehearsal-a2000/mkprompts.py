import json, hashlib, random
random.seed(0)
for B in (1, 4):
    prompts = [[random.randrange(1000, 30000) for _ in range(64)] for _ in range(B)]
    json.dump({"batch": B, "prompts": prompts, "prompts_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest()},
              open(f"/root/p86r/prompts_b{B}.json", "w"))
print("prompts ok")
