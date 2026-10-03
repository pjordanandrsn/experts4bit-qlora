# SC1 cross-box read

```
reducer: main bb044779 sha256 89ffa3fc20d12072
sc1a-5090-4 -> box A; its staged reducer sha256 89ffa3fc20d12072; e4b c19dd52
sc1b-5090-3 -> box B; its staged reducer sha256 799b3302eaba28df; e4b 9dd712b
sc1c-5090-5 -> box C; its staged reducer sha256 799b3302eaba28df; e4b 9dd712b
```

# SC1 cross-box anchor ratios (vllm/gptq_graph over the box's e4b sched anchor; measured wherever both arms are VALID and stable, the quotation flag beside -- A11)
| B | box A vllm/e4b-sched (interval; quoted?) | box B vllm/e4b-sched (interval; quoted?) | box C vllm/e4b-sched (interval; quoted?) | gap | within 5 % |
|---|---|---|---|---|---|
| 1 | 1.187 [1.183, 1.191]; measured, not quoted | 1.203 [1.191, 1.216]; measured, not quoted | 1.167 [1.157, 1.176]; measured, not quoted | +3.1 % | True |
| 16 | 1.167 [1.165, 1.170]; measured, not quoted | 1.172 [1.159, 1.185]; measured, not quoted | 1.177 [1.172, 1.182]; measured, not quoted | +0.8 % | True |

Substitution licence (box A, |lic/rtn - 1| <= 2 % at both B): HOLDS -- B=1 +0.2 %, B=16 +0.2 %. Boxes B/C anchors are RTN: licensed as the e4b stand-in.

P13: HOLD -- B=1 A 1.187, B 1.203, C 1.167 (gap 3.1 % vs 5 %); B=16 A 1.167, B 1.172, C 1.177 (gap 0.8 % vs 5 %)
wrote /private/tmp/claude-501/-Users-jordananderson-code/d479c659-f03e-44eb-9883-8d2aa60a2184/scratchpad/xbox/A/RESULTS-sc1-A-generated.md
wrote /private/tmp/claude-501/-Users-jordananderson-code/d479c659-f03e-44eb-9883-8d2aa60a2184/scratchpad/xbox/B/RESULTS-sc1-B-generated.md
wrote /private/tmp/claude-501/-Users-jordananderson-code/d479c659-f03e-44eb-9883-8d2aa60a2184/scratchpad/xbox/C/RESULTS-sc1-C-generated.md
