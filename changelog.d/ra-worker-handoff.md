### RA: proposed same-process verified worker continuation

Continue the isolated payload/startup/import handoff into a fixed RA wrapper
only after checking pinned tools, inputs, stage and output boundaries. Retain
worker failures separately from a passed handoff and recheck bytes/origins after
return. CPU synthetic controls do not establish nested startup, GPU engagement
or launch clearance. Actual executor: Codex desktop.
