#!/bin/sh
# three short GPU jobs, the lock released between them so other lanes interleave
cd /share/Container/gnf4-interp/p63
rm -rf reh/out reh/*.md reh/*.json reh/run_*.log logs/chain.done
sh gpu_job.sh reh-int4 "STACKS=int4 bash /w/rehearse.sh"
sh gpu_job.sh reh-nf4 "STACKS=nf4 bash /w/rehearse.sh"
sh gpu_job.sh reh-int4nf "STACKS=int4nf bash /w/rehearse.sh"
echo "chain done $(date -u +%FT%TZ)" > logs/chain.done
