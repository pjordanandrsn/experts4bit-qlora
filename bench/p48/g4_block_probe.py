#!/usr/bin/env python3
"""Per-BLOCK NF4 probe for chosen layers (P48 P4 follow-up): distribution of per-block relative error and of block absmax/rms,
plus the fraction of weight energy in 'outlier' blocks (absmax > 6x the tensor rms). CPU/numpy, streams one tensor at a time."""
import json, struct, sys, os
import numpy as np
D, OUT = sys.argv[1], sys.argv[2]; LAYERS = [int(x) for x in sys.argv[3].split(",")]
NF4 = np.array([-1.0,-0.6961928009986877,-0.5250730514526367,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.09105003625154495,0.0,
                0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.5626170039176941,0.7229568362236023,1.0],dtype=np.float32)
BOUND=(NF4[1:]+NF4[:-1])/2
def bf16_to_f32(u16): return (u16.astype(np.uint32)<<16).view(np.float32)
idx=json.load(open(os.path.join(D,"model.safetensors.index.json")))["weight_map"]
out=open(OUT,"w")
for L in LAYERS:
    for proj in ("gate_up_proj","down_proj"):
        k=f"model.language_model.layers.{L}.experts.{proj}"; f=idx[k]
        with open(os.path.join(D,f),"rb") as fh:
            n=struct.unpack("<Q",fh.read(8))[0]; hdr=json.loads(fh.read(n)); base=8+n; s,e=hdr[k]["data_offsets"]; fh.seek(base+s); raw=fh.read(e-s)
        x=bf16_to_f32(np.frombuffer(raw,dtype=np.uint16)).reshape(hdr[k]["shape"])
        flat=x.reshape(-1); pad=(-flat.size)%64
        if pad: flat=np.concatenate([flat,np.zeros(pad,np.float32)])
        b=flat.reshape(-1,64); amax=np.abs(b).max(axis=1); rms_t=float(np.sqrt(np.mean(x.astype(np.float64)**2)))
        am=amax.copy(); am[am==0]=1.0; nb=b/am[:,None]; q=np.searchsorted(BOUND,nb); deq=NF4[q]*am[:,None]; err=deq-b
        blk_err=np.sqrt((err**2).mean(axis=1)); blk_rms=np.sqrt((b**2).mean(axis=1)); rel=blk_err/np.maximum(blk_rms,1e-30)
        energy=(b**2).sum(axis=1); tot=energy.sum()
        rec={"layer":L,"proj":proj,"n_blocks":int(b.shape[0]),"tensor_rms":rms_t,"tensor_absmax":float(np.abs(x).max()),
             "blk_rel_err_p50":float(np.percentile(rel,50)),"blk_rel_err_p90":float(np.percentile(rel,90)),"blk_rel_err_p99":float(np.percentile(rel,99)),"blk_rel_err_max":float(rel.max()),
             "blk_amax_over_blkrms_p50":float(np.percentile(amax/np.maximum(blk_rms,1e-30),50)),"blk_amax_over_blkrms_p99":float(np.percentile(amax/np.maximum(blk_rms,1e-30),99)),
             "frac_energy_in_outlier_blocks_6x":float(energy[amax>6*rms_t].sum()/tot),"frac_blocks_outlier_6x":float((amax>6*rms_t).mean()),
             "frac_energy_in_outlier_blocks_10x":float(energy[amax>10*rms_t].sum()/tot),
             "total_rel_err":float(np.sqrt((err**2).sum()/ (b**2).sum())),
             "sq_err_share_top1pct_blocks":float(np.sort((err**2).sum(axis=1))[::-1][:max(1,b.shape[0]//100)].sum()/(err**2).sum())}
        out.write(json.dumps(rec)+"\n"); out.flush(); print(L,proj,{k:(round(v,4) if isinstance(v,float) else v) for k,v in rec.items() if k not in("layer","proj")},flush=True)
