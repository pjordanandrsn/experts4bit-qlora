import hashlib,json,pathlib,urllib.request,tarfile,posixpath,time
w=pathlib.Path(__file__).parent
m=json.loads((w/'ra-image-manifest.json').read_text())
c=json.loads((w/'ra-image-config.json').read_text())
layers=iter(m['layers']); selected=[]
for h in c['history']:
 if not h.get('empty_layer'):
  layer=next(layers)
  if h.get('created_by')=='COPY /opt/conda /opt/conda # buildkit':selected.append(layer)
assert len(selected)==1
pin=selected[0]; root=w/'ra-proof-python'; root.mkdir(exist_ok=True)
t=json.load(urllib.request.urlopen('https://auth.docker.io/token?service=registry.docker.io&scope=repository:pytorch/pytorch:pull'))['token']
r=urllib.request.urlopen(urllib.request.Request('https://registry-1.docker.io/v2/pytorch/pytorch/blobs/'+pin['digest'],headers={'Authorization':'Bearer '+t}),timeout=60)
class Reader:
 def __init__(self,r):self.r=r;self.h=hashlib.sha256();self.n=0;self.next=256*1024**2
 def read(self,n=-1):
  b=self.r.read(n);self.h.update(b);self.n+=len(b)
  if self.n>=self.next: print(json.dumps({'compressed_read':self.n,'expected':pin['size']}),flush=True);self.next+=256*1024**2
  return b
rr=Reader(r); files=[]
with tarfile.open(fileobj=rr,mode='r|gz') as a:
 for p in a:
  name=p.name.rstrip('/');parts=pathlib.PurePosixPath(name).parts
  if not parts or name.startswith('/') or '..' in parts:raise ValueError('unsafe layer path')
  keep=(name.startswith('opt/conda/bin/python') or (name.startswith('opt/conda/lib/python3.') and '/site-packages' not in name) or (name.startswith('opt/conda/lib/') and len(parts)==4 and ('.so' in parts[-1])) or (name.startswith('opt/conda/conda-meta/python-') and name.endswith('.json')))
  if not keep:continue
  dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True)
  if p.isdir():dst.mkdir(exist_ok=True);continue
  if p.issym():
   target=p.linkname
   if target.startswith('/') or '..' in pathlib.PurePosixPath(target).parts:raise ValueError('unsafe selected symlink')
   dst.symlink_to(target);files.append({'path':name,'symlink':target});continue
  if not p.isfile():raise ValueError('unsupported selected member')
  h=hashlib.sha256()
  with a.extractfile(p) as src,dst.open('wb') as out:
   while b:=src.read(1024**2):h.update(b);out.write(b)
  dst.chmod(p.mode&0o777);files.append({'path':name,'sha256':h.hexdigest(),'size':p.size})
while rr.read(1024**2):pass
assert rr.n==pin['size'] and 'sha256:'+rr.h.hexdigest()==pin['digest'], 'layer integrity failed'
receipt={'schema':1,'scope':'OCI-verified extracted interpreter/stdlib only; no full image boot/install/GPU claim','image_manifest_sha256':hashlib.sha256((w/'ra-image-manifest.json').read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((w/'ra-image-config.json').read_bytes()).hexdigest(),'layer':pin,'files':files}
(w/'ra-proof-python-extraction.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'verified':True,'retained_files':len(files),'retained_bytes':sum(p.get('size',0) for p in files),'python_metadata':[p['path'] for p in files if p['path'].startswith('opt/conda/conda-meta/')]}),flush=True)
