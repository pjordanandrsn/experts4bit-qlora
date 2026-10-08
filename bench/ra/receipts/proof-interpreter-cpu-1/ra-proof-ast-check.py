import ast,hashlib,json,sys,sysconfig,types,textwrap,pathlib,platform
x=json.loads(pathlib.Path(sys.argv[1]).read_text())
m=types.ModuleType('ra_fallback');exec(compile(x['helper'],'reviewed-ra-fallback','exec'),m.__dict__)
rows=[]
for c in x['cases']:
 tree=ast.parse(textwrap.dedent(c['source']) if c['body'] else c['source'])
 target=tree.body[0] if c['body'] else tree
 actual=m.ast_digest(target)
 assert actual==c['sha256'],c['name']
 tree.body[0].body.insert(0,ast.parse('RA_MUTATION = 1').body[0]) if c['body'] else tree.body.append(ast.parse('RA_MUTATION = 1').body[0])
 assert m.ast_digest(target)!=actual,c['name']
 rows.append({'name':c['name'],'sha256':actual,'matched':True,'mutation_rejected':True})
r={'schema':1,'scope':'CPU canonical AST validation on OCI-verified extracted proof interpreter; no full-image/install/GPU evidence','python':sys.version,'platform':platform.platform(),'soabi':sysconfig.get_config_var('SOABI'),'executable_sha256':hashlib.sha256(pathlib.Path(sys.executable).read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256(x['helper'].encode()).hexdigest(),'rows':rows,'proves_gpu_engagement':False}
print(json.dumps(r,sort_keys=True,indent=2))
