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
tree=ast.parse('def f(x): return x + 1').body[0]
expected=m.ast_digest(tree)
if hasattr(tree,'type_params'): del tree.type_params
assert m.ast_digest(tree)==expected
tree.type_params=[]
assert m.ast_digest(tree)==expected
tree.type_params=[ast.Name(id='T',ctx=ast.Load())]
native='type_params' in ast.FunctionDef._fields
assert (m.ast_digest(tree)!=expected) if native else (m.ast_digest(tree)==expected)
r={'empty_optional_matches':True,'type_params_native_field':native,'optional_field_contract_passed':True,'schema':1,'scope':'CPU canonical AST validation on OCI-verified extracted proof interpreter; no full-image/install/GPU evidence','python':sys.version,'platform':platform.platform(),'soabi':sysconfig.get_config_var('SOABI'),'executable_sha256':hashlib.sha256(pathlib.Path(sys.executable).read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256(x['helper'].encode()).hexdigest(),'rows':rows,'proves_gpu_engagement':False}
print(json.dumps(r,sort_keys=True,indent=2))
