# Supplied Python function snapshots

`ra_python_function.FunctionBinding(original, reference)` repeatedly compares one
original Python function with a supplied reference. `check(original)` preserves
that original object and checks its complete marshalled code, nested code objects,
name, qualified name, module, docstring, defaults, keyword defaults, annotations,
attributes, closure values and type parameters. It invokes neither function.

Metadata uses exact types and bounded immutable projections. Opaque values,
subclasses, cycles, empty closure cells and deferred annotations refuse. The last
case refuses before accessing annotations because that access can execute code
on Python 3.14. Any failed repeated check makes that binding terminal.

This is supplied snapshot equality only. Matching forged reference/live functions
can pass. Imports, global dependencies and attribute or parameter dispatch remain
outside its authority; the helper does not inspect or authenticate those edges.
Code and metadata hashes are interpreter-specific. There is no source, class,
site, native-extension, native-read or consumer authentication, execution wrapper,
worker inventory expansion or ABBA wiring. Cooperating serialized callers are
required; the binding's private fields do not authenticate arbitrary same-process
tampering or close races between checks.
