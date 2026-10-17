# Retained dependency artifacts, CPU only

The exact OCI-extracted CPython 3.11.13 interpreter on native Linux ran the
included sources under `-I -S -B`. The host libc differs from the pinned image.
The startup audit checked all 87 retained wheel archives and 24,223 payload
files against outer RECORDs, including the proposed setuptools hook bytes.
No startup hook or release module ran. The separate closure check verified all
87 pinned distributions and 174 active dependency edges, with Linux/Python
markers and requested extras. The verified packaging parser was imported
from its retained wheel; all archives were rechecked after metadata analysis.

The included lock identifies public archives. Local absolute artifact paths
are retained privately; regenerate the audit manifest from those filenames
under your own archive root. These receipts establish archived bytes and
metadata closure. They prove no installation, startup activation, release/source
binding, runtime imports, GPU engagement, clearance or spending authority.
