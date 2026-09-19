"""kl_prompts_heldout.py — a HELD-OUT prompt set, disjoint from `kl_prompts.py`.

COMMITTED BEFORE THE MEASUREMENT IT GATES. `kl_prompts.py`'s 200 prompts have been used by
every lane from P44 to P51, and P48's per-layer sensitivity profile — the measurement that
chose the graded store map's tier boundaries — was taken on them. A gate for that map
therefore cannot be scored on the same prompts without being, in part, a fitted metric.
These 100 are new text, written for this purpose, and no arm, threshold or boundary has
been chosen using them.

Composition (100 prompts, the same four strata in the same proportions as the committed set,
so the two are comparable stratum by stratum):

  general   (36) — factual recall, deliberately spread from common to obscure. Obscure facts
                   degrade first under quantization, so a set of only common ones under-reports.
  technical (28) — domain prose: physics, chemistry, biology, medicine, law, economics,
                   linguistics, engineering.
  code      (28) — completion and reasoning across languages and paradigms.
  longctx    (8) — multi-hundred-token passages, where attention and KV damage surfaces.

Disjointness is by construction: every prompt was written new, and no fact, snippet or
passage topic is shared with `kl_prompts.py`. `assert_disjoint_from_committed()` checks the
mechanical part of that (no shared prompt text) and is exercised by the test suite.

Scoring, aggregation and the token-weighting caveat are identical to `kl_prompts.py`: every
prompt is scored TEACHER-FORCED over its full text, the mean is token-weighted, and the
long-context stratum dominates that mean, so per-stratum numbers must be reported beside it.
"""
from __future__ import annotations

import hashlib

_GENERAL = [
    "The capital of Canada is",
    "The freezing point of mercury in degrees Celsius is approximately",
    "The novel Midnight's Children was written by",
    "The chemical symbol for antimony is",
    "The Treaty of Westphalia was signed in the year",
    "The largest moon of Neptune is called",
    "In Norse mythology, the guardian of the Bifrost bridge is",
    "The currency of Poland is the",
    "The deepest lake in the world by volume is",
    "The inventor of the mechanical television system was",
    "The highest peak in the Caucasus range is",
    "The first person to win Nobel Prizes in two different sciences was",
    "The ancient city of Petra is located in the modern country of",
    "The chemical process by which plants convert light into sugar is called",
    "The composer of the opera The Magic Flute was",
    "The smallest bone in the human body is the",
    "The trans-Saharan trade routes primarily carried gold northward and",
    "The number of federal states in Germany is",
    "The scientist who formulated the uncertainty principle was",
    "The island nation off the southeast coast of Africa is",
    "The Rosetta Stone was inscribed in three scripts: hieroglyphic, demotic, and",
    "The largest sand sea on Earth is the",
    "The first successful powered flight was made by the",
    "The primary language spoken in Brazil is",
    "The Byzantine Empire's capital city was",
    "The unit of magnetic flux density is named after",
    "The mammal capable of true sustained flight is the",
    "The Hanseatic League was a commercial confederation of merchant guilds in",
    "The plant pigment responsible for absorbing red and blue light is",
    "The longest-reigning British monarch was",
    "The Antikythera mechanism is believed to have been used to predict",
    "The element with atomic number 79 is",
    "The historical figure who led the Carthaginian army across the Alps was",
    "The layer of the atmosphere containing most of the ozone is the",
    "The traditional Japanese art of paper folding is called",
    "The mathematician who proved that there are infinitely many primes was",
]

_TECHNICAL = [
    "In a diffusion model, the reverse process is trained to predict",
    "The Chandra X-ray Observatory achieves its angular resolution through",
    "Under the rule against perpetuities, a contingent future interest is void unless",
    "Restriction enzymes cut DNA at specific sites because they recognise",
    "In monetary policy, the transmission mechanism from interest rates to inflation operates through",
    "A superconductor expels magnetic flux from its interior, an effect known as",
    "In linguistics, an ergative-absolutive alignment marks the subject of an intransitive verb",
    "The half-life of a first-order reaction is independent of initial concentration because",
    "In distributed systems, linearizability differs from serializability in that",
    "The blood-brain barrier restricts passage of large molecules because its endothelial cells",
    "In structural engineering, a statically indeterminate structure requires compatibility equations because",
    "The Nernst equation relates electrode potential to ion concentration by",
    "Gödel's second incompleteness theorem states that a consistent formal system cannot",
    "In immunology, clonal selection explains antibody diversity by proposing that",
    "A turbofan engine achieves better fuel efficiency than a turbojet at subsonic speeds because",
    "In probability, a martingale is a stochastic process whose conditional expectation",
    "The greenhouse effect warms a planet because greenhouse gases are transparent to",
    "In compiler design, static single assignment form simplifies optimisation by ensuring",
    "The Michaelis-Menten constant of an enzyme corresponds to the substrate concentration at which",
    "In seismology, S-waves do not propagate through the outer core, which indicates that",
    "Under admiralty law, the doctrine of general average requires that",
    "A phase-locked loop achieves frequency synthesis by comparing",
    "In population genetics, Hardy-Weinberg equilibrium assumes no selection, no mutation, and",
    "The Carnot efficiency of a heat engine depends only on",
    "In cryptography, a commitment scheme must satisfy both hiding and",
    "Osmotic pressure in a solution arises because the solvent's chemical potential is",
    "In control theory, a system is observable if its internal state can be determined from",
    "The Coriolis effect deflects moving air in the northern hemisphere toward",
]

_CODE = [
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi =",
    "SELECT department, AVG(salary) FROM employees WHERE hire_date > '2020-01-01' GROUP BY department ORDER BY",
    "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})\nprint(df.groupby('a').sum().shape)  # output is",
    "func reverse(s string) string {\n    runes := []rune(s)\n    for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {\n        runes[i], runes[j] =",
    "const memoize = (fn) => {\n  const cache = new Map();\n  return (...args) => {\n    const key = JSON.stringify(args);\n    if (cache.has(key)) return",
    "impl<T: Ord> BinaryTree<T> {\n    fn insert(&mut self, value: T) {\n        match self.root {\n            None =>",
    "class LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.cache = OrderedDict()\n\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return",
    "#include <vector>\nstd::vector<int> primes_below(int n) {\n    std::vector<bool> sieve(n, true);\n    for (int i = 2; i * i < n; ++i)\n        if (sieve[i])\n            for (int j = i * i; j < n; j +=",
    "with open('data.csv') as f:\n    reader = csv.DictReader(f)\n    totals = defaultdict(float)\n    for row in reader:\n        totals[row['category']] +=",
    "async function fetchAll(urls) {\n  const results = await Promise.allSettled(urls.map(u => fetch(u)));\n  return results.filter(r => r.status ===",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right =",
    "CREATE INDEX idx_orders_customer ON orders (customer_id, created_at DESC) WHERE status =",
    "let rec map f = function\n  | [] -> []\n  | x :: rest -> f x ::",
    "@dataclass\nclass Point:\n    x: float\n    y: float\n\n    def distance_to(self, other: 'Point') -> float:\n        return math.sqrt((self.x - other.x) ** 2 +",
    "public static int gcd(int a, int b) {\n    while (b != 0) {\n        int temp = b;\n        b = a % b;\n        a =",
    "df.loc[df['value'].isna(), 'value'] = df.groupby('group')['value'].transform(",
    "trap 'rm -f \"$tmpfile\"' EXIT\ntmpfile=$(mktemp)\nwhile IFS= read -r line; do\n    printf '%s\\n' \"${line//old/new}\" >>",
    "def merge_intervals(intervals):\n    intervals.sort(key=lambda x: x[0])\n    merged = []\n    for start, end in intervals:\n        if merged and merged[-1][1] >= start:\n            merged[-1][1] = max(merged[-1][1],",
    "type Result<T> = { ok: true; value: T } | { ok: false; error:",
    "SELECT o.id FROM orders o LEFT JOIN shipments s ON s.order_id = o.id WHERE s.id IS",
    "def rotate_matrix(m):\n    n = len(m)\n    for i in range(n // 2):\n        for j in range(i, n - i - 1):\n            tmp = m[i][j]\n            m[i][j] = m[n-1-j][i]\n            m[n-1-j][i] =",
    "module counter (input clk, input rst, output reg [7:0] q);\n  always @(posedge clk) begin\n    if (rst) q <= 8'b0;\n    else q <=",
    "fn main() {\n    let mut handles = vec![];\n    for i in 0..4 {\n        handles.push(thread::spawn(move || {\n            println!(\"worker {}\", i);\n        }));\n    }\n    for h in handles {\n        h.join().",
    "def levenshtein(a, b):\n    prev = list(range(len(b) + 1))\n    for i, ca in enumerate(a, 1):\n        cur = [i]\n        for j, cb in enumerate(b, 1):\n            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca !=",
    "curl -sS -X POST https://api.example.com/v1/jobs -H 'Content-Type: application/json' -d '{\"name\":\"build\"}' |",
    "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\n\ndef detect_cycle(head):\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow is",
    "SELECT COUNT(DISTINCT user_id) FROM events WHERE event_time >= NOW() - INTERVAL '7 days' AND event_type IN (",
    "def flatten(nested):\n    for item in nested:\n        if isinstance(item, (list, tuple)):\n            yield from flatten(item)\n        else:\n            yield",
]

_LONGCTX_SOURCES = [
    ("Cartography before satellite survey depended on a chain of measurements whose errors compounded in ways "
     "that mapmakers understood but could not always correct. A national survey began with a baseline measured "
     "directly on flat ground, sometimes with metal rods whose thermal expansion had to be modelled, and from "
     "that single line the rest of the country was triangulated outward. Each triangle inherited the error of "
     "the one before it, so a survey that started with a baseline accurate to a few centimetres could drift by "
     "tens of metres a thousand kilometres away. The Great Trigonometrical Survey of India took decades partly "
     "for this reason: the surveyors kept returning to remeasure baselines in new regions, closing loops to "
     "detect accumulated error rather than trusting the chain. What made the work scientifically valuable was "
     "not the maps alone but the discrepancies, because a triangulation that failed to close by more than its "
     "expected error indicated something physical — a deflection of the plumb line caused by the gravitational "
     "pull of nearby mountains. Measuring that deflection let surveyors estimate the density of the Himalayas, "
     "and the answer implied the ranges were less massive than their volume suggested, which in turn suggested ",
     "that the crust beneath them was displacing denser material at depth."),
    ("The economics of lighthouse provision became a standard example in public goods theory, and the example "
     "turns out to be more complicated than the textbooks that use it. A lighthouse appears to be the clearest "
     "possible case of a good that cannot be withheld from those who do not pay: the beam is visible to every "
     "ship within range, and excluding a free rider would require turning it off for everyone. From this it was "
     "argued that lighthouses must be publicly funded, since no private operator could collect. The historical "
     "record complicates that reasoning. For long stretches, English lighthouses were built and operated by "
     "private parties who collected dues at the ports the ships were bound for, which worked because a ship "
     "benefiting from the light almost always docked somewhere it could be charged. The exclusion happened not "
     "at the point of consumption but at a chokepoint downstream, and that is a general pattern worth noticing: "
     "a good that looks non-excludable in isolation may be perfectly excludable once the surrounding institutions "
     "are considered. The disagreement among economists has not been about the facts of collection but about ",
     "whether port dues collected under a royal patent count as private provision at all."),
    ("Antibiotic resistance spreads through mechanisms that make the timescales of evolution much shorter than "
     "vertical inheritance alone would allow. A bacterium that acquires a resistance gene by mutation passes it "
     "to its descendants, which is slow, but bacteria also exchange genetic material horizontally, through "
     "plasmids that move between cells of different species. A plasmid carrying several resistance genes can "
     "confer resistance to several unrelated drugs in a single transfer event, which is why exposure to one "
     "antibiotic can select for resistance to others that the population has never encountered. This coupling "
     "means that resistance is not simply a dose-response phenomenon in a single species but an ecological one "
     "across a community, including the commensal organisms that are not the target of treatment at all. The "
     "reservoir matters as much as the pathogen: a gut flora carrying resistance plasmids constitutes a store "
     "from which a later infection can draw. Efforts to slow resistance by cycling drug classes assume that "
     "resistance carries a fitness cost that will cause it to decline when the drug is withdrawn, and it often ",
     "does, but compensatory mutations frequently restore fitness while retaining resistance."),
    ("Type systems in programming languages sit on a trade-off that has no universally correct resolution, and "
     "the history of the argument is mostly a history of people talking past each other about which errors "
     "matter. A type checker rejects programs, and every checker rejects some programs that would have run "
     "correctly, because the analysis is necessarily an approximation of behaviour that is undecidable in "
     "general. The question is not whether false rejections occur but whether the errors caught are worth the "
     "programs lost and the annotation burden imposed. Empirically the answer depends heavily on what the "
     "program does. In a codebase where most defects are logic errors within a single function, types catch "
     "little, because the mistakes are consistent with the types. In a codebase where most defects arise at "
     "interfaces between components written by different people at different times, types catch a great deal, "
     "because the mistakes are precisely disagreements about shape. This suggests that arguments about static "
     "versus dynamic typing which ignore the structure of the codebase and the team are unlikely to generalise, ",
     "and that the same language can be the right and wrong choice for two projects of equal size."),
    ("Glacial cycles over the past million years show a periodicity that orbital forcing alone does not explain, "
     "and the mismatch is one of the more instructive open problems in palaeoclimate. The Milankovitch theory "
     "attributes ice ages to variations in Earth's orbit and axial tilt, which change the distribution of "
     "sunlight by latitude and season. Those variations have known periods, and the dominant one in the "
     "insolation record is roughly forty-one thousand years. For the early Pleistocene the ice record matches "
     "that period well. Then, around a million years ago, the ice ages shifted to a cycle closer to a hundred "
     "thousand years, which corresponds to a much weaker orbital term. The forcing did not change; the response "
     "did. Explanations invoke internal feedbacks that accumulate over multiple orbital cycles — the slow "
     "erosion of a regolith layer beneath the ice sheets, changes in the depth of ocean carbon storage, or "
     "nonlinear ice-sheet dynamics that require several weak forcings in sequence before collapsing. What the ",
     "transition establishes is that the climate system's response can change character while its driver does not."),
    ("Standardised time is an artefact of railways, and the resistance it met is a useful reminder that "
     "infrastructure imposes conventions rather than discovering them. Before scheduled rail travel, towns kept "
     "local solar time, so noon in one town differed from noon in another fifty miles away by a few minutes. "
     "This caused nobody any difficulty, because nothing moved fast enough for the difference to matter. Rail "
     "timetables made it matter immediately: a schedule published in one town's time was wrong in every other "
     "town, and collisions on single-track lines were a direct consequence of clocks that disagreed. Railway "
     "companies therefore adopted their own uniform time, initially as a private convention printed on their "
     "own timetables, and for some decades a town might have two times — the clock on the church and the clock "
     "at the station, deliberately set differently and both correct. Legal standardisation followed commercial "
     "practice by a long interval, and in some jurisdictions the courts continued to interpret contracts by "
     "local solar time well after the railways had moved on, which produced litigation about when exactly a ",
     "deadline expired for a document delivered by train."),
    ("Peer review acquired its present authority recently enough that its history is documented, and the "
     "documentation does not support the belief that it has always been the mechanism by which science "
     "validates claims. For much of the period in which the foundational results of physics and biology were "
     "published, editors decided what to print, sometimes consulting a colleague and sometimes not, and the "
     "filtering that mattered happened after publication rather than before it — through replication, citation, "
     "and use. Systematic external refereeing became standard in the middle of the twentieth century, driven "
     "partly by the volume of submissions and partly by the need to allocate scarce journal pages. That origin "
     "explains some of its properties: it is good at catching obvious error and poor at catching fraud, because "
     "a referee reads a manuscript rather than a laboratory. It also explains why fields differ so much in what "
     "they expect of it. The question worth asking is not whether peer review works in the abstract but which ",
     "specific failure it is supposed to prevent, and whether it is the cheapest way of preventing that one."),
    ("Compression and prediction are the same problem viewed from different ends, and recognising the identity "
     "clarifies several arguments that otherwise seem to be about different things. A compressor assigns short "
     "codes to likely symbols and long codes to unlikely ones, so to compress well it must know the "
     "probability distribution over what comes next; a predictor that knows that distribution can be turned "
     "into a compressor mechanically, and a good compressor can be turned into a predictor by asking what "
     "continuation it would encode most cheaply. The number of bits a model needs to encode a text is "
     "therefore a direct measure of how well the model predicts that text, which is why compression benchmarks "
     "and language-model evaluations report quantities that are the same measurement in different units. This "
     "identity also bounds what compression can achieve: no compressor can do better on average than the "
     "entropy of the source, and any compressor that beats that bound on some inputs must do worse on others, ",
     "because the code lengths must sum over all possible messages in a way that admits no free improvement."),
]

_LONGCTX = [src + " " + tail for src, tail in _LONGCTX_SOURCES]


def _build() -> list[dict]:
    out: list[dict] = []
    for stratum, texts in (("general", _GENERAL), ("technical", _TECHNICAL),
                           ("code", _CODE), ("longctx", _LONGCTX)):
        for i, t in enumerate(texts):
            out.append({"id": f"{stratum}-h{i:03d}", "stratum": stratum, "text": t})
    return out


PROMPTS: list[dict] = _build()


def digest() -> str:
    """sha256 over (id, text) in order. Order-sensitive by design."""
    h = hashlib.sha256()
    for p in PROMPTS:
        h.update(p["id"].encode())
        h.update(b"\x1f")
        h.update(p["text"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def strata_counts() -> dict[str, int]:
    c: dict[str, int] = {}
    for p in PROMPTS:
        c[p["stratum"]] = c.get(p["stratum"], 0) + 1
    return c


def assert_disjoint_from_committed() -> int:
    """No prompt text is shared with `kl_prompts.py`. Returns the number checked.

    The mechanical half of disjointness. The other half — that no fact, snippet or passage
    topic was reused — is a property of how these were written and cannot be asserted here.
    """
    from kl_prompts import PROMPTS as COMMITTED

    shared = {p["text"] for p in PROMPTS} & {p["text"] for p in COMMITTED}
    if shared:
        raise AssertionError(f"{len(shared)} prompt(s) shared with kl_prompts.py: {sorted(shared)[:2]}")
    return len(PROMPTS)
