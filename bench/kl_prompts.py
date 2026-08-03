"""kl_prompts.py — the committed prompt set for KL fidelity measurement (K1).

COMMITTED BEFORE ANY MEASUREMENT. KL is prompt-distribution dependent: a prompt set
chosen after seeing results is a fitted metric. The sha256 below is recorded in every
receipt; if it changes, past numbers are not comparable and the receipt will show it.

Composition (200 prompts, 4 strata):
  general   (60) — factual/世界 knowledge across eras, regions, disciplines. Includes a
                   deliberate spread from common to obscure, because the motivating
                   result (Quesma 2026-08-03) reports that the MOST OBSCURE facts
                   degrade first under quantization. A prompt set of only common facts
                   would under-report damage.
  technical (55) — domain prose: physics, biology, law, medicine, finance, linguistics.
  code      (55) — code completion/reasoning across languages and paradigms.
  longctx   (30) — multi-hundred-token passages, to exercise attention over distance
                   where KV/attention quantization damage would surface.

Every prompt is scored TEACHER-FORCED — the text below is the full scored sequence, not
a generation seed. Token counts are tokenizer-dependent and therefore recorded at
measurement time (in the receipt), not here.

CONSEQUENCE OF TOKEN-WEIGHTING (state this whenever the headline number is quoted):
aggregation is token-weighted, and the long-context stratum is ~58% of all scored tokens
despite being 9% of the prompts (18 of 200). The headline mean is therefore dominated by
long-context behaviour. That is a deliberate choice — long contexts are where attention
and KV-cache damage surfaces, and a prompt-weighted mean would let 24-character factual
stubs outvote them — but it means K2 MUST also report per-stratum KL. A single pooled
number would hide a format that is fine on short prompts and degrades over distance, or
the reverse. Approximate token shares: longctx 58%, code 18%, technical 12%, general 12%.
"""
from __future__ import annotations

import hashlib

_GENERAL = [
    "The capital of Australia is",
    "Water boils at 100 degrees Celsius at a pressure of",
    "The author of One Hundred Years of Solitude is",
    "The chemical symbol for tungsten is",
    "The Battle of Hastings took place in the year",
    "The largest moon of Saturn is called",
    "In Greek mythology, the goddess of wisdom is",
    "The currency of South Korea is the",
    "The longest river in South America is the",
    "Photosynthesis converts carbon dioxide and water into glucose and",
    "The Treaty of Westphalia was signed in",
    "The inventor of the mercury thermometer was",
    "The smallest prime number greater than 100 is",
    "The city of Timbuktu is located in the modern country of",
    "The composer of the opera Der Rosenkavalier was",
    "The deepest point in the ocean is called the",
    "The first woman to win a Nobel Prize was",
    "The language with the most native speakers worldwide is",
    "The bone in the human body that is longest is the",
    "The Rosetta Stone was discovered in the year",
    "The founder of the Maurya Empire was",
    "A group of crows is collectively known as a",
    "The half-life of carbon-14 is approximately",
    "The painter of The Garden of Earthly Delights was",
    "The tallest mountain in Africa is",
    "The元素 with atomic number 79 is",
    "The Antikythera mechanism is believed to have been used to",
    "The capital of Kazakhstan was renamed in 2019 to",
    "The novel Things Fall Apart was written by",
    "The unit of electrical resistance is named after",
    "The Silk Road primarily connected China with",
    "The largest desert in the world by area is the",
    "The first successful heart transplant was performed by",
    "The mathematician who proved Fermat's Last Theorem was",
    "The volcanic eruption that buried Pompeii occurred in",
    "The national animal of Scotland is the",
    "The Voynich manuscript is written in a script that remains",
    "The blue color of the sky is caused by",
    "The founder of the Ottoman Empire was",
    "The number of bones in the adult human body is",
    "The most abundant gas in Earth's atmosphere is",
    "The Dead Sea Scrolls were discovered near",
    "The physicist who formulated the uncertainty principle was",
    "The capital of Bhutan is",
    "The process by which a liquid turns directly into gas below its boiling point is",
    "The oldest continuously inhabited city in the world is often said to be",
    "The Peloponnesian War was fought between Athens and",
    "The organ that produces insulin is the",
    "The architect of the Sagrada Família was",
    "The last Tsar of Russia was",
    "The three primary colors of light are red, green, and",
    "The Magna Carta was sealed at",
    "The theory of continental drift was proposed by",
    "A tsunami is most commonly caused by",
    "The poet who wrote The Waste Land was",
    "The country with the most time zones is",
    "The enzyme that unwinds DNA during replication is called",
    "The Sykes-Picot Agreement divided territory between Britain and",
    "The heaviest naturally occurring element is",
    "The philosopher who wrote Critique of Pure Reason was",
    # deliberately obscure tail — the motivating result reports that the least common
    # facts degrade first, so a set of only well-known items would under-report damage
    "The Kingdom of Dahomey maintained an all-female military regiment known as the",
    "The Tunguska event of 1908 is most widely attributed to",
    "The mathematician Emmy Noether is best known for a theorem connecting symmetries to",
    "The Nabataean capital carved into rose-colored rock is",
    "The Ediacaran biota predates the Cambrian explosion by approximately",
    "The Sogdian merchants of Central Asia primarily traded along the",
    "The medieval Islamic scholar Ibn al-Haytham is considered a pioneer of",
    "The last speaker of the Eyak language of Alaska died in",
    "The Antonine Plague is believed by modern historians to have been",
    "The Rhind Mathematical Papyrus contains problems concerning",
    "The bristlecone pine named Methuselah is notable for being",
    "The Younger Dryas was a sudden return to glacial conditions caused most likely by",
]

_TECHNICAL = [
    "In a transformer architecture, the purpose of multi-head attention is to",
    "The Chandrasekhar limit describes the maximum mass of",
    "Under the doctrine of promissory estoppel, a promise becomes enforceable when",
    "The CRISPR-Cas9 system achieves targeted cleavage by",
    "In options pricing, the Greek letter vega measures sensitivity to",
    "A phoneme differs from an allophone in that",
    "The second law of thermodynamics states that the entropy of an isolated system",
    "In renal physiology, the loop of Henle establishes a countercurrent gradient by",
    "The Nyquist-Shannon sampling theorem requires a sampling rate of at least",
    "In immunology, class switching allows a B cell to change",
    "The bid-ask spread widens during periods of",
    "Bayes' theorem relates the posterior probability to the prior by",
    "In seismology, S-waves cannot propagate through liquids because",
    "The Michaelis-Menten constant Km represents the substrate concentration at which",
    "Under IFRS 16, operating leases are recognized on the balance sheet as",
    "The Casimir effect arises from vacuum fluctuations between",
    "In compiler design, static single assignment form simplifies optimization by",
    "The Hardy-Weinberg equilibrium is violated when",
    "In pharmacokinetics, the volume of distribution relates dose to",
    "A martingale is a stochastic process in which the conditional expectation",
    "The Cerenkov radiation is emitted when a charged particle",
    "In linguistics, ergative-absolutive alignment differs from nominative-accusative by",
    "The Higgs mechanism gives mass to gauge bosons through",
    "In distributed systems, the CAP theorem states that one cannot simultaneously guarantee",
    "The Krebs cycle produces ATP indirectly by generating",
    "Under the business judgment rule, courts defer to directors unless",
    "The Doppler effect causes the observed frequency to increase when",
    "In cryptography, a nonce must never be reused because",
    "The blood-brain barrier restricts passage of molecules based primarily on",
    "In fluid dynamics, the Reynolds number predicts the transition to",
    "Quantitative easing affects long-term interest rates primarily through",
    "The Pauli exclusion principle prohibits two fermions from",
    "In epidemiology, the basic reproduction number R0 represents",
    "A p-value does not represent the probability that",
    "In materials science, work hardening increases strength by",
    "The Coriolis effect deflects moving air to the right in",
    "Gene expression is regulated post-transcriptionally by mechanisms including",
    "In control theory, a system is BIBO stable if",
    "The photoelectric effect demonstrated that light behaves as",
    "Under admiralty law, general average requires that",
    "In statistics, heteroskedasticity violates the assumption that",
    "The mitochondrial electron transport chain generates a proton gradient across",
    "A zero-knowledge proof allows a prover to demonstrate knowledge without",
    "The Lorentz factor approaches infinity as velocity approaches",
    "In organic chemistry, an SN2 reaction proceeds with inversion of",
    "The efficient market hypothesis in its semi-strong form asserts that prices reflect",
    "In neural networks, batch normalization reduces internal covariate shift by",
    "The greenhouse effect operates because certain gases are transparent to",
    "In population genetics, genetic drift has the strongest effect in",
    "The Nernst equation relates membrane potential to",
    "In machine learning, regularization prevents overfitting by",
    "The wave function collapse in quantum mechanics occurs upon",
    "Antibiotic resistance spreads horizontally through mechanisms such as",
    "In signal processing, a Butterworth filter is characterized by",
    "The Gini coefficient measures inequality on a scale where zero represents",
]

_CODE = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "SELECT customer_id, COUNT(*) AS order_count FROM orders GROUP BY customer_id HAVING",
    "import numpy as np\narr = np.array([[1, 2], [3, 4]])\nprint(arr.T @ arr)  # output is",
    "func reverse(s string) string {\n    runes := []rune(s)\n    for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def pop(self):\n        if not self.items:\n            raise",
    "const memoize = (fn) => {\n  const cache = new Map();\n  return (...args) => {\n    const key = JSON.stringify(args);",
    "impl<T: Ord> BinaryTree<T> {\n    fn insert(&mut self, value: T) {\n        match self.root {",
    "#include <stdio.h>\nint main() {\n    int *p = malloc(sizeof(int) * 10);\n    if (p == NULL) {",
    "async function fetchWithRetry(url, retries = 3) {\n  try {\n    const res = await fetch(url);",
    "public static <T extends Comparable<T>> T max(List<T> list) {\n    if (list.isEmpty()) throw new",
    "SELECT e.name, d.dept_name FROM employees e LEFT JOIN departments d ON e.dept_id = d.id WHERE d.id IS",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]",
    "type Result<T, E> = Ok(T) | Err(E);\nfn divide(a: f64, b: f64) -> Result<f64, String> {\n    if b == 0.0 {",
    "git rebase -i HEAD~3 opens an editor in which the command to combine a commit with the previous one is",
    "docker run -d --name web -p 8080:80 nginx  # the -d flag causes the container to",
    "with open('data.csv') as f:\n    reader = csv.DictReader(f)\n    total = sum(float(row['amount']) for row in reader if",
    "const debounce = (fn, ms) => {\n  let timer;\n  return (...args) => {\n    clearTimeout(timer);",
    "@dataclass(frozen=True)\nclass Point:\n    x: float\n    y: float\n    def __add__(self, other):",
    "for i in range(len(matrix)):\n    for j in range(i + 1, len(matrix)):\n        matrix[i][j], matrix[j][i] =",
    "SELECT DATE_TRUNC('month', created_at) AS month, SUM(revenue) FROM sales WHERE created_at >= NOW() - INTERVAL",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2",
    "useEffect(() => {\n  const sub = source.subscribe(setValue);\n  return () => {",
    "package main\nimport \"sync\"\nfunc main() {\n    var wg sync.WaitGroup\n    for i := 0; i < 5; i++ {\n        wg.Add(1)",
    "CREATE INDEX CONCURRENTLY idx_users_email ON users(email);  -- CONCURRENTLY avoids",
    "try:\n    value = int(user_input)\nexcept ValueError:\n    logger.warning('bad input: %s', user_input)",
    "let sum = numbers.iter().filter(|&&x| x % 2 == 0).map(|&x| x * x).sum::<i32>();  // this computes",
    "def merge_sorted(a, b):\n    out, i, j = [], 0, 0\n    while i < len(a) and j < len(b):",
    "kubectl get pods -n production -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\\n' | grep",
    "class LRUCache:\n    def __init__(self, capacity):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n    def get(self, key):",
    "SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.fk = t1.id) -- semantically equivalent to",
    "std::unique_ptr<Node> head = std::make_unique<Node>();  // ownership is",
    "def flatten(nested):\n    for item in nested:\n        if isinstance(item, (list, tuple)):\n            yield from",
    "app.use((err, req, res, next) => {\n  console.error(err.stack);\n  res.status(500).json({",
    "awk -F',' '{sum += $3} END {print sum/NR}' data.csv  # this prints the",
    "def transpose(m):\n    return [list(row) for row in zip(*m)]  # for a 2x3 input the output shape is",
    "ALTER TABLE orders ADD CONSTRAINT fk_customer FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE",
    "match value {\n    Some(x) if x > 10 => println!(\"big\"),\n    Some(x) => println!(\"small\"),\n    None =>",
    "const worker = new Worker('worker.js');\nworker.postMessage(data);  // this runs the script on",
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return",
    "SELECT rank() OVER (PARTITION BY dept ORDER BY salary DESC) FROM employees  -- rank differs from dense_rank in",
    "if __name__ == '__main__':\n    multiprocessing.set_start_method('spawn')  # required on macOS because",
    "template<typename... Args>\nvoid log(Args&&... args) {\n    (std::cout << ... << args)",
    "curl -sS -X POST -H 'Content-Type: application/json' -d '{\"k\":1}' http://api/v1/items | jq",
    "def is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned ==",
    "pub fn spawn_worker(rx: Receiver<Job>) -> JoinHandle<()> {\n    thread::spawn(move || {\n        while let Ok(job) = rx.recv() {",
    "SELECT COALESCE(nickname, first_name, 'anonymous') FROM users  -- COALESCE returns",
    "np.einsum('ij,jk->ik', A, B)  # this is equivalent to the operation",
    "def retry(times):\n    def decorator(fn):\n        @functools.wraps(fn)\n        def wrapper(*a, **kw):",
    "docker build --no-cache -t app:latest .  # --no-cache forces Docker to",
    "let cancelled = false;\nconst promise = new Promise((resolve, reject) => {\n  setTimeout(() => cancelled ? reject() :",
    "def chunked(it, size):\n    it = iter(it)\n    while chunk := list(islice(it, size)):",
    "SELECT a.id FROM a INNER JOIN b USING (id) -- USING differs from ON in that",
    "@pytest.mark.parametrize('inp,expected', [(1, 2), (2, 4)])\ndef test_double(inp, expected):",
    "os.environ.setdefault('TZ', 'UTC')\ntime.tzset()  # tzset is required because",
    "func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {\n    ctx, cancel := context.WithTimeout(r.Context(),",
]

_LONGCTX_SOURCES = [
    ("The transition from bronze to iron metallurgy did not occur simultaneously across the "
     "ancient world, and the reasons are more economic than technical. Bronze requires tin, "
     "and tin deposits are geographically rare and concentrated, which meant that bronze "
     "production depended on long-distance trade networks vulnerable to political disruption. "
     "Iron ore, by contrast, is abundant almost everywhere, but smelting it requires "
     "sustained higher temperatures and a reducing atmosphere that early furnaces struggled "
     "to maintain. The result is that iron was known long before it was economical. When the "
     "eastern Mediterranean trade networks collapsed near the end of the second millennium "
     "BCE, the tin supply became unreliable, and communities that had previously treated iron "
     "as a curiosity had strong reason to invest in the harder metallurgical problem. "
     "Archaeological evidence supports this: early iron objects appear as prestige goods "
     "centuries before iron tools appear in agricultural contexts. The adoption curve "
     "therefore tracks the collapse of a supply chain rather than a metallurgical "
     "breakthrough, which is why the",
     "conclusion about causation is"),
    ("Memory hierarchies exist because there is no single storage technology that is "
     "simultaneously fast, large, and cheap. Registers are the fastest storage available to a "
     "processor, but a core has only a few hundred of them. Static RAM used for cache is "
     "roughly an order of magnitude slower and considerably more expensive per bit, so caches "
     "are measured in megabytes rather than gigabytes. Dynamic RAM trades further latency for "
     "density, and persistent storage trades far more latency for capacity and durability. "
     "The performance of real programs therefore depends less on raw processor speed than on "
     "whether the working set fits in a given level. This is why an algorithm with worse "
     "asymptotic complexity can outperform a better one on real hardware: a linear scan over "
     "contiguous memory issues predictable prefetches, while a pointer-chasing structure with "
     "superior complexity may stall on every dereference. The practical consequence for "
     "engineers is that data layout frequently matters more than",
     "algorithmic choice because"),
    ("Central banks face a credibility problem that is fundamentally about time consistency. "
     "A bank that promises low inflation and is believed will find, once expectations are "
     "set, that a surprise expansion delivers real output gains at little immediate cost. "
     "Rational agents anticipate this incentive, so the promise alone is not believed, and "
     "the economy settles at higher inflation with no output benefit whatsoever. The "
     "institutional response has been to remove discretion: independent central banks, "
     "explicit inflation targets, and published reaction functions all serve to make the "
     "promise costly to break rather than merely stated. The effectiveness of these "
     "arrangements is difficult to test because the counterfactual is unobservable, and "
     "because periods of central bank independence have coincided with favorable supply "
     "conditions such as globalization and demographic expansion. Disentangling institutional "
     "credibility from",
     "favorable circumstance requires"),
    ("Protein folding is often described as a search problem, and framed that way it appears "
     "impossible: a modest polypeptide has more conformations than there are atoms in the "
     "observable universe, and random search would take longer than the age of the cosmos. "
     "This is Levinthal's paradox, and its resolution is that folding is not a search over a "
     "flat landscape but a descent down a funnel. Local interactions form quickly and are "
     "energetically favorable, constraining the accessible conformational space enormously "
     "before the global structure is determined. Secondary structures nucleate, hydrophobic "
     "residues collapse inward away from solvent, and the resulting molten globule explores a "
     "vastly reduced space. The funnel metaphor also explains misfolding: a landscape with "
     "kinetic traps allows a chain to reach a local minimum that is stable enough to persist, "
     "which is the mechanism underlying amyloid diseases. The therapeutic implication is that "
     "targeting the folding pathway may be more tractable than",
     "targeting the aggregate because"),
    ("The doctrine of adverse possession seems, on first encounter, to reward wrongdoing: a "
     "trespasser who occupies land openly, continuously, and without permission for a "
     "statutory period may acquire title against the true owner. The justification is not "
     "moral but administrative. Titles decay: records are lost, boundaries are described "
     "relative to landmarks that no longer exist, and heirs disperse without ever learning "
     "what they inherited. A rule that lets long-settled possession ripen into ownership "
     "converts an unresolvable historical question into an answerable present one, and it "
     "penalizes the owner who neglected the land rather than the occupant who improved it. "
     "The requirements are calibrated to this purpose. Possession must be open and notorious, "
     "so that a diligent owner would have noticed; hostile, so that permissive use does not "
     "count; and continuous, so that sporadic trespass does not accumulate. Each element "
     "exists to ensure the doctrine only extinguishes claims that the owner had a fair "
     "opportunity to",
     "assert, which is why"),
    ("Distributed consensus protocols must handle a failure mode that single-machine systems "
     "never confront: a node that is merely slow is indistinguishable from a node that has "
     "died. This ambiguity is not an engineering deficiency but a theoretical result — in an "
     "asynchronous network with even one faulty process, no deterministic protocol can "
     "guarantee both safety and liveness. Practical systems escape the impossibility by "
     "weakening an assumption. Some assume partial synchrony, where message delays are "
     "bounded after some unknown time, which permits leader election with timeouts. Others "
     "accept probabilistic guarantees, converging with probability approaching one rather "
     "than certainty. Randomized protocols escape by removing determinism entirely. What no "
     "system can do is satisfy the original conditions, which is why every production "
     "consensus implementation embeds a timeout somewhere, and why tuning those timeouts is "
     "an operational rather than",
     "theoretical activity, since"),
]

_LONGCTX_SOURCES += [
    ("Coral bleaching is frequently described as coral death, which is inaccurate and "
     "obscures the mechanism. Reef-building corals host symbiotic dinoflagellates that "
     "supply most of the animal's carbon through photosynthesis. Under thermal stress the "
     "symbionts' photosystems produce reactive oxygen faster than they can be quenched, and "
     "the host expels them — a response that is plausibly adaptive in the short term, since "
     "retaining a symbiont that is generating oxidative damage is worse than starving "
     "briefly. The coral is then white because the animal tissue is transparent and the "
     "skeleton beneath is calcium carbonate. A bleached coral can recover if temperatures "
     "fall within weeks and if symbionts are available for reacquisition. Mortality follows "
     "from prolonged starvation, not from the expulsion itself, which is why the duration of "
     "a thermal anomaly matters more than",
     "its peak magnitude, since"),
    ("The reason aircraft fly at altitudes near eleven kilometres is a compromise between two "
     "opposing curves. Air density falls with altitude, which reduces drag and therefore fuel "
     "burn for a given true airspeed — an argument for climbing as high as possible. But "
     "engine thrust also falls with density, and the lift required to stay aloft does not, so "
     "the aircraft must fly faster to generate it. As the true airspeed rises toward the "
     "speed of sound, wave drag climbs sharply. The optimum sits where the marginal saving "
     "from thinner air equals the marginal penalty from approaching the transonic regime, and "
     "it shifts upward through a flight as fuel burns off and the aircraft becomes lighter. "
     "This is why long-haul flights request step climbs rather than a single cruise altitude, "
     "and why the optimum for a given airframe depends on",
     "weight rather than distance, because"),
    ("Sourdough fermentation is a competition that the baker rigs rather than controls. A "
     "flour-and-water mixture initially hosts a broad microbial population, but the "
     "conditions of repeated feeding select ruthlessly: lactic acid bacteria acidify the "
     "medium, and the resulting low pH excludes most competitors while remaining tolerable "
     "for acid-adapted yeasts. The stable end state is a two-organism consortium in which the "
     "bacteria produce acids and the yeast produces carbon dioxide, each tolerating "
     "conditions the other creates. This explains why mature starters are robust against "
     "contamination while young ones spoil easily, and why a starter maintained at a "
     "different temperature or hydration develops a measurably different flavor profile: the "
     "selection pressure has changed, so the",
     "surviving consortium differs, meaning"),
    ("Insurance markets fail in a specific and predictable way when the buyer knows more about "
     "their own risk than the seller does. If a policy is priced at the population average, "
     "it is a good deal for high-risk buyers and a poor one for low-risk buyers, so low-risk "
     "buyers disproportionately decline. The remaining pool is riskier than assumed, claims "
     "exceed premiums, and the price must rise — which drives out the next tier of relatively "
     "low-risk buyers, and so on. The equilibrium can be a market that serves almost nobody, "
     "even though mutually beneficial trades existed at the outset. Every practical remedy "
     "attacks the information asymmetry or removes the choice: mandatory participation, "
     "underwriting and medical examination, group policies tied to employment, or waiting "
     "periods that penalize buying only when a claim is anticipated. Each of these is best "
     "understood not as a market restriction but as",
     "a repair of the selection problem, since"),
    ("Radiocarbon dates are not calendar dates, and treating them as such is the most common "
     "error in reading archaeological literature. The method measures residual carbon-14, "
     "which requires assuming the atmospheric concentration at the time of death. That "
     "concentration was not constant: it varies with solar activity and geomagnetic field "
     "strength, and it was disturbed dramatically by fossil fuel combustion and by "
     "atmospheric nuclear testing. Calibration curves built from tree rings and other "
     "independently dated material convert measured radiocarbon years into calendar ranges, "
     "and because the curve is not monotonic, a single measurement can map onto several "
     "disjoint calendar intervals. Published dates therefore carry both a laboratory "
     "uncertainty and a calibration structure, and a date quoted without indicating whether "
     "it is calibrated is",
     "effectively unusable, because"),
    ("Noise-cancelling headphones work by destructive interference, which imposes a hard "
     "physical constraint that marketing rarely mentions. A microphone samples the incoming "
     "wave, the electronics invert it, and the speaker emits the inverse so the sum "
     "approaches zero at the eardrum. The scheme requires the correction to arrive in phase, "
     "and the available time is the propagation delay from microphone to ear — a few hundred "
     "microseconds. For low-frequency sound, whose wavelength is long relative to that "
     "distance, this is comfortably achievable, and cancellation is excellent. For high "
     "frequencies the wavelength approaches the geometry of the earcup and head, phase varies "
     "across the ear, and no single correction can null the field everywhere. This is why "
     "active cancellation excels against engine rumble and fails against speech, and why "
     "manufacturers address the high end with",
     "passive isolation instead, given that"),
    ("Antarctic sea ice behaves differently from Arctic sea ice in ways that make simple "
     "comparisons misleading. Arctic ice sits in an ocean nearly enclosed by land, so much of "
     "it survives multiple summers and thickens; its decline is therefore visible in both "
     "extent and age. Antarctic ice forms around a continent surrounded by open ocean with a "
     "circumpolar current, so it is overwhelmingly first-year ice that melts almost entirely "
     "each summer regardless of trend. Its extent is governed strongly by winds and ocean "
     "circulation, which can push ice outward and increase extent even while ocean warming "
     "continues. A period of stable or rising Antarctic extent is therefore not evidence "
     "against warming, and treating the two poles as symmetrical measurements of the same "
     "quantity",
     "misreads the physics, because"),
    ("Double-entry bookkeeping is often taught as an arithmetic convention, but its value is "
     "as an error-detecting code. Every transaction is recorded twice, as a debit and a "
     "matching credit, so the books contain redundancy that a single-entry system lacks. A "
     "transposition, an omission, or a duplicated posting breaks the accounting identity and "
     "the imbalance is detectable without reference to any external record. The system does "
     "not detect all errors — a transaction posted twice in full, or posted to the wrong "
     "account of the correct type, leaves the books balanced — which is precisely why "
     "reconciliation against bank statements and physical inventory remains necessary. "
     "Understood this way, the ledger is a checksum whose failure modes are known, and the "
     "controls layered on top exist to catch",
     "the errors the checksum cannot, namely"),
    ("The placebo effect is routinely described as belief curing illness, which overstates it "
     "in one direction and understates it in another. It does not shrink tumours or clear "
     "infections; objective disease measures are largely unmoved. What it reliably changes "
     "are subjective and centrally mediated outcomes — pain, nausea, fatigue — and it does so "
     "through identifiable physiology, including endogenous opioid release that can be "
     "blocked pharmacologically. That last detail is what elevates it above suggestion: the "
     "effect has a mechanism that can be interrupted. Its magnitude depends on ritual, "
     "expectation, and the practitioner relationship, which is why open-label placebos retain "
     "some effect and why trial design must control for the act of treatment rather than "
     "merely for",
     "the substance administered, since"),
    ("Language death rarely happens because the last speaker dies; it happens a generation "
     "earlier, when parents stop transmitting the language to children. A language with ten "
     "thousand fluent adult speakers and no child speakers is functionally extinct on a "
     "predictable timetable, while a language with two thousand speakers who raise children "
     "in it is stable. This is why speaker counts are a poor health metric and "
     "intergenerational transmission is the standard one. The mechanism of the break is "
     "usually economic rather than coercive: a dominant language controls access to schooling "
     "and employment, and parents rationally optimize for their children's prospects. "
     "Revitalization efforts that target adults while ignoring the domains where children "
     "actually acquire language therefore tend to",
     "produce learners but not speakers, because"),
    ("Superconductivity was discovered before there was any theory capable of explaining it, "
     "and the forty-six year gap is instructive. Resistance vanishing entirely at low "
     "temperature had no place in classical transport theory, and early attempts to treat it "
     "as merely very good conduction failed against the Meissner effect: a superconductor "
     "expels magnetic field from its interior, which is a thermodynamic property of a distinct "
     "phase, not an extreme of ordinary conduction. The eventual explanation required a "
     "mechanism by which electrons, which repel one another, could nonetheless pair — "
     "mediated by lattice vibrations, so that the effective interaction becomes attractive at "
     "the relevant energies. That the pairing is indirect and lattice-mediated is why the "
     "theory predicts an isotope effect, and why observing that effect was regarded as",
     "decisive confirmation rather than mere consistency, since"),
    ("Urban heat islands are not caused primarily by waste heat from air conditioning and "
     "engines, though that contributes. The dominant terms are surface properties: dark "
     "asphalt and roofing absorb more shortwave radiation than vegetation, masonry stores "
     "that heat and releases it through the night, and the removal of plants eliminates "
     "evaporative cooling that would otherwise convert incoming energy into latent rather "
     "than sensible heat. Street canyons compound the effect by trapping longwave radiation "
     "through multiple reflections and by reducing sky view, which is how a city can be "
     "several degrees warmer than its surroundings most strongly at night rather than at "
     "midday. Mitigation therefore targets albedo and vegetation rather than energy use, and "
     "the counterintuitive nighttime peak is the signature that distinguishes",
     "storage-driven warming from direct emission, meaning"),
]

_LONGCTX = [src + " " + tail for src, tail in _LONGCTX_SOURCES]


def _build() -> list[dict]:
    out: list[dict] = []
    for stratum, texts in (("general", _GENERAL), ("technical", _TECHNICAL),
                           ("code", _CODE), ("longctx", _LONGCTX)):
        for i, t in enumerate(texts):
            out.append({"id": f"{stratum}-{i:03d}", "stratum": stratum, "text": t})
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


if __name__ == "__main__":
    print(f"prompts: {len(PROMPTS)}")
    print(f"strata:  {strata_counts()}")
    print(f"chars:   min={min(len(p['text']) for p in PROMPTS)} "
          f"max={max(len(p['text']) for p in PROMPTS)} "
          f"mean={sum(len(p['text']) for p in PROMPTS) // len(PROMPTS)}")
    print(f"sha256:  {digest()}")
