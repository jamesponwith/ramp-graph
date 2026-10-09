"""LadybugDB mirror of ramp-graph's insert and query benches (see docs/spec.md, "Against
LadybugDB"). Same datasets as benches/insert.rs and benches/query.rs; one node table keyed
by the unique value string, the one property per node as a column, one rel table.

    uv venv .venv && uv pip install --python .venv/bin/python real_ladybug pyarrow
    .venv/bin/python -I -u scripts/bench-ladybugdb.py [N=1000000] [all|copy|create|query]

`create` (per-object CREATE in one txn) scales superlinearly on edges; run it at <= 20000.
LB_THREADS=1 pins COPY to one thread for a thread-for-thread load comparison.
Results print counts next to timings so they can be checked against the Rust benches."""
import os, shutil, sys, time, tempfile
import pyarrow as pa, pyarrow.parquet as pq
import real_ladybug as lb

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
MODE = sys.argv[2] if len(sys.argv) > 2 else "all"
WORK = tempfile.mkdtemp(prefix="work_", dir=os.path.dirname(os.path.abspath(__file__)))

def splitmix(state):
    while True:
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        yield z ^ (z >> 31)

def nodes_table():
    cols = {"value": [str(x) for x in range(N)], "type": [f"node{x % 5}" for x in range(N)]}
    for k in range(5):
        cols[f"prop{k}"] = [f"value{k}" if x % 5 == k else None for x in range(N)]
    return pa.table(cols)

def insert_edges():  # insert bench: N unique random pairs, type edge{(x+y)%5}, value i
    rng = splitmix(42); pairs = set()
    while len(pairs) < N:
        pairs.add((next(rng) % N, next(rng) % N))
    pairs = sorted(pairs)  # BTreeSet order, as the Rust bench iterates it
    return pa.table({"src": [str(x) for x, _ in pairs], "dst": [str(y) for _, y in pairs],
                     "type": [f"edge{(x + y) % 5}" for x, y in pairs], "value": [str(i) for i in range(N)]})

def query_edges():  # query bench: x -> (x*7919+13)%N, type edge{y%5}, value x
    ys = [(x * 7919 + 13) % N for x in range(N)]
    return pa.table({"src": [str(x) for x in range(N)], "dst": [str(y) for y in ys],
                     "type": [f"edge{y % 5}" for y in ys], "value": [str(x) for x in range(N)]})

def fresh(name):
    path = os.path.join(WORK, name)
    db = lb.Database(path); c = lb.Connection(db)
    c.execute("CREATE NODE TABLE N(value STRING, type STRING, prop0 STRING, prop1 STRING, prop2 STRING, prop3 STRING, prop4 STRING, PRIMARY KEY(value))")
    c.execute("CREATE REL TABLE E(FROM N TO N, type STRING, value STRING)")
    return db, c, path

def dir_size(p):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs) if os.path.isdir(p) else os.path.getsize(p)

def timed(label, f):
    t0 = time.perf_counter(); r = f(); dt = time.perf_counter() - t0
    print(f"{label:<42} {dt:8.3f}s  {r}", flush=True); return dt

def load_copy(c, nodes, edges):
    if os.environ.get("LB_THREADS"):
        c.set_max_threads_for_exec(int(os.environ["LB_THREADS"]))
    pq.write_table(nodes, f"{WORK}/nodes.parquet"); pq.write_table(edges, f"{WORK}/edges.parquet")
    tn = timed("COPY nodes (+props as columns)", lambda: (c.execute(f"COPY N FROM '{WORK}/nodes.parquet'"), "")[1])
    te = timed("COPY edges", lambda: (c.execute(f"COPY E FROM '{WORK}/edges.parquet'"), "")[1])
    print(f"  -> {N/tn:,.0f} nodes/s   {N/te:,.0f} edges/s", flush=True)

def load_create(c, nodes, edges, batch=10_000):
    c.execute("BEGIN TRANSACTION")
    nrows = nodes.to_pylist()
    t0 = time.perf_counter()
    for i in range(0, N, batch):
        c.execute("UNWIND $rows AS r CREATE (:N {value: r.value, type: r.type, prop0: r.prop0, prop1: r.prop1, prop2: r.prop2, prop3: r.prop3, prop4: r.prop4})", {"rows": nrows[i:i+batch]})
    tn = time.perf_counter() - t0
    print(f"  CREATE nodes {tn:.3f}s ({N/tn:,.0f}/s)", flush=True)
    erows = edges.to_pylist()
    t0 = time.perf_counter()
    for i in range(0, N, batch):
        c.execute("UNWIND $rows AS r MATCH (a:N {value: r.src}), (b:N {value: r.dst}) CREATE (a)-[:E {type: r.type, value: r.value}]->(b)", {"rows": erows[i:i+batch]})
    te = time.perf_counter() - t0
    t0 = time.perf_counter(); c.execute("COMMIT"); tc = time.perf_counter() - t0
    print(f"  CREATE edges {te:.3f}s ({N/te:,.0f}/s)  commit {tc:.3f}s", flush=True)

def count(c, q, p=None):
    return c.execute(q, p).get_next()[0]

def queries(c, threads):
    c.set_max_threads_for_exec(threads)
    print(f"-- queries, {threads} thread(s)", flush=True)
    q = "MATCH (n:N) WHERE n.value = $v AND n.type = $t RETURN n.value"
    def points():
        hits = 0
        for i in range(10_000):
            k = (i * 104_729) % N
            hits += 1 if c.execute(q, {"v": str(k), "t": f"node{k % 5}"}).has_next() else 0
        return hits
    timed("10k point lookups (one call each)", points)
    keys = [str((i * 104_729) % N) for i in range(10_000)]
    timed("10k point lookups (one statement)", lambda: count(c, "UNWIND $keys AS k MATCH (n:N {value: k}) RETURN count(*)", {"keys": keys}))
    timed('n(type="node3")', lambda: count(c, "MATCH (n:N) WHERE n.type = 'node3' RETURN count(*)"))
    timed('n(prop2="value2")', lambda: count(c, "MATCH (n:N) WHERE n.prop2 = 'value2' RETURN count(*)"))
    timed('e(type="edge3")', lambda: count(c, "MATCH ()-[e:E]->() WHERE e.type = 'edge3' RETURN count(*)"))
    timed('n(type="node1")->e()->n()', lambda: count(c, "MATCH (a:N)-[e:E]->(b:N) WHERE a.type = 'node1' RETURN count(*)"))
    timed('n(type="node1")-n()-n()', lambda: count(c, "MATCH (a:N)-[]-(b:N)-[]-(c:N) WHERE a.type = 'node1' RETURN count(*)"))
    timed('n(type="node1")-e(type="edge2")-n()', lambda: count(c, "MATCH (a:N)-[e:E]-(b:N) WHERE a.type = 'node1' AND e.type = 'edge2' RETURN count(*)"))

try:
    print(f"LadybugDB {lb.__version__}, N={N}, mode={MODE}", flush=True)
    t0 = time.perf_counter(); nodes = nodes_table(); print(f"(generated node rows in {time.perf_counter()-t0:.1f}s)", flush=True)
    if MODE in ("all", "copy"):
        db, c, path = fresh("copy.lbdb")
        print("== insert-bench dataset, COPY FROM parquet", flush=True)
        t0 = time.perf_counter(); e = insert_edges(); print(f"(generated edge rows in {time.perf_counter()-t0:.1f}s)", flush=True)
        load_copy(c, nodes, e)
        print(f"size       {dir_size(path) >> 20} MiB", flush=True); del c, db
    if MODE in ("all", "create"):
        db, c, path = fresh("create.lbdb")
        print("== insert-bench dataset, CREATE statements", flush=True)
        load_create(c, nodes, insert_edges())
        print(f"size       {dir_size(path) >> 20} MiB", flush=True); del c, db
    if MODE in ("all", "query"):
        db, c, path = fresh("query.lbdb")
        print("== query-bench dataset, COPY FROM parquet", flush=True)
        load_copy(c, nodes, query_edges())
        queries(c, 1); queries(c, 8); del c, db
finally:
    shutil.rmtree(WORK, ignore_errors=True)
