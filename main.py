"""
VectorDB — Python port of the C++ vector database
Implements HNSW, KD-Tree, and Brute Force search + RAG pipeline via Ollama.
"""

import math
import random
import time
import threading
import json
import re
from typing import Callable, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import heapq

import requests
from flask import Flask, request, jsonify, send_file
import os

# =====================================================================
#  CONSTANTS
# =====================================================================
DIMS = 16  # demo vector dimensions

# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: list[float]

DistFn = Callable[[list[float], list[float]], float]

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))

def manhattan(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))

def get_dist_fn(m: str) -> DistFn:
    if m == "cosine":    return cosine
    if m == "manhattan": return manhattan
    return euclidean

# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: list[VectorItem] = []

    def insert(self, v: VectorItem):
        self.items.append(v)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        results = [(dist(q, v.emb), v.id) for v in self.items]
        results.sort()
        return results[:k]

    def remove(self, id: int):
        self.items = [v for v in self.items if v.id != id]

# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    def __init__(self, item: VectorItem):
        self.item = item
        self.left: Optional['KDNode'] = None
        self.right: Optional['KDNode'] = None

class KDTree:
    def __init__(self, dims: int):
        self.root: Optional[KDNode] = None
        self.dims = dims

    def _insert(self, node: Optional[KDNode], v: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(v)
        ax = depth % self.dims
        if v.emb[ax] < node.item.emb[ax]:
            node.left = self._insert(node.left, v, depth + 1)
        else:
            node.right = self._insert(node.right, v, depth + 1)
        return node

    def insert(self, v: VectorItem):
        self.root = self._insert(self.root, v, 0)

    def _knn(self, node: Optional[KDNode], q: list[float], k: int, depth: int,
             dist: DistFn, heap: list):
        if node is None:
            return
        dn = dist(q, node.item.emb)
        # Max-heap using negative distances
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, node.item.id))
            if len(heap) > k:
                heapq.heappop(heap)
        ax = depth % self.dims
        diff = q[ax] - node.item.emb[ax]
        closer  = node.left  if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left
        self._knn(closer, q, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist, heap)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        heap = []
        self._knn(self.root, q, k, 0, dist, heap)
        results = [(-d, id) for d, id in heap]
        results.sort()
        return results

    def rebuild(self, items: list[VectorItem]):
        self.root = None
        for v in items:
            self.insert(v)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSW:
    @dataclass
    class Node:
        item: VectorItem
        max_lyr: int
        nbrs: list[list[int]] = field(default_factory=list)

    @dataclass
    class GraphInfo:
        top_layer: int
        node_count: int
        nodes_per_layer: list[int]
        edges_per_layer: list[int]
        nodes: list[dict]
        edges: list[dict]

    def __init__(self, m: int = 16, ef_build: int = 200):
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(m)
        self.top_layer = -1
        self.entry_pt = -1
        self.G: dict[int, HNSW.Node] = {}
        random.seed(42)

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(random.random()) * self.mL))

    def _search_layer(self, q: list[float], ep: int, ef: int, lyr: int,
                      dist: DistFn) -> list[tuple[float, int]]:
        vis = {ep}
        d0 = dist(q, self.G[ep].item.emb)
        # Min-heap for candidates, max-heap (neg) for found
        cands = [(d0, ep)]
        found = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            node = self.G.get(cid)
            if node is None or lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if nid in vis or nid not in self.G:
                    continue
                vis.add(nid)
                nd = dist(q, self.G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        results = [(-d, id) for d, id in found]
        results.sort()
        return results

    def _select_nbrs(self, cands: list[tuple[float, int]], max_m: int) -> list[int]:
        return [c[1] for c in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn):
        id = item.id
        lvl = self._rand_level()
        node = HNSW.Node(item=item, max_lyr=lvl, nbrs=[[] for _ in range(lvl + 1)])
        self.G[id] = node

        if self.entry_pt == -1:
            self.entry_pt = id
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if ep in self.G and lc < len(self.G[ep].nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(W, max_m)
            self.G[id].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                nd = self.G[nid]
                if len(nd.nbrs) <= lc:
                    nd.nbrs.extend([] for _ in range(lc + 1 - len(nd.nbrs)))
                conn = nd.nbrs[lc]
                conn.append(id)
                if len(conn) > max_m:
                    ds = [(dist(nd.item.emb, self.G[c].item.emb), c)
                          for c in conn if c in self.G]
                    ds.sort()
                    nd.nbrs[lc] = [c for _, c in ds[:max_m]]

            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = id

    def knn(self, q: list[float], k: int, ef: int, dist: DistFn) -> list[tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if ep in self.G and lc < len(self.G[ep].nbrs):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, id: int):
        if id not in self.G:
            return
        for nid, nd in self.G.items():
            for layer in nd.nbrs:
                if id in layer:
                    layer.remove(id)
        if self.entry_pt == id:
            self.entry_pt = -1
            for nid in self.G:
                if nid != id:
                    self.entry_pt = nid
                    break
        del self.G[id]

    def get_info(self) -> 'HNSW.GraphInfo':
        max_l = max(self.top_layer + 1, 1)
        nodes_per = [0] * max_l
        edges_per = [0] * max_l
        nodes = []
        edges = []
        for id, nd in self.G.items():
            nodes.append({"id": id, "metadata": nd.item.metadata,
                          "category": nd.item.category, "maxLyr": nd.max_lyr})
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id < nid:
                            edges_per[lc] += 1
                            edges.append({"src": id, "dst": nid, "lyr": lc})
        return HNSW.GraphInfo(
            top_layer=self.top_layer, node_count=len(self.G),
            nodes_per_layer=nodes_per, edges_per_layer=edges_per,
            nodes=nodes, edges=edges
        )

    def size(self) -> int:
        return len(self.G)

# =====================================================================
#  VECTOR DATABASE  (demo 16D index)
# =====================================================================

class VectorDB:
    def __init__(self, d: int):
        self.dims = d
        self.store: dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(d)
        self.hnsw = HNSW(16, 200)
        self.lock = threading.Lock()
        self.next_id = 1

    def insert(self, meta: str, cat: str, emb: list[float], dist: DistFn) -> int:
        with self.lock:
            v = VectorItem(id=self.next_id, metadata=meta, category=cat, emb=emb)
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, id: int) -> bool:
        with self.lock:
            if id not in self.store:
                return False
            del self.store[id]
            self.bf.remove(id)
            self.hnsw.remove(id)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: list[float], k: int, metric: str, algo: str) -> dict:
        with self.lock:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)
            us = int((time.perf_counter() - t0) * 1_000_000)
            hits = [{"id": id, "meta": self.store[id].metadata,
                     "cat": self.store[id].category,
                     "emb": self.store[id].emb, "dist": d}
                    for d, id in raw if id in self.store]
            return {"hits": hits, "us": us, "algo": algo, "metric": metric}

    def benchmark(self, q: list[float], k: int, metric: str) -> dict:
        with self.lock:
            dfn = get_dist_fn(metric)
            def timed(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)
            return {
                "bfUs":   timed(lambda: self.bf.knn(q, k, dfn)),
                "kdUs":   timed(lambda: self.kdt.knn(q, k, dfn)),
                "hnswUs": timed(lambda: self.hnsw.knn(q, k, 50, dfn)),
                "n":      len(self.store)
            }

    def all(self) -> list[VectorItem]:
        with self.lock:
            return list(self.store.values())

    def hnsw_info(self):
        with self.lock:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.lock:
            return len(self.store)

# =====================================================================
#  OLLAMA CLIENT
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> list[float]:
        try:
            r = requests.post(
                f"{self.base}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30
            )
            if r.status_code != 200:
                return []
            data = r.json()
            return data.get("embedding", [])
        except Exception:
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
#  DOCUMENT DATABASE
# =====================================================================

@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: list[float]

class DocumentDB:
    def __init__(self):
        self.store: dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.lock = threading.Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title: str, text: str, emb: list[float]) -> int:
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(id=self.next_id, title=title, text=text, emb=emb)
            self.next_id += 1
            self.store[item.id] = item
            vi = VectorItem(id=item.id, metadata=title, category="doc", emb=emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: list[float], k: int, max_dist: float = 0.7) -> list[tuple[float, DocItem]]:
        with self.lock:
            if not self.store:
                return []
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, 50, cosine)
            return [(d, self.store[id]) for d, id in raw
                    if id in self.store and d <= max_dist]

    def remove(self, id: int) -> bool:
        with self.lock:
            if id not in self.store:
                return False
            del self.store[id]
            self.hnsw.remove(id)
            self.bf.remove(id)
            return True

    def all(self) -> list[DocItem]:
        with self.lock:
            return list(self.store.values())

    def size(self) -> int:
        with self.lock:
            return len(self.store)

    def get_dims(self) -> int:
        return self.dims

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    step = chunk_words - overlap_words
    chunks = []
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks

# =====================================================================
#  DEMO DATA  (16D categorical vectors)
# =====================================================================

def load_demo(db: VectorDB):
    dist = get_dist_fn("cosine")
    # Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    demo_data = [
        ("Linked List: nodes connected by pointers", "cs",
         [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
        ("Calculus: derivatives integrals and limits", "math",
         [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
    ]
    for meta, cat, emb in demo_data:
        db.insert(meta, cat, emb, dist)

# =====================================================================
#  HTTP SERVER (Flask)
# =====================================================================

app = Flask(__name__)

db     = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

load_demo(db)

def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.after_request
def after_request(response):
    return add_cors(response)

@app.route("/", methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path=""):
    return "", 204

# ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────

@app.route("/search")
def search():
    v_str = request.args.get("v", "")
    try:
        q = [float(x) for x in v_str.split(",") if x.strip()]
    except ValueError:
        q = []
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400

    k = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    algo   = request.args.get("algo", "hnsw")

    out = db.search(q, k, metric, algo)
    results = [{
        "id":        h["id"],
        "metadata":  h["meta"],
        "category":  h["cat"],
        "distance":  round(h["dist"], 6),
        "embedding": [round(x, 4) for x in h["emb"]]
    } for h in out["hits"]]
    return jsonify({"results": results, "latencyUs": out["us"],
                    "algo": out["algo"], "metric": out["metric"]})

@app.route("/insert", methods=["POST"])
def insert():
    body = request.get_json(force=True, silent=True) or {}
    meta = body.get("metadata", "")
    cat  = body.get("category", "")
    emb  = body.get("embedding", [])
    if not meta or len(emb) != DIMS:
        return jsonify({"error": "invalid body"}), 400
    id = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return jsonify({"id": id})

@app.route("/delete/<int:id>", methods=["DELETE"])
def delete(id):
    ok = db.remove(id)
    return jsonify({"ok": ok})

@app.route("/items")
def items():
    all_items = db.all()
    return jsonify([{
        "id":        v.id,
        "metadata":  v.metadata,
        "category":  v.category,
        "embedding": [round(x, 4) for x in v.emb]
    } for v in all_items])

@app.route("/benchmark")
def benchmark():
    v_str = request.args.get("v", "")
    try:
        q = [float(x) for x in v_str.split(",") if x.strip()]
    except ValueError:
        q = []
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    k = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    b = db.benchmark(q, k, metric)
    return jsonify({"bruteforceUs": b["bfUs"], "kdtreeUs": b["kdUs"],
                    "hnswUs": b["hnswUs"], "itemCount": b["n"]})

@app.route("/hnsw-info")
def hnsw_info():
    gi = db.hnsw_info()
    return jsonify({
        "topLayer":     gi.top_layer,
        "nodeCount":    gi.node_count,
        "nodesPerLayer": gi.nodes_per_layer,
        "edgesPerLayer": gi.edges_per_layer,
        "nodes": gi.nodes,
        "edges": gi.edges
    })

@app.route("/stats")
def stats():
    return jsonify({
        "count":      db.size(),
        "dims":       DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics":    ["euclidean", "cosine", "manhattan"]
    })

# ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────

@app.route("/doc/insert", methods=["POST"])
def doc_insert():
    body = request.get_json(force=True, silent=True) or {}
    title = body.get("title", "")
    text  = body.get("text", "")
    if not title or not text:
        return jsonify({"error": "need title and text"}), 400

    chunks = chunk_text(text, 250, 30)
    ids = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify({"error":
                "Ollama unavailable. Install from https://ollama.com then run: "
                "ollama pull nomic-embed-text && ollama pull llama3.2"}), 503
        chunk_title = (f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title)
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})

@app.route("/doc/delete/<int:id>", methods=["DELETE"])
def doc_delete(id):
    ok = doc_db.remove(id)
    return jsonify({"ok": ok})

@app.route("/doc/list")
def doc_list():
    docs = doc_db.all()
    result = []
    for d in docs:
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        result.append({
            "id":      d.id,
            "title":   d.title,
            "preview": preview,
            "words":   len(d.text.split())
        })
    return jsonify(result)

@app.route("/doc/search", methods=["POST"])
def doc_search():
    body = request.get_json(force=True, silent=True) or {}
    question = body.get("question", "")
    k = body.get("k", 3)
    if not question:
        return jsonify({"error": "need question"}), 400
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 503
    hits = doc_db.search(q_emb, k)
    return jsonify({"contexts": [
        {"id": d.id, "title": d.title, "distance": round(dist, 4)}
        for dist, d in hits
    ]})

@app.route("/doc/ask", methods=["POST"])
def doc_ask():
    body = request.get_json(force=True, silent=True) or {}
    question = body.get("question", "")
    k = body.get("k", 3)
    if not question:
        return jsonify({"error": "need question"}), 400

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 503

    hits = doc_db.search(q_emb, k)

    ctx = ""
    for i, (dist, d) in enumerate(hits):
        ctx += f"[{i+1}] {d.title}:\n{d.text}\n\n"

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx}"
        f"Question: {question}\n\nAnswer:"
    )

    answer = ollama.generate(prompt)
    return jsonify({
        "answer":   answer,
        "model":    ollama.gen_model,
        "contexts": [{
            "id":       d.id,
            "title":    d.title,
            "text":     d.text,
            "distance": round(dist, 4)
        } for dist, d in hits],
        "docCount": doc_db.size()
    })

@app.route("/status")
def status():
    up = ollama.is_available()
    return jsonify({
        "ollamaAvailable": up,
        "embedModel":      ollama.embed_model,
        "genModel":        ollama.gen_model,
        "docCount":        doc_db.size(),
        "docDims":         doc_db.get_dims(),
        "demoDims":        DIMS,
        "demoCount":       db.size()
    })

@app.route("/")
def index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(html_path):
        return "index.html not found", 404
    return send_file(html_path)

# =====================================================================
#  ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    ollama_up = ollama.is_available()
    print("=== VectorDB Engine (Python) ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")
    app.run(host="0.0.0.0", port=8080, threaded=True)
