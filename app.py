import os
import json
import re
import math
import io
import hashlib
from types import SimpleNamespace
from collections import Counter, defaultdict
from datetime import datetime, timezone

from flask import Flask, render_template, request, jsonify
from flask_compress import Compress  # ← compressão HTTP

# ---- K8s client ----
from kubernetes import client, config
from kubernetes.client import ApiException
from kubernetes.client.rest import ApiException as _K8sApiEx

# ---- ML deps ----
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# ---- Matplotlib (gráficos no servidor) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import time
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed  # ← paralelismo

# =====================================================
# Caches simples em memória (TTL) para aliviar o backend
# =====================================================
class TTLCache:
    def __init__(self, ttl=60, maxsize=64):
        self.ttl = ttl
        self.maxsize = maxsize
        self._d = {}
        self._lock = Lock()
    def get(self, key):
        with self._lock:
            v = self._d.get(key)
            if not v:
                return None
            payload, ts = v
            if time.time() - ts > self.ttl:
                self._d.pop(key, None)
                return None
            return payload
    def set(self, key, payload):
        with self._lock:
            if len(self._d) >= self.maxsize:
                # descarte FIFO simples
                self._d.pop(next(iter(self._d)))
            self._d[key] = (payload, time.time())

CACHE_ERRORS   = TTLCache(ttl=45)  # agrega erros/summary
CACHE_CLUSTER  = TTLCache(ttl=45)  # projeção 2D
CACHE_GROUPS   = TTLCache(ttl=45)  # grupos brutos (ns/limit/tail)
LOGS_CACHE_TTL   = int(os.environ.get("LOGS_TTL", "30"))
EVENTS_CACHE_TTL = int(os.environ.get("EVENTS_TTL", "30"))
CACHE_LOGS     = TTLCache(ttl=LOGS_CACHE_TTL, maxsize=256)
CACHE_EVENTS   = TTLCache(ttl=EVENTS_CACHE_TTL, maxsize=256)

# Workers para buscar logs/eventos em paralelo
K8S_FETCH_WORKERS = int(os.environ.get("K8S_FETCH_WORKERS", "12"))
EXEC = ThreadPoolExecutor(max_workers=K8S_FETCH_WORKERS)

# ===========================
# Kubernetes client bootstrap
# ===========================
def _init_kube():
    try:
        config.load_kube_config()          # desenvolvimento local
        print("[k8s] load_kube_config() OK – rodando fora do cluster")
    except Exception:
        config.load_incluster_config()     # execução dentro do cluster
        print("[k8s] load_incluster_config() OK – rodando *dentro* do cluster")

_init_kube()

v1          = client.CoreV1Api()
apps        = client.AppsV1Api()
networking  = client.NetworkingV1Api()
batch       = client.BatchV1Api()

# ---------------------------
# Current namespace discovery
# ---------------------------
def _current_namespace() -> str:
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r", encoding="utf-8") as f:
            return f.read().strip() or "default"
    except Exception:
        return os.environ.get("POD_NAMESPACE", "default")

CURRENT_NAMESPACE = _current_namespace()
print(f"[k8s] running in namespace: {CURRENT_NAMESPACE}")

# ---------------------------------------------------------------------------
# Safe list-all helper – tenta cluster-scope, faz fallback para o próprio ns
# ---------------------------------------------------------------------------
_original_list_all = v1.list_pod_for_all_namespaces  # backup

def _list_all_with_fallback(*args, **kwargs):
    """
    1ª tentativa: listagem cluster-wide.
    Se receber 403 (RBAC), refaz a chamada namespaced
    preservando o tipo de retorno (V1PodList).
    """
    try:
        return _original_list_all(*args, **kwargs)
    except _K8sApiEx as e:
        if e.status == 403:
            print(f"[k8s] 403 ao listar todos os pods – usando namespace '{CURRENT_NAMESPACE}'")
            try:
                return v1.list_namespaced_pod(namespace=CURRENT_NAMESPACE, *args, **kwargs)
            except Exception as inner:
                print("[k8s] list_namespaced_pod também falhou:", inner)
                return client.V1PodList(items=[])
        raise  # outras exceções propagam

# monkey-patch global
v1.list_pod_for_all_namespaces = _list_all_with_fallback

# ===========================
# Config: dataset + snapshots
# ===========================
DATASET_PATH = os.environ.get("DATASET_PATH", "dataset.json")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"dataset.json not found at {DATASET_PATH}. Place your dataset file in the project root."
    )

# ---------- helpers de pattern / dataset ----------
def _compile_pattern(pat: str):
    if not isinstance(pat, str) or not pat.strip():
        return None
    s = pat.strip()
    if s.startswith("re:"):
        return re.compile(s[3:], re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return re.compile(re.escape(s), re.IGNORECASE | re.MULTILINE | re.DOTALL)

def _safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s or '')

def _ensure_dataset_ids_and_compile(dataset: list):
    changed = False
    for item in dataset:
        if not item.get("id"):
            raw = (
                item.get("error", "")
                + "|" + "|".join(item.get("patterns") or [])
                + "|" + item.get("solution", "")
            ).encode("utf-8")
            item["id"] = hashlib.sha1(raw).hexdigest()[:12]
            changed = True
        pats = item.get("patterns") or [item.get("error", "")]
        compiled = [rx for rx in (_compile_pattern(p) for p in pats) if rx is not None]
        item["_compiled"] = compiled
    return changed

def _load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("dataset.json deve conter uma lista de objetos.")
    changed = _ensure_dataset_ids_and_compile(data)
    return data, changed

def _save_dataset(dataset: list):
    to_save = []
    for item in dataset:
        it = dict(item)
        it.pop("_compiled", None)
        to_save.append(it)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)

# Carrega dataset
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    DATASET = json.load(f)
_ensure_dataset_ids_and_compile(DATASET)

HISTORY_DIR = os.environ.get("HISTORY_DIR", "./history")
os.makedirs(HISTORY_DIR, exist_ok=True)

# ===========================
# Tempo/formatos
# ===========================
def _safe_ts(ts):
    try:
        return ts.strftime("%d/%m %H:%M") if ts else ""
    except Exception:
        return ""

def _age_str(dt):
    if not dt:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    delta = datetime.utcnow().replace(tzinfo=timezone.utc) - dt
    s = int(delta.total_seconds())
    if s < 60:  return f"{s}s"
    m = s//60
    if m < 60:  return f"{m}m"
    h = m//60
    if h < 48:  return f"{h}h"
    d = h//24
    return f"{d}d"

def _to_epoch_utc(dt):
    if not dt:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.timestamp()

def _fmt_time(dt):
    if not dt:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%d/%m %H:%M")

# ===========================
# Helpers / filtros
# ===========================
DNS1123_LABEL_RX = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
def _is_dns1123_label(s: str) -> bool:
    return bool(s and DNS1123_LABEL_RX.match(s))

def _fmt_pod_list(pods_set: set[str]) -> tuple[str, int]:
    pods = sorted(pods_set)
    n = len(pods)
    if n == 0:
        return "", 0
    if n == 1:
        return pods[0], 1
    if n == 2:
        return f"{pods[0]} e {pods[1]}", 2
    if n == 3:
        return f"{pods[0]}, {pods[1]} e {pods[2]}", 3
    return f"{pods[0]}, {pods[1]} e outros {n-2}", n

def _verb_agree(n: int, singular="está", plural="estão") -> str:
    return singular if n == 1 else plural

# ===========================
# K8s helpers
# ===========================
def get_pod(ns: str, pod: str):
    try:
        return v1.read_namespaced_pod(pod, ns)
    except ApiException:
        return None

def get_pod_basic(ns: str, pod: str):
    p = get_pod(ns, pod)
    if not p:
        return {}
    restarts = sum((cs.restart_count or 0) for cs in (p.status.container_statuses or []))
    conditions = [
        {
            "type": c.type,
            "status": c.status,
            "reason": getattr(c, "reason", "") or "",
            "message": getattr(c, "message", "") or "",
        }
        for c in (p.status.conditions or [])
    ]
    containers = []
    for cs in (p.status.container_statuses or []):
        state = (
            "running" if cs.state and cs.state.running else
            "waiting" if cs.state and cs.state.waiting else
            "terminated" if cs.state and cs.state.terminated else
            "unknown"
        )
        reason = (
            (cs.state.waiting and cs.state.waiting.reason) or
            (cs.state.terminated and cs.state.terminated.reason) or
            ""
        )
        containers.append({
            "name": cs.name,
            "ready": bool(cs.ready),
            "restarts": cs.restart_count or 0,
            "state": state,
            "reason": reason
        })
    return {
        "name": p.metadata.name,
        "namespace": p.metadata.namespace,
        "labels": p.metadata.labels or {},
        "annotations": p.metadata.annotations or {},
        "phase": p.status.phase,
        "pod_ip": p.status.pod_ip,
        "host_ip": p.status.host_ip,
        "node_name": getattr(p.spec, "node_name", ""),
        "containers": containers,
        "conditions": conditions,
        "restarts": restarts,
        "start_time": _safe_ts(p.status.start_time),
        "owner_refs": [{"kind": o.kind, "name": o.name, "uid": o.uid} for o in (p.metadata.owner_references or [])]
    }

# ===========================
# Logs (com cache)
# ===========================
def _container_status_map(p):
    by_name = {}
    for arr_name in ("container_statuses", "init_container_statuses", "ephemeral_container_statuses"):
        arr = getattr(p.status, arr_name, None) or []
        for cs in arr:
            by_name[cs.name] = cs
    return by_name

def _iter_all_container_names(p):
    names = []
    for arr_name in ("containers", "init_containers", "ephemeral_containers"):
        arr = getattr(p.spec, arr_name, None) or []
        for c in arr:
            n = getattr(c, "name", None)
            if n:
                names.append((arr_name, n))
    order = {"init_containers": 0, "containers": 1, "ephemeral_containers": 2}
    names.sort(key=lambda t: (order.get(t[0], 99), t[1]))
    return names

def collect_logs(namespace: str, pod: str, tail: int = 800) -> str:
    p = get_pod(namespace, pod)
    if not p:
        return "Erro ao obter logs: Pod não encontrado."
    out_parts = []
    status_by_name = _container_status_map(p)
    for kind, cname in _iter_all_container_names(p):
        header_base = f"===== [{kind.replace('_', ' ')}/{cname}] "
        try:
            curr = v1.read_namespaced_pod_log(
                name=pod, namespace=namespace,
                container=cname, tail_lines=tail, previous=False
            )
            if curr:
                out_parts.append(f"{header_base}atual =====\n{curr.rstrip()}\n")
        except Exception as e:
            out_parts.append(f"{header_base}atual =====\n<erro ao ler logs: {e}>\n")
        rcount = getattr(status_by_name.get(cname, None), "restart_count", 0) or 0
        if rcount > 0:
            try:
                prev = v1.read_namespaced_pod_log(
                    name=pod, namespace=namespace,
                    container=cname, tail_lines=tail, previous=True
                )
                if prev:
                    out_parts.append(f"{header_base}anterior =====\n{prev.rstrip()}\n")
            except Exception as e:
                out_parts.append(f"{header_base}anterior =====\n<erro ao ler logs anteriores: {e}>\n")
    return ("\n".join(out_parts)).rstrip() or "Sem logs disponíveis para os contêineres."

def collect_logs_cached(namespace: str, pod: str, tail: int = 800) -> str:
    key = (namespace, pod, int(tail))
    cached = CACHE_LOGS.get(key)
    if cached is not None:
        return cached
    val = collect_logs(namespace, pod, tail=tail)
    CACHE_LOGS.set(key, val)
    return val

# ========= Última execução =========
def collect_last_run(namespace: str, pod: str, tail: int = 800) -> dict:
    p = get_pod(namespace, pod)
    if not p:
        return {"containers": [], "last_finished_at": None, "summary": "Pod não encontrado."}

    out = {"containers": [], "last_finished_at": None}
    last_finished = None

    for cs in (p.status.container_statuses or []):
        name = cs.name
        rcount = cs.restart_count or 0
        logs_txt = ""
        finished_at = None
        reason = None
        exit_code = None

        if rcount > 0:
            try:
                logs_txt = v1.read_namespaced_pod_log(
                    name=pod, namespace=namespace, container=name, tail_lines=tail, previous=True
                ) or ""
            except Exception as e:
                logs_txt = f"<erro ao ler logs anteriores: {e}>"
            if cs.last_state and cs.last_state.terminated:
                t = cs.last_state.terminated
                finished_at = getattr(t, "finished_at", None)
                reason = t.reason
                exit_code = t.exit_code
        elif cs.state and cs.state.terminated:
            try:
                logs_txt = v1.read_namespaced_pod_log(
                    name=pod, namespace=namespace, container=name, tail_lines=tail, previous=False
                ) or ""
            except Exception as e:
                logs_txt = f"<erro ao ler logs (terminado): {e}>"
            t = cs.state.terminated
            finished_at = getattr(t, "finished_at", None)
            reason = t.reason
            exit_code = t.exit_code
        else:
            logs_txt = "<sem execução anterior registrada (container em execução e sem restarts)>"

        if finished_at and finished_at.tzinfo is None:
            finished_at = finished_at.replace(tzinfo=timezone.utc)
        if finished_at and (last_finished is None or finished_at > last_finished):
            last_finished = finished_at

        out["containers"].append({
            "name": name,
            "restart_count": rcount,
            "terminated": bool(cs.state and cs.state.terminated),
            "last_reason": reason,
            "last_exit_code": exit_code,
            "finished_at": finished_at.isoformat() if finished_at else None,
            "logs": logs_txt,
        })

    for cs in (p.status.init_container_statuses or []):
        name = cs.name
        logs_txt = ""
        finished_at = None
        reason = None
        exit_code = None
        try:
            logs_txt = v1.read_namespaced_pod_log(
                name=pod, namespace=namespace, container=name, tail_lines=tail, previous=False
            ) or ""
        except Exception as e:
            logs_txt = f"<erro ao ler logs init: {e}>"
        if cs.state and cs.state.terminated:
            t = cs.state.terminated
            finished_at = getattr(t, "finished_at", None)
            reason = t.reason
            exit_code = t.exit_code
            if finished_at and finished_at.tzinfo is None:
                finished_at = finished_at.replace(tzinfo=timezone.utc)
            if finished_at and (last_finished is None or finished_at > last_finished):
                last_finished = finished_at

        out["containers"].append({
            "name": name,
            "is_init": True,
            "restart_count": cs.restart_count or 0,
            "terminated": True,
            "last_reason": reason,
            "last_exit_code": exit_code,
            "finished_at": finished_at.isoformat() if finished_at else None,
            "logs": logs_txt,
        })

    out["last_finished_at"] = last_finished.isoformat() if last_finished else None
    return out

# ===========================
# Eventos (com cache)
# ===========================
def _read_workload(ns: str, kind: str, name: str):
    try:
        kind_l = (kind or "").lower()
        if kind_l == "replicaset":
            obj = apps.read_namespaced_replica_set(name, ns)
        elif kind_l == "deployment":
            obj = apps.read_namespaced_deployment(name, ns)
        elif kind_l == "statefulset":
            obj = apps.read_namespaced_stateful_set(name, ns)
        elif kind_l == "daemonset":
            obj = apps.read_namespaced_daemon_set(name, ns)
        elif kind_l == "job":
            obj = batch.read_namespaced_job(name, ns)
        elif kind_l == "cronjob":
            obj = batch.read_namespaced_cron_job(name, ns)
        else:
            return None, []
        uid = getattr(obj.metadata, "uid", None)
        orefs = [{"kind": o.kind, "name": o.name, "uid": o.uid} for o in (obj.metadata.owner_references or [])]
        return uid, orefs
    except Exception:
        return None, []

def _collect_owner_uids(ns: str, pod_obj):
    uids = set()
    queue = []
    pod_uid = getattr(pod_obj.metadata, "uid", None)
    if pod_uid:
        uids.add(pod_uid)
    for o in (pod_obj.metadata.owner_references or []):
        queue.append({"kind": o.kind, "name": o.name, "uid": getattr(o, "uid", None)})
    visited = set()
    while queue:
        item = queue.pop(0)
        key = (item["kind"], item["name"])
        if key in visited:
            continue
        uid = item.get("uid")
        if not uid:
            uid, orefs = _read_workload(ns, item["kind"], item["name"])
        else:
            _, orefs = _read_workload(ns, item["kind"], item["name"])
        if uid:
            uids.add(uid)
        for o in (orefs or []):
            queue.append({"kind": o["kind"], "name": o["name"], "uid": o.get("uid")})
    return uids

def collect_pod_events(namespace: str, pod: str, limit: int = 400) -> list[str]:
    try:
        pod_obj = v1.read_namespaced_pod(pod, namespace)
    except Exception:
        pod_obj = None
    selectors = []
    if pod_obj:
        for uid in _collect_owner_uids(namespace, pod_obj):
            selectors.append(f"involvedObject.uid={uid}")
    else:
        selectors.append(f"involvedObject.name={pod},involvedObject.namespace={namespace}")
    items = []
    seen = set()
    for sel in selectors:
        try:
            evs = v1.list_namespaced_event(namespace, field_selector=sel, limit=limit).items
        except Exception:
            evs = []
        for e in evs:
            lt = getattr(e, "last_timestamp", None)
            et = getattr(e, "event_time", None)
            ct = getattr(e, "metadata", None) and getattr(e.metadata, "creation_timestamp", None)
            ts = lt or et or ct
            reason = getattr(e, "reason", "") or ""
            msg = getattr(e, "message", "") or ""
            inv = getattr(e, "involved_object", None)
            ikind = getattr(inv, "kind", "") if inv else ""
            iname = getattr(inv, "name", "") if inv else ""
            ikey = getattr(inv, "uid", "") if inv else ""
            dedup_key = (ikey or f"{ikind}:{iname}", reason, msg, str(ts))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            items.append(e)
    def _ev_time(e):
        return getattr(e, "last_timestamp", None) or getattr(e, "event_time", None) or (
            getattr(e, "metadata", None) and getattr(e.metadata, "creation_timestamp", None)
        )
    items.sort(key=lambda e: _to_epoch_utc(_ev_time(e)), reverse=True)
    out = []
    for e in items:
        t = _ev_time(e)
        t_str = _fmt_time(t) if t else ""
        etype  = getattr(e, "type", "") or ""
        reason = getattr(e, "reason", "") or ""
        msg    = getattr(e, "message", "") or ""
        inv    = getattr(e, "involved_object", None)
        ikind  = getattr(inv, "kind", "") if inv else ""
        iname  = getattr(inv, "name", "") if inv else ""
        suffix = f"{ikind}/{iname}" if (ikind or iname) else ""
        out.append(f"[{etype}] {reason}: {msg} • {t_str}" + (f" • {suffix}" if suffix else ""))
    return out[:limit]

def collect_pod_events_cached(namespace: str, pod: str, limit: int = 400) -> list[str]:
    key = (namespace, pod, int(limit))
    cached = CACHE_EVENTS.get(key)
    if cached is not None:
        return cached
    val = collect_pod_events(namespace, pod, limit=limit)
    CACHE_EVENTS.set(key, val)
    return val

# ===========================
# Snapshot de última execução
# ===========================
def _snapshot_path(ns: str, pod: str) -> str:
    return os.path.join(HISTORY_DIR, f"last_{_safe_name(ns)}_{_safe_name(pod)}.json")

def _write_last_snapshot(ns: str, pod: str, payload: dict):
    fn = _snapshot_path(ns, pod)
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

def read_last_snapshot(ns: str, pod: str) -> dict | None:
    fn = _snapshot_path(ns, pod)
    if not os.path.exists(fn):
        return None
    try:
        with open(fn, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _pod_has_finalized(p) -> bool:
    phase = (p.status.phase or "").lower()
    if phase in ("succeeded", "failed"):
        return True
    for cs in (p.status.container_statuses or []):
        if cs.last_state and cs.last_state.terminated:
            return True
        if cs.state and cs.state.terminated:
            return True
    return False

def maybe_save_last_finalized_snapshot(ns: str, pod: str, tail: int = 800):
    p = get_pod(ns, pod)
    if not p:
        return False, "pod not found"
    if not _pod_has_finalized(p):
        return False, "not finalized"

    last_run = collect_last_run(ns, pod, tail=tail)
    events = collect_pod_events_cached(ns, pod)

    snap = {
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "namespace": ns,
        "pod": pod,
        "phase": p.status.phase,
        "node_name": getattr(p.spec, "node_name", ""),
        "last_finished_at": last_run.get("last_finished_at"),
        "containers": last_run.get("containers", []),
        "events": events,
    }
    _write_last_snapshot(ns, pod, snap)
    return True, "saved"

# ===========================
# Dataset matching (logs + eventos)
# ===========================
def find_matching_solutions(dataset, logs, events, ns=None, pod=None):
    log_text = logs or ""
    evt_text = "\n".join(events or [])

    matches = []
    for item in (dataset or []):
        for rx in item.get("_compiled", []):
            hit_logs = bool(log_text) and bool(rx.search(log_text))
            hit_evts = bool(evt_text) and bool(rx.search(evt_text))
            if hit_logs or hit_evts:
                sol  = _apply_placeholders(item.get("solution", ""), ns, pod)
                err  = _apply_placeholders(item.get("error", ""), ns, pod)
                expl = _apply_placeholders(item.get("explanation", ""), ns, pod)
                diags = _apply_placeholders_list(item.get("diagnostics"), ns, pod)
                fixes = _apply_placeholders_list(item.get("fix_steps"), ns, pod)
                cmds  = _apply_placeholders_list(item.get("k8s_commands"), ns, pod)
                refs  = _apply_placeholders_list(item.get("references"), ns, pod)

                matches.append({
                    "id": item.get("id"),
                    "error": err,
                    "patterns": item.get("patterns", []),
                    "solution": sol,
                    "matched_by": rx.pattern,
                    "source": "logs" if hit_logs else "events",
                    "category": item.get("category"),
                    "severity": item.get("severity"),
                    "explanation": expl,
                    "diagnostics": diags,
                    "fix_steps": fixes,
                    "tags": item.get("tags"),
                    "k8s_commands": cmds,
                    "references": refs,
                })
                break
    return matches

# ===========================
# Busca (dashboard)
# ===========================
def _tok(s): return (s or "").strip().lower()

def _parse_query(q: str):
    filters = {"ns":[], "label":[], "phase":[], "node":[], "owner":[], "name":[]}
    terms = []
    for raw in re.findall(r'(?:"[^"]+"|\S+)', q or ""):
        t = raw.strip().strip('"')
        if ":" in t:
            k, v = t.split("1", 1) if False else t.split(":", 1)  # guard contra lint
            k = k.lower(); v = v.strip()
            if not v: continue
            if k in ("ns","namespace"): filters["ns"].append(v)
            elif k in ("name",): filters["name"].append(v)
            elif k in ("phase","status"): filters["phase"].append(v)
            elif k == "label" and "=" in v:
                lk, lv = v.split("=", 1); filters["label"].append((lk.strip(), lv.strip()))
            elif k == "node": filters["node"].append(v)
            elif k == "owner": filters["owner"].append(v)
            else: terms.append(t)
        else:
            terms.append(t)
    return filters, terms

def _rank_pod(p, filters, terms):
    meta, stat, spec = p.metadata, p.status, p.spec
    name = meta.name or ""; ns = meta.namespace or ""
    phase = (stat.phase or ""); lbls = meta.labels or {}
    node = getattr(spec, "node_name", "") or ""
    owners = [o.name for o in (meta.owner_references or [])]
    for wanted in filters["ns"]   : 
        if _tok(wanted) not in _tok(ns): return False, 0.0
    for wanted in filters["phase"]:
        if _tok(wanted) not in _tok(phase): return False, 0.0
    for wanted in filters["node"] :
        if _tok(wanted) not in _tok(node): return False, 0.0
    for wanted in filters["owner"]:
        if not any(_tok(wanted) in _tok(o) for o in owners): return False, 0.0
    for (lk, lv) in filters["label"]:
        if lbls.get(lk) is None or _tok(lv) not in _tok(lbls.get(lk, "")): return False, 0.0
    for wanted in filters["name"]:
        if _tok(wanted) not in _tok(name): return False, 0.0
    score = 50.0
    base = f"{name} {ns}"
    for t in terms:
        if _tok(t) in _tok(base): score += 10
    if (stat.phase or "").lower() in ("failed","pending"): score += 5
    rsts = sum((cs.restart_count or 0) for cs in (stat.container_statuses or []))
    score += min(rsts, 5)
    return True, score

# ===========================
# Tags / regex de erros
# ===========================
TAG_PATTERNS = [
    ("timeout", re.compile(r"\b(i/o timeout|timed? ?out|context deadline exceeded|ETIMEDOUT)\b", re.I)),
    ("refused", re.compile(r"\b(connection refused|ECONNREFUSED|no endpoints available)\b", re.I)),
    ("tls", re.compile(r"\b(x509|TLS handshake|bad certificate|unknown authority)\b", re.I)),
    ("imagepull", re.compile(r"\b(ImagePullBackOff|ErrImagePull|manifest unknown|unauthorized)\b", re.I)),
    ("oom", re.compile(r"\b(OOMKilled|Out of memory|memory cgroup out of memory)\b", re.I)),
    ("probe", re.compile(r"\b(liveness probe failed|readiness probe failed|startup probe failed)\b", re.I)),
    ("dns", re.compile(r"\b(no such host|Temporary failure in name resolution)\b", re.I)),
    ("storage", re.compile(r"\b(ephemeral-storage|No space left on device|ENOSPC|Read-only file system)\b", re.I)),
]

def quick_tags(text: str):
    tags = set()
    for name, rx in TAG_PATTERNS:
        if rx.search(text or ""):
            tags.add(name)
    return sorted(tags)

# ===========================
# Normalização / templates
# ===========================

def _flatten_json_for_tpl(x, prefix=(), out=None, budget=200):
    # Gera features "k1.k2=<TYPE>" com limite de orçamento p/ não explodir
    if out is None: out = {}
    if len(out) >= budget: return out
    if isinstance(x, dict):
        for k, v in (x.items() if isinstance(x, dict) else []):
            _flatten_json_for_tpl(v, prefix + (str(k),), out, budget)
    elif isinstance(x, list):
        # Só marca que é lista (não explodir por índice)
        out[".".join(prefix) + "[]"] = "<ARR>"
    elif isinstance(x, (int, float)):
        out[".".join(prefix)] = "<NUM>"
    elif isinstance(x, bool):
        out[".".join(prefix)] = "<BOOL>"
    elif x is None:
        out[".".join(prefix)] = "<NULL>"
    else:
        out[".".join(prefix)] = "<STR>"
    return out

def _jsonish_signature(s: str) -> str | None:
    s = (s or "").strip()
    try:
        obj = json.loads(s)
        feat = _flatten_json_for_tpl(obj, budget=260)
        parts = [f"{k}={v}" for k, v in sorted(feat.items())[:60]]
        return "JSON{" + ",".join(parts) + "}"
    except Exception:
        pass
    m = re.search(r"\{.*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        feat = _flatten_json_for_tpl(obj, budget=260)
        parts = [f"{k}={v}" for k, v in sorted(feat.items())[:60]]
        return "JSON{" + ",".join(parts) + "}"
    except Exception:
        return None

IPV4_RX = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")
UUID_RX = re.compile(r"\b[0-9a-f]{8}\-[0-9a-f]{4}\-[0-9a-f]{4}\-[0-9a-f]{4}\-[0-9a-f]{12}\b", re.I)
HEXLONG_RX = re.compile(r"\b[0-9a-f]{12,}\b", re.I)
TIME_RX = re.compile(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b")
ISOTS_RX = re.compile(r"\b\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}(?:\.\d+)?z?\b", re.I)
NUM_RX = re.compile(r"\b\d+\b")
PATH_RX = re.compile(r"(/[A-Za-z0-9._\-]+(?:/[A-Za-z0-9._\-]+)+)")
QUOTED_RX = re.compile(r"(['\"]).*?\1")
HOST_RX = re.compile(
    r"\b([a-z][a-z0-9-]*(?:\.[a-z0-9-]+)*\.(?:cluster\.local|localdomain|svc(?:\.cluster\.local)?|lan|local|com|net|org))\b",
    re.I
)
URL_HOST_RX = re.compile(
    r"https?://([a-z][a-z0-9-]*(?:\.[a-z0-9-]+)*\.(?:cluster\.local|localdomain|svc(?:\.cluster\.local)?|lan|local|com|net|org))(?:\:\d+)?",
    re.I
)
URL_ANY_RX = re.compile(r'https?://([^/\s":]+)', re.I)
HOST_KV_RX = re.compile(r'\b(host|target|addr|service|svc)=([a-z0-9]([-a-z0-9]*[a-z0-9]))(?::\d+)?\b', re.I)
RBAC_RX = re.compile(r"forbidden:.*?(?:User|serviceaccount).*? cannot (?P<verb>get|list|watch|create|update|patch|delete|deletecollection) (?P<resource>[a-zA-Z0-9./-]+)", re.I)
PERM_PATH_RX = re.compile(r"(?:permission denied|EACCES|EPERM):?\s*(?P<path>/[^\s]+)?", re.I)
TIMEOUT_RX = re.compile(r"(i/o timeout|context deadline exceeded|timed out\b|(?<!\w)timeout(?!\s*=\s*\d))", re.I)
REFUSED_RX = re.compile(r"(connection refused|ECONNREFUSED)", re.I)
NO_ENDPOINTS_RX = re.compile(r'no endpoints available for service\s+"?([\w\-.]+)"?', re.I)
PROBE_FAIL_RX = re.compile(r"(readiness probe failed|liveness probe failed|startup probe failed)", re.I)
DNS_FAIL_RX = re.compile(r"(no such host|Temporary failure in name resolution)", re.I)

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = ISOTS_RX.sub(" ", s)
    s = TIME_RX.sub(" ", s)
    s = re.sub(r"\b[0-9a-f]{8,}\b", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def strip_identifiers(text: str, ns: str, pod: str) -> str:
    if not text:
        return text
    toks = [re.escape(ns or ""), re.escape(pod or "")]
    toks = [t for t in toks if t]
    if not toks:
        return text
    rx = re.compile(r"\b(" + "|".join(toks) + r")\b", re.I)
    return rx.sub(" ", text)

def template_line(s: str) -> str:
    s = s or ""
    sig = _jsonish_signature(s)
    if sig:
        return sig
    s = QUOTED_RX.sub('"<STR>"', s)
    s = PATH_RX.sub("<PATH>", s)
    s = IPV4_RX.sub("<IP>", s)
    s = UUID_RX.sub("<UUID>", s)
    s = HEXLONG_RX.sub("<HEX>", s)
    s = ISOTS_RX.sub("<TIME>", s)
    s = TIME_RX.sub("<TIME>", s)
    s = NUM_RX.sub("<NUM>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:240] + "…") if len(s) > 240 else s


def extract_templates_from_text(text: str) -> Counter:
    c = Counter()
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        tpl = template_line(ln)
        if not tpl:
            continue
        c[tpl] += 1
    return c

def template_token(tpl: str) -> str:
    h = hashlib.sha1(tpl.encode("utf-8")).hexdigest()[:10]
    return f"tmpl_{h}"

def _svc_from_dns(host: str):
    if not host:
        return None, None
    parts = host.split(".")
    if len(parts) >= 4 and parts[-3] == "svc" and parts[-2] == "cluster" and parts[-1] == "local":
        if len(parts) >= 5:
            svc, ns = parts[-5], parts[-4]
        else:
            return None, None
    elif len(parts) >= 3 and parts[-1] == "svc":
        svc, ns = parts[-3], parts[-2]
    else:
        return None, None
    if _is_dns1123_label(svc) and _is_dns1123_label(ns):
        return svc, ns
    return None, None

def _service_endpoints(ns: str, svc_name: str):
    try:
        eps = v1.read_namespaced_endpoints(svc_name, ns)
    except ApiException:
        return []
    out = []
    for ss in (eps.subsets or []):
        for a in (ss.addresses or []):
            out.append(a.ip)
    return out

def _pods_by_ip_map(pods=None):
    m = {}
    if pods is None:
        pods = v1.list_pod_for_all_namespaces().items
    for p in pods:
        ip = getattr(p.status, "pod_ip", None)
        if ip:
            m[ip] = p
    return m

def _service_exists(ns: str, name: str):
    try:
        v1.read_namespaced_service(name, ns)
        return True
    except ApiException:
        return False

def _resolve_target_from_text(ns_default: str, text: str):
    out = []
    for ip in set(IPV4_RX.findall(text or "")):
        out.append({"type": "ip", "value": ip})
    for h in set(URL_HOST_RX.findall(text or "")):
        if IPV4_RX.fullmatch(h):
            out.append({"type": "ip", "value": h})
        else:
            out.append({"type": "host", "value": h})
    for h in set(HOST_RX.findall(text or "")):
        if IPV4_RX.fullmatch(h):
            out.append({"type": "ip", "value": h})
        else:
            out.append({"type": "host", "value": h})
    for netloc in set(URL_ANY_RX.findall(text or "")):
        host = netloc.split(":", 1)[0]
        if IPV4_RX.fullmatch(host):
            out.append({"type": "ip", "value": host})
        elif _is_dns1123_label(host):
            out.append({"type": "service", "value": host, "ns": ns_default})
        else:
            out.append({"type": "host", "value": host})
    for _, svc, _ in set(HOST_KV_RX.findall(text or "")):
        if _is_dns1123_label(svc):
            out.append({"type": "service", "value": svc, "ns": ns_default})
    for svc in set(NO_ENDPOINTS_RX.findall(text or "")):
        if _is_dns1123_label(svc):
            out.append({"type": "service", "value": svc, "ns": ns_default})
    new_out = []
    for t in out:
        if t["type"] == "host":
            svc, ns = _svc_from_dns(t["value"])
            if svc and ns:
                new_out.append({"type": "service", "value": svc, "ns": ns})
    out.extend(new_out)
    seen = set()
    dedup = []
    for t in out:
        key = (t["type"], t["value"], t.get("ns"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(t)
    return dedup

def _error_kind(line: str):
    low = line.lower()
    if "security purposes" in low and "timed out" in low:
        return None
    if TIMEOUT_RX.search(line): return "timeout"
    if REFUSED_RX.search(line): return "refused"
    if DNS_FAIL_RX.search(line): return "dns"
    if PROBE_FAIL_RX.search(line): return "probe"
    if "oomkilled" in low or "out of memory" in low: return "oom"
    if ("imagepullbackoff" in low or "errimagepull" in low or "manifest unknown" in low or "unauthorized" in low):
        return "imagepull"
    if "permission denied" in low or "forbidden" in low or "unauthorized" in low:
        return "permission"
    return None

# ==================================================
# Agrupamento por entidade/tipo (com paralelismo/TTL)
# ==================================================
def _fetch_logs_events(ns: str, pod: str, tail_lines: int):
    try:
        logs = collect_logs_cached(ns, pod, tail=tail_lines)
    except Exception as e:
        logs = f"<erro logs: {e}>"
    try:
        events = collect_pod_events_cached(ns, pod)
    except Exception as e:
        events = [f"<erro eventos: {e}>"]
    return ns, pod, logs, events

def analyze_similar_errors(
    ns_filter: list[str],
    limit_pods: int = 120,
    tail_lines: int = 600,
    item_limit: int = 0,
    per_pod_limit: int = 0
):
    if (not item_limit or item_limit <= 0) and (not per_pod_limit or per_pod_limit <= 0):
        key = ("groups", tuple(sorted(ns_filter or [])), int(limit_pods), int(tail_lines))
        cached = CACHE_GROUPS.get(key)
        if cached is not None:
            return cached

    pods = v1.list_pod_for_all_namespaces().items
    if ns_filter:
        pods = [p for p in pods if p.metadata.namespace in ns_filter]
    pods = pods[:limit_pods]

    ip_map = _pods_by_ip_map(pods)
    groups = {}
    perm_groups = {}

    def target_status(target, src_ns):
        if target["type"] == "service":
            svc = target["value"]
            tns = target.get("ns") or src_ns
            if not tns:
                return {"description": f"Service {svc} (ns desconhecido)", "cause": "unknown", "details": {}}
            if not _service_exists(tns, svc):
                return {"description": f"Service {svc}.{tns} não encontrado", "cause": "unknown", "details": {}}
            eps = _service_endpoints(tns, svc)
            if not eps:
                return {"description": f"Service {svc}.{tns} sem Endpoints", "cause": "no_endpoints", "details": {}}
            unhealthy = []
            for ip in eps:
                pod = ip_map.get(ip)
                if pod:
                    phase = (pod.status.phase or "")
                    ready = all(getattr(cs, "ready", False) for cs in (pod.status.container_statuses or []))
                    if phase != "Running" or not ready:
                        unhealthy.append({"pod": pod.metadata.name, "ns": pod.metadata.namespace, "phase": phase, "ready": ready})
            if unhealthy:
                return {"description": f"Service {svc}.{tns} com pods não prontos", "cause": "pods_unhealthy", "details": {"unhealthy": unhealthy}}
            return {"description": f"Service {svc}.{tns} com Endpoints OK", "cause": None, "details": {}}
        if target["type"] == "ip":
            ip = target["value"]
            pod = ip_map.get(ip)
            if pod:
                phase = (pod.status.phase or "")
                ready = all(getattr(cs, "ready", False) for cs in (pod.status.container_statuses or []))
                if phase != "Running" or not ready:
                    return {"description": f"Destino é pod {pod.metadata.name} ({ip}) não pronto", "cause": "pod_down",
                            "details": {"pod": pod.metadata.name, "ns": pod.metadata.namespace, "phase": phase, "ready": ready}}
                return {"description": f"Destino é pod {pod.metadata.name} ({ip}) pronto", "cause": None, "details": {}}
            return {"description": f"IP {ip} (pod não encontrado)", "cause": "unknown", "details": {}}
        if target["type"] == "host":
            return {"description": f"Host {target['value']}", "cause": "unknown", "details": {}}
        return {"description": "Alvo desconhecido", "cause": "unknown", "details": {}}

    futures = [EXEC.submit(_fetch_logs_events, p.metadata.namespace, p.metadata.name, tail_lines) for p in pods]
    for fut in as_completed(futures):
        ns, name, logs, events = fut.result()

        lines = []
        for blk in [logs] + ["\n".join(events)]:
            for ln in (blk or "").splitlines():
                l = ln.strip()
                if not l:
                    continue
                lines.append(l)

        for ln in lines:
            kind = _error_kind(ln)
            if kind:
                targets = _resolve_target_from_text(ns, ln)
                if not targets:
                    key = (kind, "<sem-alvo>")
                    grp = groups.setdefault(key, {"kind": kind, "entity": "<sem-alvo>", "items": [], "tmpl": Counter()})
                    grp["items"].append({"ns": ns, "pod": name, "line": ln})
                    grp["tmpl"][template_line(ln)] += 1
                else:
                    for t in targets:
                        ent = t["value"] if t["type"] != "service" else f"{t['value']}.{t.get('ns') or ns}"
                        key = (kind, f"{t['type']}:{ent}")
                        grp = groups.setdefault(key, {"kind": kind, "entity": f"{t['type']}:{ent}", "items": [], "target": t, "sources": set(), "tmpl": Counter()})
                        grp["items"].append({"ns": ns, "pod": name, "line": ln})
                        grp["sources"].add((ns, name))
                        grp["tmpl"][template_line(ln)] += 1

            m = RBAC_RX.search(ln)
            if m:
                verb = m.group("verb")
                resource = m.group("resource")
                k = (verb, resource)
                grp = perm_groups.setdefault(k, {"kind": "permission", "verb": verb, "resource": resource, "items": [], "tmpl": Counter()})
                grp["items"].append({"ns": ns, "pod": name, "line": ln})
                grp["tmpl"][template_line(ln)] += 1
            m2 = PERM_PATH_RX.search(ln)
            if m2 and ("permission denied" in ln.lower() or "eacces" in ln.lower() or "eperm" in ln.lower()):
                path = m2.group("path") or ""
                k = ("path", path or "<desconhecido>")
                grp = perm_groups.setdefault(k, {"kind": "permission", "path": path or "<desconhecido>", "items": [], "tmpl": Counter()})
                grp["items"].append({"ns": ns, "pod": name, "line": ln})
                grp["tmpl"][template_line(ln)] += 1

    out = []

    def _balanced_slice_by_pod(items, total_limit: int, per_pod_cap: int):
        if not items:
            return []
        by_pod = {}
        order = []
        for it in items:
            key = f"{it.get('ns','')}/{it.get('pod','')}"
            if key not in by_pod:
                by_pod[key] = []
                order.append(key)
            by_pod[key].append(it)
        if per_pod_cap and per_pod_cap > 0:
            for k in list(by_pod.keys()):
                by_pod[k] = by_pod[k][:per_pod_cap]
        out_rr = []
        i = 0
        max_len = max(len(v) for v in by_pod.values())
        while True:
            added = False
            for k in order:
                lst = by_pod[k]
                if i < len(lst):
                    out_rr.append(lst[i])
                    added = True
                    if total_limit and total_limit > 0 and len(out_rr) >= total_limit:
                        return out_rr
            if not added:
                break
            i += 1
        return out_rr

    for (kind, entkey), g in groups.items():
        desc = None
        cause = None
        details = {}
        if "target" in g:
            any_src_ns = next(iter(g.get("sources", {(None, None)})))[0]
            ts = target_status(g["target"], any_src_ns)
            desc = ts["description"]; cause = ts["cause"]; details = ts["details"]

        pods_set = {f"{it['ns']}/{it['pod']}" for it in g["items"]}
        pods_txt, n_pods = _fmt_pod_list(pods_set)
        entity_nice = g["entity"]
        if entity_nice.startswith("ip:"): entity_nice = entity_nice[3:]
        if entity_nice.startswith("service:"): entity_nice = entity_nice[8:]
        if entity_nice.startswith("host:"): entity_nice = entity_nice[5:]
        v_est = _verb_agree(n_pods, singular="está", plural="estão")

        if kind == "timeout":
            summary = f"{pods_txt} {v_est} com timeout no mesmo alvo {entity_nice}"
        elif kind == "refused":
            summary = f"{pods_txt} {v_est} recebendo conexão recusada no mesmo alvo {entity_nice}"
        elif kind == "dns":
            summary = f"{pods_txt} {v_est} com falha de DNS para {entity_nice}"
        elif kind == "probe":
            summary = f"{pods_txt} {v_est} com falhas de probe relacionadas ao mesmo alvo {entity_nice}"
        elif kind == "imagepull":
            summary = f"{pods_txt} {v_est} com problema de pull de imagem relacionado a {entity_nice}"
        elif kind == "oom":
            summary = f"{pods_txt} {v_est} com OOM envolvendo {entity_nice}"
        elif kind == "permission":
            summary = f"{pods_txt} {v_est} com falha de permissão relacionada a {entity_nice}"
        else:
            summary = f"{pods_txt} {v_est} com {kind} envolvendo {entity_nice}"

        if cause == "no_endpoints":
            summary += " • possivelmente porque o Service está **sem Endpoints**"
        elif cause == "pods_unhealthy":
            bad = details.get("unhealthy", [])
            if bad:
                btxt = ", ".join([f"{x['ns']}/{x['pod']}({x['phase']},{'ready' if x['ready'] else 'not-ready'})" for x in bad[:3]])
                summary += f" • possivelmente porque os pods de destino estão **não prontos**: {btxt}"
        elif cause == "pod_down":
            dpod = details.get("pod")
            if dpod:
                summary += f" • possivelmente porque o destino **{dpod}** está inativo/não pronto"

        items_full = g["items"]
        items_balanced = items_full if (not item_limit or item_limit <= 0) else _balanced_slice_by_pod(items_full, item_limit, per_pod_limit)

        out.append({
            "type": kind,
            "entity": g["entity"],
            "summary": summary,
            "enrichment": {"description": desc, "cause": cause, "details": details},
            "templates": [[tpl, cnt] for tpl, cnt in g["tmpl"].most_common(10)],
            "items": items_balanced,
            "pods": sorted(pods_set),
        })

    for key, g in perm_groups.items():
        pods_set = {f"{it['ns']}/{it['pod']}" for it in g["items"]}
        pods_txt, n_pods = _fmt_pod_list(pods_set)
        v_est = _verb_agree(n_pods, singular="está", plural="estão")

        if "verb" in g:
            summary = f"{pods_txt} {v_est} com **falha de permissão** no recurso **{g['resource']}** (verbo: {g['verb']})"
            ent = f"rbac:{g['verb']}:{g['resource']}"
        else:
            summary = f"{pods_txt} {v_est} com **falha de permissão** no caminho **{g['path']}**"
            ent = f"path:{g['path']}"

        items_full = g["items"]
        items_balanced = items_full if (not item_limit or item_limit <= 0) else _balanced_slice_by_pod(items_full, item_limit, per_pod_limit)

        out.append({
            "type": "permission",
            "entity": ent,
            "summary": summary,
            "enrichment": {},
            "templates": [[tpl, cnt] for tpl, cnt in g["tmpl"].most_common(10)],
            "items": items_balanced,
            "pods": sorted(pods_set),
        })

    out.sort(key=lambda x: (-len(x["items"]), x["type"], x["entity"]))

    if (not item_limit or item_limit <= 0) and (not per_pod_limit or per_pod_limit <= 0):
        key = ("groups", tuple(sorted(ns_filter or [])), int(limit_pods), int(tail_lines))
        CACHE_GROUPS.set(key, out)

    return out

# ===========================
# Flask app & rotas
# ===========================
app = Flask(__name__, static_folder="static", template_folder="templates")
Compress(app)  # ← gzip/brotli nas respostas

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/analysis")
def analysis_page():
    return render_template("analysis.html")

@app.route("/ml-errors")
def ml_errors_page():
    return render_template("ml_errors.html")

# ---------- API: pods ----------
@app.route("/api/pods")
def api_pods():
    q = (request.args.get("q") or "").strip()
    limit = int(request.args.get("limit", "200"))
    filters, terms = _parse_query(q)
    pods = v1.list_pod_for_all_namespaces().items
    items = []
    for p in pods:
        ok, score = _rank_pod(p, filters, terms)
        if not ok:
            continue
        meta, stat = p.metadata, p.status
        items.append({
            "name": meta.name,
            "namespace": meta.namespace,
            "phase": stat.phase,
            "age": _age_str(getattr(stat, "start_time", None)),
            "restarts": sum((cs.restart_count or 0) for cs in (stat.container_statuses or [])),
            "labels": meta.labels or {},
            "score": float(score),
        })
    items.sort(key=lambda x: (-x["score"], x["namespace"], x["name"]))
    return jsonify({"pods": items[:limit]})

@app.route("/api/pod/<ns>/<pod>")
def api_pod(ns, pod):
    logs = collect_logs_cached(ns, pod)
    events = collect_pod_events_cached(ns, pod)
    basic = get_pod_basic(ns, pod)
    if not basic:
        return jsonify({"error": "Pod not found"}), 404
    matches = find_matching_solutions(DATASET, logs, events, ns=ns, pod=pod)
    p_obj = get_pod(ns, pod)
    services = find_services_for_pod(ns, p_obj) if p_obj else []
    service_names = [s["name"] for s in services]
    endpoints = get_endpoints_for_services(ns, service_names)
    ingresses = get_ingresses_for_services(ns, service_names)
    pvcs = get_pvcs_for_pod(ns, p_obj) if p_obj else []
    node = get_node_status(basic.get("node_name"))

    try:
        maybe_save_last_finalized_snapshot(ns, pod, tail=800)
    except Exception as e:
        print("Failed to write last-run snapshot:", e)

    return jsonify({
        "basic": basic,
        "logs": logs,
        "events": events,
        "services": services,
        "endpoints": endpoints,
        "ingresses": ingresses,
        "pvcs": pvcs,
        "node": node,
        "solutions": matches
    })

# ---------- API última execução ----------
@app.route("/api/pod/<ns>/<pod>/last-run")
def api_pod_last_run(ns, pod):
    tail = int(request.args.get("tail", "800"))
    live = collect_last_run(ns, pod, tail=tail)
    events = collect_pod_events_cached(ns, pod)
    live["events"] = events

    snap = read_last_snapshot(ns, pod)
    has_snap = snap is not None
    return jsonify({
        "live": live,
        "snapshot_available": has_snap,
        "snapshot_meta": {
            "saved_at": snap.get("saved_at") if has_snap else None,
            "last_finished_at": snap.get("last_finished_at") if has_snap else None,
        } if has_snap else None
    })

@app.route("/api/pod/<ns>/<pod>/last-run/snapshot")
def api_pod_last_run_snapshot(ns, pod):
    snap = read_last_snapshot(ns, pod)
    if not snap:
        return jsonify({"error": "Snapshot não encontrado"}), 404
    return jsonify(snap)

# ---------- Rotas do DATASET (CRUD básico) ----------
@app.route("/api/dataset/solutions", methods=["POST"])
def api_dataset_add_solution():
    """
    Body: { error: str, solution: str, patterns?: [str] }
    """
    global DATASET
    data = request.get_json(silent=True) or {}
    error = (data.get("error") or "").strip()
    solution = (data.get("solution") or "").strip()
    patterns = data.get("patterns") or []
    if not error or not solution:
        return jsonify({"error": "Campos 'error' e 'solution' são obrigatórios."}), 400
    if not isinstance(patterns, list):
        return jsonify({"error": "'patterns' deve ser uma lista de strings."}), 400
    item = {
        "error": error,
        "solution": solution,
        "patterns": [str(p) for p in patterns if str(p).strip()]
    }
    _ensure_dataset_ids_and_compile([item])
    DATASET.append(item)
    try:
        _save_dataset(DATASET)
    except Exception as e:
        return jsonify({"error": f"Falha ao salvar dataset: {e}"}), 500
    return jsonify({"ok": True, "id": item["id"]})

@app.route("/api/dataset/solutions/<sid>", methods=["PUT"])
def api_dataset_update_solution(sid):
    """
    Body: { error?: str, solution?: str, patterns?: [str] }
    """
    global DATASET
    sid = (sid or "").strip()
    data = request.get_json(silent=True) or {}
    found = None
    for it in DATASET:
        if it.get("id") == sid:
            found = it
            break
    if not found:
        return jsonify({"error": "Item não encontrado."}), 404

    if "error" in data:
        found["error"] = (data.get("error") or "").strip()
    if "solution" in data:
        found["solution"] = (data.get("solution") or "").strip()
    if "patterns" in data:
        pats = data.get("patterns")
        if pats is not None:
            if not isinstance(pats, list):
                return jsonify({"error": "'patterns' deve ser uma lista de strings."}), 400
            found["patterns"] = [str(p).strip() for p in pats if str(p).strip()]

    pats = found.get("patterns") or [found.get("error", "")]
    found["_compiled"] = [rx for rx in (_compile_pattern(p) for p in pats) if rx is not None]

    try:
        _save_dataset(DATASET)
    except Exception as e:
        return jsonify({"error": f"Falha ao salvar dataset: {e}"}), 500

    return jsonify({"ok": True})

# ---------- ML helpers / endpoints existentes ----------
def choose_k(n_docs: int, k_param: int | None, avg_sim: float | None = None) -> int:
    if n_docs <= 0:
        return 0
    if k_param and 1 <= k_param <= n_docs:
        return int(k_param)
    if n_docs <= 3:
        return 1
    if avg_sim is not None and avg_sim >= 0.75:
        return 1
    k = max(2, int(math.sqrt(max(2, n_docs)/2.0)))
    return max(1, min(k, 10, n_docs))

def top_terms_for_cluster(tfidf, labels, cluster_id, feature_names, topn=12):
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return []
    mean_vec = np.asarray(tfidf[idx].mean(axis=0)).ravel()
    top_idx = mean_vec.argsort()[::-1][:topn]
    return [feature_names[i] for i in top_idx if mean_vec[i] > 0 and not feature_names[i].startswith("tmpl_")]

# ---------- RESUMO /api/ml/summary ----------
def aggregate_error_stats(ns_list: list[str], limit_pods: int = 120, tail_lines: int = 600):
    groups = analyze_similar_errors(ns_list, limit_pods=limit_pods, tail_lines=tail_lines, item_limit=0, per_pod_limit=0)

    type_counts = Counter()
    pod_counts_overall = Counter()
    pods_by_type = defaultdict(Counter)

    for g in groups:
        gtype = g.get("type") or "unknown"
        items = g.get("items", [])
        type_counts[gtype] += len(items)

        g_pods = g.get("pods") or []
        for pod_key in g_pods:
            pod_counts_overall[pod_key] += 1
            pods_by_type[gtype][pod_key] += 1

    type_counts = dict(sorted(type_counts.items(), key=lambda x: (-x[1], x[0])))
    top_pods = pod_counts_overall.most_common(30)
    pods_by_type_sorted = {t: pods_by_type[t].most_common(300) for t in pods_by_type}

    return {
        "types": type_counts,
        "top_pods": top_pods,
        "pods_by_type": pods_by_type_sorted,
        "params": {"ns": ns_list, "limit_pods": limit_pods, "tail": tail_lines}
    }

@app.route("/api/ml/summary")
def api_ml_summary():
    key = ("summary", request.args.get("ns",""), request.args.get("limit_pods","120"), request.args.get("tail","600"))
    cached = CACHE_ERRORS.get(key)
    if cached: return jsonify(cached)

    ns_filter = (request.args.get("ns") or "").strip()
    ns_list = [x.strip() for x in ns_filter.split(",") if x.strip()] if ns_filter else []
    limit_pods = int(request.args.get("limit_pods", "120"))
    tail_lines = int(request.args.get("tail", "600"))

    data = aggregate_error_stats(ns_list, limit_pods=limit_pods, tail_lines=tail_lines)
    CACHE_ERRORS.set(key, data)
    return jsonify(data)

# ---------- Erros por tipo ----------
def _aggregate_errors_by_type(groups):
    by_type = {}
    for g in groups:
        t = g.get("type") or "unknown"
        it = by_type.setdefault(t, {"total": 0, "pods": defaultdict(int), "entities": Counter(), "examples": [], "templates": Counter()})

        items = g.get("items", [])
        it["total"] += len(items)

        g_pods = g.get("pods")
        if g_pods:
            for pod_key in g_pods:
                it["pods"][pod_key] += 1
        else:
            for itx in items:
                key = f"{itx['ns']}/{itx['pod']}"
                it["pods"][key] += 1

        for itx in items:
            if len(it["examples"]) < 12:
                ln = (itx.get("line") or "").strip()
                if ln:
                    it["examples"].append(ln)

        ent = g.get("entity") or ""
        if ent:
            it["entities"][ent] += len(items)
        for tpl, cnt in (g.get("templates") or []):
            it["templates"][tpl] += cnt

    for t, data in by_type.items():
        data["pods"] = dict(sorted(data["pods"].items(), key=lambda kv: (-kv[1], kv[0]))[:300])
        data["entities"] = [[k, v] for k, v in data["entities"].most_common(30)]
        seen = set()
        uniq = []
        for s in data["examples"]:
            s2 = (s or "").strip()
            if not s2 or s2 in seen:
                continue
            seen.add(s2)
            uniq.append(s2 if len(s2) <= 400 else (s2[:397] + "..."))
            if len(uniq) >= 12:
                break
        data["examples"] = uniq
        data["templates"] = [[tpl, cnt] for tpl, cnt in data["templates"].most_common(12)]

    return {"types": by_type}

@app.route("/healthz")
def healthz():
    return "ok", 200

@app.route("/api/ml/errors-by-type")
def api_ml_errors_by_type():
    key = ("errors-by-type", request.args.get("ns",""), request.args.get("limit_pods","120"), request.args.get("tail","600"))
    cached = CACHE_ERRORS.get(key)
    if cached: return jsonify(cached)

    ns_filter = (request.args.get("ns") or "").strip()
    ns_list = [x.strip() for x in ns_filter.split(",") if x.strip()] if ns_filter else []
    limit_pods = int(request.args.get("limit_pods", "120"))
    tail_lines = int(request.args.get("tail", "600"))

    groups = analyze_similar_errors(ns_list, limit_pods=limit_pods, tail_lines=tail_lines, item_limit=0, per_pod_limit=0)
    payload = _aggregate_errors_by_type(groups)
    out = { **payload, "params": {"ns": ns_list, "limit_pods": limit_pods, "tail": tail_lines} }
    CACHE_ERRORS.set(key, out)
    return jsonify(out)

# ---------- NOVO: lista bruta de grupos (para "Por entidade") ----------
@app.route("/api/ml/errors")
def api_ml_errors():
    ns_filter = (request.args.get("ns") or "").strip()
    ns_list = [x.strip() for x in ns_filter.split(",") if x.strip()] if ns_filter else []
    limit_pods = int(request.args.get("limit_pods", "120"))
    tail_lines = int(request.args.get("tail", "600"))

    item_limit = int(request.args.get("item_limit", "0"))
    per_pod_limit = int(request.args.get("per_pod_limit", "0"))

    groups = analyze_similar_errors(
        ns_list,
        limit_pods=limit_pods,
        tail_lines=tail_lines,
        item_limit=item_limit,
        per_pod_limit=per_pod_limit
    )
    return jsonify({
        "groups": groups,
        "params": {"ns": ns_list, "limit_pods": limit_pods, "tail": tail_lines, "item_limit": item_limit, "per_pod_limit": per_pod_limit}
    })

# ---------- Projeção 2D ----------
def _cluster_projection_2d(docs, use_svd: bool, X=None, vectorizer=None):
    if X is None or vectorizer is None:
        vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=60000, min_df=1)
        X = vectorizer.fit_transform(docs)

    n_samples = X.shape[0]
    if n_samples <= 1:
        coords2d = np.zeros((n_samples, 2), dtype=float)
        X_model = X
        return coords2d, X_model, vectorizer, None

    svd = None
    if use_svd and X.shape[1] > 3 and n_samples >= 3:
        n_comp = min(max(2, n_samples - 1), max(2, min(200, X.shape[1]-1)))
        svd = TruncatedSVD(n_components=n_comp, random_state=0)
        X_red = svd.fit_transform(X)
        coords2d = X_red[:, :2] if X_red.shape[1] >= 2 else np.pad(X_red, ((0,0),(0, 2-X_red.shape[1])), mode='constant')
        X_model = normalize(X_red)
    else:
        if X.shape[1] > 2 and n_samples >= 2:
            svd = TruncatedSVD(n_components=2, random_state=0)
            coords2d = svd.fit_transform(X)
        else:
            coords2d = X.toarray() if hasattr(X, "toarray") else X
            if coords2d.ndim == 1:
                coords2d = np.stack([coords2d, np.zeros_like(coords2d)], axis=1)
            if coords2d.shape[1] == 1:
                coords2d = np.hstack([coords2d, np.zeros((coords2d.shape[0],1))])
        X_model = X
    return coords2d, X_model, vectorizer, svd

def _filter_text_by_type(raw_text: str, err_type: str) -> str:
    if not err_type:
        return raw_text
    lines = []
    for ln in (raw_text or "").splitlines():
        if _error_kind(ln) == err_type:
            lines.append(ln)
    return "\n".join(lines)

# ---------- /api/cluster-2d ----------
@app.route("/api/cluster-2d")
def api_cluster_2d():
    ns_filter = (request.args.get("ns") or "").strip()
    ns_list = [x.strip() for x in ns_filter.split(",") if x.strip()] if ns_filter else []
    limit_pods = int(request.args.get("limit_pods", "80"))
    tail_lines = int(request.args.get("tail", "400"))
    k_param    = request.args.get("k")
    k_param    = int(k_param) if k_param and k_param.isdigit() else None
    use_svd    = (request.args.get("svd", "1") != "0")
    err_type   = (request.args.get("type") or "").strip().lower() or None

    all_pods = v1.list_pod_for_all_namespaces().items
    pods = [p for p in all_pods if (not ns_list or p.metadata.namespace in ns_list)]
    pods = pods[:limit_pods]

    def _types_from_text(text: str) -> Counter:
        c = Counter()
        for ln in (text or "").splitlines():
            kind = _error_kind(ln)
            if kind:
                c[kind] += 1
        if not c:
            c["unknown"] += 1
        return c

    docs, meta = [], []

    futures = [EXEC.submit(_fetch_logs_events, p.metadata.namespace, p.metadata.name, tail_lines) for p in pods]
    for fut in as_completed(futures):
        ns, name, logs, events = fut.result()

        raw_text = (logs or "") + "\n" + "\n".join(events or [])
        text_for_type = _filter_text_by_type(raw_text, err_type) if err_type else raw_text
        if err_type and not text_for_type.strip():
            continue

        norm = normalize_text(text_for_type)
        norm = strip_identifiers(norm, ns, name)

        types = _types_from_text(text_for_type if err_type else raw_text)
        dom_type = err_type if err_type else max(types.items(), key=lambda kv: kv[1])[0]

        tmpl_counter = extract_templates_from_text(text_for_type)
        tmpl_tokens = " ".join((template_token(tpl) + " ") * min(cnt, 5) for tpl, cnt in tmpl_counter.items())

        doc = (norm + " " + tmpl_tokens).strip()
        docs.append(doc)
        meta.append({
            "ns": ns,
            "pod": name,
            "phase": get_pod_basic(ns, name).get("phase") if get_pod_basic(ns, name) else "Unknown",
            "restarts": int(sum((cs.restart_count or 0) for cs in (get_pod(ns, name).status.container_statuses or []))) if get_pod(ns, name) else 0,
            "tags": quick_tags(norm),
            "types": dict(types),
            "dom_type": dom_type,
            "templates": tmpl_counter,
        })

    n_docs = len(docs)
    if n_docs == 0:
        payload = {"points": [], "clusters": [], "summary": {"n_docs": 0}}
        if err_type: payload["summary"]["err_type"] = err_type
        return jsonify(payload)

    vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=60000, min_df=1)
    X = vectorizer.fit_transform(docs)

    avg_sim = None
    if n_docs >= 2:
        S = cosine_similarity(X)
        mask = ~np.eye(n_docs, dtype=bool)
        if mask.sum() > 0:
            avg_sim = float(S[mask].mean())

    coords2d, X_model, _, _ = _cluster_projection_2d(docs, use_svd=use_svd, X=X, vectorizer=vectorizer)

    if n_docs == 1:
        points = [{
            "x": float(coords2d[0,0]),
            "y": float(coords2d[0,1]),
            "cluster": 0,
            "ns": meta[0]["ns"],
            "pod": meta[0]["pod"],
            "label": f"{meta[0]['ns']}/{meta[0]['pod']}",
            "phase": meta[0]["phase"],
            "restarts": int(meta[0]["restarts"]),
            "tags": meta[0]["tags"],
            "dom_type": meta[0]["dom_type"],
            "distance": 0.0,
            "outlier": False,
        }]
        feature_names = np.array(vectorizer.get_feature_names_out())
        xi = X.toarray()[0]
        top_idx = xi.argsort()[::-1][:8]
        terms = [feature_names[i] for i in top_idx if not feature_names[i].startswith("tmpl_")]
        tmpl_c = meta[0]["templates"]
        top_templates = [[tpl, cnt] for tpl, cnt in tmpl_c.most_common(8)]
        types_agg = list(Counter(meta[0]["types"]).most_common(6))
        tags_agg = list(Counter(meta[0]["tags"]).most_common(8))
        clusters_meta = [{
            "cluster": 0,
            "size": 1,
            "top_terms": terms,
            "tags": tags_agg,
            "types": types_agg,
            "top_templates": top_templates,
            "top_members": [{
                "ns": meta[0]["ns"], "pod": meta[0]["pod"], "restarts": meta[0]["restarts"],
                "tags": meta[0]["tags"], "dom_type": meta[0]["dom_type"]
            }]
        }]
        summary = {"n_docs": 1, "k": 1, "outlier_threshold": 0.0}
        if err_type: summary["err_type"] = err_type
        return jsonify({"points": points, "clusters": clusters_meta, "summary": summary})

    k = choose_k(n_docs, k_param, avg_sim=avg_sim)
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=10, batch_size=1024)
    y = km.fit_predict(X_model)

    centers = km.cluster_centers_
    dists = np.empty(n_docs, dtype=float)
    for i in range(n_docs):
        c = y[i]
        xi = X_model[i]
        if hasattr(xi, "toarray"):
            xi = xi.toarray().ravel()
        dists[i] = np.linalg.norm(xi - centers[c])
    thr = float(np.percentile(dists, 95))
    outliers = dists > thr

    points = []
    for i in range(n_docs):
        points.append({
            "x": float(coords2d[i, 0]),
            "y": float(coords2d[i, 1]),
            "cluster": int(y[i]),
            "ns": meta[i]["ns"],
            "pod": meta[i]["pod"],
            "label": f"{meta[i]['ns']}/{meta[i]['pod']}",
            "phase": meta[i]["phase"],
            "restarts": int(meta[i]["restarts"]),
            "tags": meta[i]["tags"],
            "dom_type": meta[i]["dom_type"],
            "distance": float(dists[i]),
            "outlier": bool(outliers[i]),
        })

    feature_names = np.array(vectorizer.get_feature_names_out())
    clusters_meta = []
    for cid in sorted(set(y)):
        idx = np.where(y == cid)[0]
        size = int(len(idx))

        def top_terms_for_idx(idxs, topn=8):
            mean_vec = np.asarray(X[idxs].mean(axis=0)).ravel()
            top_idx = mean_vec.argsort()[::-1][:topn]
            return [feature_names[i] for i in top_idx if mean_vec[i] > 0 and not feature_names[i].startswith("tmpl_")]

        terms = top_terms_for_idx(idx, topn=8)

        tags_c = Counter()
        for i in idx:
            tags_c.update(meta[i]["tags"])
        tags_agg = tags_c.most_common(8)

        types_c = Counter()
        for i in idx:
            types_c.update(meta[i]["types"])
        types_agg = types_c.most_common(6)

        tmpl_c = Counter()
        for i in idx:
            tmpl_c.update(meta[i]["templates"])
        top_templates = tmpl_c.most_common(8)

        members = sorted(
            (
                {
                    "ns": meta[i]["ns"],
                    "pod": meta[i]["pod"],
                    "restarts": meta[i]["restarts"],
                    "tags": meta[i]["tags"],
                    "dom_type": meta[i]["dom_type"],
                }
                for i in idx
            ),
            key=lambda m: (-m["restarts"], m["ns"], m["pod"])
        )[:12]

        clusters_meta.append({
            "cluster": int(cid),
            "size": size,
            "top_terms": terms,
            "tags": tags_agg,
            "types": types_agg,
            "top_templates": [[tpl, cnt] for tpl, cnt in top_templates],
            "top_members": members,
        })

    summary = {"n_docs": n_docs, "k": int(k), "outlier_threshold": thr}
    if err_type:
        summary["err_type"] = err_type
    if avg_sim is not None:
        summary["avg_cosine_similarity"] = avg_sim

    return jsonify({
        "points": points,
        "clusters": clusters_meta,
        "summary": summary
    })

# ---- Networking / storage helpers ----
def find_services_for_pod(ns: str, pod_obj):
    if not pod_obj: return []
    svcs = v1.list_namespaced_service(ns).items
    pod_labels = pod_obj.metadata.labels or {}
    matched = []
    for s in svcs:
        selector = s.spec.selector or {}
        if selector and all(pod_labels.get(k) == v for k, v in selector.items()):
            matched.append({
                "name": s.metadata.name,
                "type": s.spec.type,
                "cluster_ip": s.spec.cluster_ip,
                "ports": [{
                    "port": p.port, "targetPort": p.target_port, "nodePort": getattr(p, "node_port", None), "protocol": p.protocol
                } for p in (s.spec.ports or [])]
            })
    return matched

def get_endpoints_for_services(ns: str, service_names: list[str]):
    if not service_names: return []
    eps = v1.list_namespaced_endpoints(ns).items
    out = []
    for e in eps:
        if e.metadata.name in service_names:
            subsets = []
            for ss in (e.subsets or []):
                addresses = [{"ip": a.ip} for a in (ss.addresses or [])]
                not_ready = [{"ip": a.ip} for a in (ss.not_ready_addresses or [])]
                ports = [{"name": p.name, "port": p.port, "protocol": p.protocol} for p in (ss.ports or [])]
                subsets.append({"addresses": addresses, "not_ready": not_ready, "ports": ports})
            out.append({"service": e.metadata.name, "subsets": subsets})
    return out

def get_ingresses_for_services(ns: str, service_names: list[str]):
    if not service_names: return []
    try:
        ings = networking.list_namespaced_ingress(ns).items
    except ApiException:
        return []
    out = []
    for ing in ings:
        matched_rules = []
        if ing.spec and ing.spec.rules:
            for r in ing.spec.rules:
                if not (r.http and r.http.paths): continue
                paths = []
                for p in r.http.paths:
                    backend = getattr(p.backend, "service", None)
                    if backend and backend.name in service_names:
                        port = getattr(backend.port, "number", None) or getattr(backend.port, "name", None)
                        paths.append({"path": p.path, "pathType": p.path_type, "service": backend.name, "port": port})
                if paths:
                    matched_rules.append({"host": r.host, "paths": paths})
        if matched_rules:
            out.append({"name": ing.metadata.name, "rules": matched_rules})
    return out

def get_pvcs_for_pod(ns: str, pod_obj):
    if not pod_obj:
        return []
    pvcs = []
    vols = pod_obj.spec.volumes or []
    for v in vols:
        if v.persistent_volume_claim:
            claim = v.persistent_volume_claim.claim_name
            try:
                pvc = v1.read_namespaced_persistent_volume_claim(claim, ns)
                pv = None
                vol_name = pvc.spec.volume_name if pvc.spec else None
                if vol_name:
                    pv = v1.read_persistent_volume(vol_name)
                pvcs.append({
                    "claim": claim,
                    "status": pvc.status.phase if pvc.status else None,
                    "storage_class": pvc.spec.storage_class_name if pvc.spec else None,
                    "capacity": (pvc.status.capacity or {}).get("storage") if pvc.status and pvc.status.capacity else None,
                    "volume": vol_name,
                    "pv_reclaim_policy": (pv.spec.persistent_volume_reclaim_policy if pv and pv.spec else None)
                })
            except ApiException:
                pvcs.append({"claim": claim, "status": "UNKNOWN"})
    return pvcs

def get_node_status(node_name: str):
    if not node_name: return {}
    try:
        node = v1.read_node(node_name)
    except ApiException:
        return {}
    addresses = [{"type": a.type, "address": a.address} for a in (node.status.addresses or [])]
    conditions = [{"type": c.type, "status": c.status, "reason": getattr(c, "reason", "") or ""} for c in (node.status.conditions or [])]
    taints = [{"key": t.key, "value": t.value, "effect": t.effect} for t in (getattr(node.spec, "taints", []) or [])]
    return {"name": node.metadata.name, "addresses": addresses, "conditions": conditions, "taints": taints}

# =====================================================================
# ==================== SEÇÃO CHATBOT (USO DO DATASET RICO) ============
# =====================================================================

# Config do Chatbot
CHAT_MAX_SOURCES = int(os.environ.get('CHAT_MAX_SOURCES', '4'))

# (mantemos as flags, mas o chatbot agora funciona bem mesmo com USE_LLM=0)
USE_FAISS = True if os.environ.get("USE_FAISS", "1") != "0" else False
USE_RERANKER = False if os.environ.get("USE_RERANKER", "1") != "0" else False
USE_LLM = False if os.environ.get("USE_LLM", "1") != "0" else False

# Modelos (apenas se você quiser ligar o LLM futuramente)
LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'unicamp-dl/ptt5-base')
LLM_FALLBACK_CAUSAL_1 = os.environ.get('LLM_FALLBACK_CAUSAL_1', 'Qwen/Qwen2.5-0.5B-Instruct')
LLM_FALLBACK_CAUSAL_2 = os.environ.get('LLM_FALLBACK_CAUSAL_2', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
RERANKER_MODEL = os.environ.get('RERANKER_MODEL', 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

# Caminho do dataset rico do chatbot
CHAT_DATASET_PATH = os.environ.get("CHAT_DATASET_PATH", "dataset_chatbot.json")

import importlib
def _lazy_imports_chat():
    st = importlib.import_module("sentence_transformers")
    np_local = importlib.import_module("numpy")
    faiss = None
    if USE_FAISS:
        try:
            faiss = importlib.import_module("faiss")
        except Exception:
            faiss = None
    return st, np_local, faiss

_chat_lock = None
try:
    from threading import Lock as _Lock2
    _chat_lock = _Lock2()
except Exception:
    class _Noop:
        def __enter__(self): pass
        def __exit__(self, *a): pass
    _chat_lock = _Noop()

_index_ready = False
_index_error = None
_reranker = None
_llm_pipe = None
_llm_kind = None  # "seq2seq" ou "causal"

def _detect_lang(text: str) -> str:
    t = " " + (text or "").lower() + " "
    if any(m in t for m in [" não ", " erro ", " solução ", " permissão ", " falha ", " segredo ", "segredo "]) or re.search(r"[áéíóúâêôãõç]", t):
        return "pt"
    return "en"

# ---------------- Dataset rico: loader & busca ----------------

def _prep_chat_item(it: dict):
    """
    Enriquecimento do item do dataset_chatbot:
    - compila padrões (patterns) a partir de patterns/aliases/name/keywords
    - constrói _kw com tokens de keywords + aliases + name + guidance.* + quick_checks
    """
    # Derivar patterns caso não exista
    pats = it.get("patterns") or []
    if not pats:
        pats = []
        if it.get("name"): pats.append(it["name"])
        for a in (it.get("aliases") or []):
            pats.append(a)
        for kw in ((it.get("signals",{}) or {}).get("log_keywords") or []):
            pats.append(kw)
    it["_compiled"] = [rx for rx in (_compile_pattern(p) for p in pats) if rx is not None]

    # tokens p/ keywords/aliases/name + guidance + quick_checks + remediations + questions_to_ask
    kw_tokens = set()
    def _add_tokens(seq):
        for k in seq or []:
            for t in re.findall(r"[a-z0-9][a-z0-9._/-]{1,}", (k or "").lower()):
                if len(t) >= 2:
                    kw_tokens.add(t)

    signals_kw = (it.get("signals",{}) or {}).get("log_keywords") or []
    gd = it.get("guidance") or {}
    _add_tokens(signals_kw)
    _add_tokens(it.get("aliases") or [])
    _add_tokens([it.get("name","")])
    _add_tokens(gd.get("common_causes") or [])
    _add_tokens(gd.get("immediate_actions") or [])
    _add_tokens(gd.get("corrections") or [])
    _add_tokens(it.get("quick_checks") or [])
    _add_tokens(it.get("remediations") or [])
    _add_tokens(it.get("questions_to_ask") or [])

    it["_kw"] = kw_tokens
    return it


def _load_chat_dataset(path=CHAT_DATASET_PATH):
    """
    Carrega dataset_chatbot.json como lista ou {"items":[...]} e
    monta _all_text abrangendo guidance/common_causes, immediate_actions, corrections,
    além de quick_checks, remediations, questions_to_ask e observability queries.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else (data.get("items") or [])
        if not isinstance(items, list):
            raise ValueError("dataset_chatbot.json deve ser uma lista ou um objeto com 'items' (lista).")

        for it in items:
            it["name"] = (it.get("name") or "").strip()
            it["slug"] = (it.get("slug") or "").strip().lower()
            it["aliases"] = [a.strip() for a in (it.get("aliases") or []) if a and str(a).strip()]

            gd = (it.get("guidance") or {})
            obs = (it.get("deep_dive") or {}).get("observability_queries", []) or []
            # Texto agregado para ranking/recall
            it["_all_text"] = " ".join([
                it.get("name",""),
                it.get("overview",""),
                " ".join(it.get("aliases",[])),
                " ".join(gd.get("common_causes",[]) or []),
                " ".join(gd.get("immediate_actions",[]) or []),
                " ".join(gd.get("corrections",[]) or []),
                " ".join(it.get("remediations") or []),
                " ".join(it.get("quick_checks") or []),
                " ".join(it.get("questions_to_ask") or []),
                " ".join([q.get("expr","") for q in obs if isinstance(q, dict)])
            ]).lower()

            _prep_chat_item(it)

        return items
    except Exception as e:
        print("[chatbot] falha ao carregar dataset:", e)
        return []


CHAT_DATASET = _load_chat_dataset()
print(f"[chatbot] itens carregados: {len(CHAT_DATASET)}")

def _tokenize_query(q: str):
    toks = re.findall(r"[a-z0-9][a-z0-9._/-]{1,}", (q or "").lower())
    stop = {"the","and","com","http","https","uma","num","nos","nas","dos","das","com","por","não","sim","que","de","do","da"}
    return [t for t in toks if t not in stop]

def _expand_synonyms(q_toks):
    """
    Sinônimos simples para melhorar recall (sem depender de contexto do pod).
    """
    syn = {
        "oom": ["oomkilled","out","memory","outofmemory","signal:killed","out-of-memory"],
        "crashloopbackoff": ["crashloop","crash","back-off","backoff"],
        "imagepull": ["imagepullbackoff","errimagepull","manifest","unauthorized","pull image","image pull"],
        "tls": ["x509","certificate","unknown authority","handshake"],
        "dns": ["no such host","name resolution","dns"],
        "refused": ["connection refused","econnrefused"],
        "timeout": ["i/o timeout","deadline exceeded","timed out","timeout"],
        "permission": ["forbidden","unauthorized","permission denied","rbac"],
        "probe": ["liveness","readiness","startup","probe failed"],
        "throttle": ["throttl","rate limit","429"],
        "storage": ["enospc","disk full","read-only file system","ephemeral-storage"],
    }
    out = set(q_toks)
    for t in list(q_toks):
        for k, vs in syn.items():
            if t == k or t in vs:
                out.add(k)
                out.update(vs)
    return list(out)

def _search_chat(question: str, k: int = 4):
    """
    Busca com boosts por slug/aliases/keywords/texto + regex patterns.
    Retorna também 'matched_by' para diagnóstico.
    """
    q = (question or "").strip().lower()
    q_toks = _expand_synonyms(_tokenize_query(q))

    scored = []
    for it in CHAT_DATASET:
        sc = 0.0
        matched_by = set()

        # slug
        if it.get("slug") and it["slug"] in q:
            sc += 7.0; matched_by.add("slug")

        # aliases
        for al in (it.get("aliases") or []):
            al_l = (al or "").lower()
            if al_l and al_l in q:
                sc += 6.0; matched_by.add("alias")

        # keywords tokenizadas (aliases + name + signals keywords)
        kw = it.get("_kw") or set()
        hit_kw = sum(1 for t in q_toks if t in kw)
        if hit_kw:
            sc += 2.0 * hit_kw
            matched_by.add("keywords")

        # texto agregado
        at = it.get("_all_text","")
        sc += sum(1.0 for t in q_toks if t in at)

        # padrões regex
        boost = 0.0
        for rx in (it.get("_compiled") or []):
            try:
                if rx.search(q):
                    boost += 3.0
            except Exception:
                pass
        if boost:
            sc += boost
            matched_by.add("patterns")

        if sc > 0:
            scored.append((sc, it, ",".join(sorted(matched_by)) or "—"))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    hits = scored[:max(k*4, k)]  # pega mais p/ robustez

    # reranker opcional
    if USE_RERANKER:
        try:
            ce = _get_reranker()
            pairs = [(q, (h[1].get("overview") or "") + " " + " ".join(h[1].get("remediations") or [])) for h in hits]
            scores = ce.predict(pairs)
            hits = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
            hits = [h[0] for h in hits][:k]
        except Exception:
            hits = hits[:k]
    else:
        hits = hits[:k]

    out = []
    for rank, (sc, it, how) in enumerate(hits, 1):
        out.append({"rank": rank, "score": float(round(sc,2)), "item": it, "matched_by": how})
    return out

def _apply_ctx_placeholders_list(lst, ns, pod):
    if not lst: return []
    out = []
    for s in lst:
        s2 = _apply_placeholders(str(s), ns, pod)
        out.append(s2)
    return out

def _dedup_bullets(items: list[str], maxn: int | None = None) -> list[str]:
    """
    Deduplica mantendo a ordem e preferindo a descrição mais longa quando
    o 'comando base' é o mesmo (ex.: 'kubectl describe pod ...' aparece
    tanto como ação imediata quanto como check rápido).
    """
    if not items:
        return []
    order = []
    best = {}

    def norm_key(s: str) -> str:
        x = (s or "").strip()
        # remove complementos entre parênteses e notas após '—'
        x = re.sub(r"\s*\([^)]*\)", "", x)
        x = re.sub(r"\s*—.*$", "", x)
        # normaliza múltiplos espaços
        x = re.sub(r"\s+", " ", x)
        return x.lower()

    for s in items:
        k = norm_key(s)
        if k not in best:
            best[k] = s
            order.append(k)
        else:
            # mantém a versão mais informativa (string mais longa)
            if len(s) > len(best[k]):
                best[k] = s

    out = [best[k] for k in order]
    return out if maxn is None else out[:maxn]


def _render_chat_answer(it: dict, ns: str | None, pod: str | None, lang: str) -> str:
    """
    Renderiza NA MESMA ORDEM do dataset, em PT/EN, sem mesclar listas.
    Só mostra se houver conteúdo em cada seção.
    """
    name = it.get("name") or it.get("slug") or "Assunto"
    overview = _apply_placeholders(it.get("overview") or "", ns, pod)
    sig = (it.get("signals") or {})
    gd  = (it.get("guidance") or {})
    dd  = (it.get("deep_dive") or {})

    # helpers
    def bullets(arr):
        arr = _apply_ctx_placeholders_list(arr or [], ns, pod)
        return None if not arr else "\n".join(f"• {x}" for x in arr)

    def obs_lines(obs_items):
        out = []
        for o in (obs_items or []):
            title = _apply_placeholders(o.get("title", "Query"), ns, pod)
            expr  = _apply_placeholders(o.get("expr", ""), ns, pod)
            out.append(f"{title}: {expr}" if expr else title)
        return None if not out else "\n".join(f"• {x}" for x in out)

    # títulos por idioma (mesma ordem do dataset)
    if lang == "pt":
        titles = {
            "overview": f"**O que é {name}?**",
            "signals": "**Sinais / Palavras-chave**",
            "common_causes": "**Causas comuns**",
            "immediate_actions": "**Ações imediatas**",
            "best_practices": "**Boas práticas**",
            "storage_diagnostics": "**Diagnóstico de storage**",
            "corrections": "**Correções**",
            "validation": "**Validação**",
            "quick_checks": "**Quick checks**",
            "k8s_commands": "**Deep dive — Comandos Kubernetes**",
            "observability": "**Deep dive — Queries de observabilidade**",
            "remediations": "**Remediações**",
            "questions_to_ask": "**Perguntas para investigar**",
            "faq": "**FAQ**",
        }
    else:
        titles = {
            "overview": f"**What is {name}?**",
            "signals": "**Signals / Keywords**",
            "common_causes": "**Common causes**",
            "immediate_actions": "**Immediate actions**",
            "best_practices": "**Best practices**",
            "storage_diagnostics": "**Storage diagnostics**",
            "corrections": "**Corrections**",
            "validation": "**Validation**",
            "quick_checks": "**Quick checks**",
            "k8s_commands": "**Deep dive — Kubernetes commands**",
            "observability": "**Deep dive — Observability queries**",
            "remediations": "**Remediations**",
            "questions_to_ask": "**Questions to ask**",
            "faq": "**FAQ**",
        }

    parts = []

    # 1) overview
    if overview.strip():
        parts += [titles["overview"], overview.strip(), ""]

    # 2) signals.log_keywords
    sig_kw = bullets((sig.get("log_keywords") or []))
    if sig_kw:
        parts += [titles["signals"], sig_kw, ""]

    # 3) guidance.* (na ordem exata)
    cc = bullets(gd.get("common_causes"))
    if cc: parts += [titles["common_causes"], cc, ""]
    ia = bullets(gd.get("immediate_actions"))
    if ia: parts += [titles["immediate_actions"], ia, ""]
    bp = bullets(gd.get("best_practices"))
    if bp: parts += [titles["best_practices"], bp, ""]
    sd = bullets(gd.get("storage_diagnostics"))
    if sd: parts += [titles["storage_diagnostics"], sd, ""]
    cr = bullets(gd.get("corrections"))
    if cr: parts += [titles["corrections"], cr, ""]
    va = bullets(gd.get("validation"))
    if va: parts += [titles["validation"], va, ""]

    # 4) quick_checks
    qc = bullets(it.get("quick_checks"))
    if qc: parts += [titles["quick_checks"], qc, ""]

    # 5) deep_dive (k8s_commands → observability_queries)
    k8s_cmds = bullets(dd.get("k8s_commands"))
    if k8s_cmds: parts += [titles["k8s_commands"], k8s_cmds, ""]
    obs = obs_lines(dd.get("observability_queries"))
    if obs: parts += [titles["observability"], obs, ""]

    # 6) remediations
    rm = bullets(it.get("remediations"))
    if rm: parts += [titles["remediations"], rm, ""]

    # 7) questions_to_ask
    qta = bullets(it.get("questions_to_ask"))
    if qta: parts += [titles["questions_to_ask"], qta, ""]

    # 8) FAQ
    faq_items = it.get("faq") or []
    if faq_items:
        parts.append(titles["faq"])
        for qa in faq_items:
            q = _apply_placeholders((qa.get("q") or "").strip(), ns, pod)
            a = _apply_placeholders((qa.get("a") or "").strip(), ns, pod)
            if not (q or a): 
                continue
            if lang == "pt":
                parts.append(f"• **Pergunta:** {q}\n  **Resposta:** {a}")
            else:
                parts.append(f"• **Question:** {q}\n  **Answer:** {a}")
        parts.append("")

    # string final
    out = "\n".join(parts).strip()
    return out if out else (overview.strip() or name)



# --------- (Mantemos utilidades de LLM caso ative no futuro) ----------
def _compose_prompt(question: str, hits: list[dict], lang: str) -> str:
    # Sem contexto de pod; apenas fontes
    context_blocks = []
    for i, h in enumerate(hits, 1):
        context_blocks.append(f"[{i}] ERROR: {h.get('error','')}\nSOLUTION: {h.get('solution','')}")
    ctx = "\n\n".join(context_blocks)
    if lang == "pt":
        instruction = (
            "Você é um assistente SRE/Kubernetes. Use APENAS as fontes abaixo.\n"
            "Responda em 6-10 linhas no formato:\n"
            "O que significa • Como confirmar (2-3 comandos) • Como corrigir (3-5 passos) • Fontes [n].\n"
            "Se as fontes forem insuficientes, diga isso claramente.\n\n"
        )
        q = f"Pergunta: {question}\n\nFontes:\n{ctx}\n\nResposta:"
    else:
        instruction = (
            "You are an SRE/Kubernetes assistant. Use ONLY the sources below.\n"
            "Answer in 6-10 lines as: What it means • How to confirm (2-3 cmds) • How to fix (3-5 steps) • Sources [n].\n"
            "If sources are insufficient, say so.\n\n"
        )
        q = f"Question: {question}\n\nSources:\n{ctx}\n\nAnswer:"
    return instruction + q

def _get_llm():
    global _llm_pipe, _llm_kind
    if not USE_LLM:
        return None
    if _llm_pipe is not None:
        return _llm_pipe
    try:
        if 'ptt5' in (LLM_MODEL_NAME or '').lower():
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            tok = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=False)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
            _llm_pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=-1)
            _llm_kind = "seq2seq"
            return _llm_pipe
    except Exception:
        _llm_pipe = None; _llm_kind = None
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch
        tok = AutoTokenizer.from_pretrained(LLM_FALLBACK_CAUSAL_1, use_fast=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            LLM_FALLBACK_CAUSAL_1,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=None
        )
        _llm_pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device=-1)
        _llm_kind = "causal"
        return _llm_pipe
    except Exception:
        _llm_pipe = None; _llm_kind = None
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch
        tok = AutoTokenizer.from_pretrained(LLM_FALLBACK_CAUSAL_2, use_fast=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            LLM_FALLBACK_CAUSAL_2,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=None
        )
        _llm_pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device=-1)
        _llm_kind = "causal"
        return _llm_pipe
    except Exception:
        _llm_pipe = None; _llm_kind = None
        return None

def _get_reranker():
    global _reranker
    if not USE_RERANKER:
        return None
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

# -------- Dataset legado (para fallback) --------
def _rewrite_query(q: str) -> str:
    syn = {
        "configmap": ["config map", "cm"],
        "secret": ["secrets"],
        "ingress": ["ingressroute", "gateway"],
        "deployment": ["deploy"],
        "statefulset": ["sts"],
        "daemonset": ["ds"],
        "pod": ["pods"],
        "imagepull": ["image pull", "pull image"],
        "permission": ["permissão", "access denied", "forbidden", "rbac"],
        "liveness": ["liveness probe", "probe liveness"],
        "readiness": ["readiness probe", "probe readiness"],
    }
    extra = []
    ql = " " + (q or "").lower() + " "
    for k, syns in syn.items():
        if any(s in ql for s in [k] + syns):
            extra += [k] + syns
    extra = " ".join(sorted(set(extra)))
    return (q or "").strip() + (" " + extra if extra else "")

def _build_index_if_needed(force: bool = False):
    # mantemos para compat (não é necessário pro fluxo determinístico)
    return True

def _cosine_topk(qvec, embs, k=5):
    import numpy as _np
    q = qvec / (_np.linalg.norm(qvec) + 1e-9)
    sims = embs @ q
    idx = sims.argsort()[-k:][::-1]
    return idx, sims[idx]

def _search(question: str, context_text: str = "", k: int = 4):
    q = (question or "").strip()
    q_ext = _rewrite_query(q)
    haystack = (q_ext).lower()

    stop = {"the","and","com","http","https","que","para","uma","num","nos","nas","dos","das","com","por","não","sim"}
    tokens = [t for t in re.findall(r"\w{3,}", q_ext.lower()) if t not in stop]

    def overlap_score(text: str) -> float:
        if not tokens:
            return 0.0
        t = (text or "").lower()
        return float(sum(1 for tok in tokens if tok in t))

    candidates = []
    pattern_hits = {}

    for i, row in enumerate(DATASET):
        err = row.get("error", "") or ""
        sol = row.get("solution", "") or ""
        pats = row.get("patterns") or []

        base = overlap_score(err + " " + sol)

        boost = 0
        for p in pats:
            try:
                rx = _compile_pattern(p)
                if rx and rx.search(haystack):
                    boost += 1
            except Exception:
                pass
        if boost > 0:
            pattern_hits[i] = boost

        score = base + 3.0 * boost
        candidates.append((score, i, base, boost))

    candidates.sort(key=lambda t: (t[0], t[2], t[3]), reverse=True)

    top = [c for c in candidates if c[0] > 0][:k] or candidates[:k]

    out = []
    for rank, (score, i, base, boost) in enumerate(top, 1):
        row = DATASET[i]
        out.append({
            "rank": rank,
            "score": float(round(score, 4)),
            "base_similarity": float(round(base, 4)),
            "pattern_boost": float(round(boost, 4)),
            "id": row.get("id"),
            "error": row.get("error"),
            "solution": row.get("solution"),
            "patterns": row.get("patterns", []),
            "matched_by": "(padrões do dataset)" if pattern_hits.get(i, 0) else None,
            "source": "dataset",
            "category": row.get("category"),
            "severity": row.get("severity"),
            "explanation": row.get("explanation"),
            "diagnostics": row.get("diagnostics"),
            "fix_steps": row.get("fix_steps"),
            "tags": row.get("tags"),
            "k8s_commands": row.get("k8s_commands"),
            "references": row.get("references"),
        })
    return out

def _generate_answer(question: str, hits: list[dict], context_text: str = "", enable_llm: bool = True):
    lang = _detect_lang(question)
    if not enable_llm:
        bullets = "\n".join([f"- {h.get('solution','')}" for h in hits[:CHAT_MAX_SOURCES]])
        return (f"Possíveis soluções com base no dataset:\n{bullets}" if lang == "pt" else f"Possible solutions:\n{bullets}"), [h["id"] for h in hits[:CHAT_MAX_SOURCES]]

    llm = _get_llm()
    if llm is None:
        bullets = "\n".join([f"- {h.get('solution','')}" for h in hits[:CHAT_MAX_SOURCES]])
        return (f"Possíveis soluções com base no dataset:\n{bullets}" if lang == "pt" else f"Possible solutions:\n{bullets}"), [h["id"] for h in hits[:CHAT_MAX_SOURCES]]

    prompt = _compose_prompt(question, hits[:CHAT_MAX_SOURCES], lang)
    if _llm_kind == "causal":
        out = llm(prompt, max_new_tokens=260, temperature=0.2, do_sample=False)
        text = out[0]["generated_text"]
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
    else:  # seq2seq
        out = llm(prompt, max_new_tokens=220, temperature=0.2, num_return_sequences=1)
        text = out[0]["generated_text"].strip()

    return text, [h["id"] for h in hits[:CHAT_MAX_SOURCES]]

# ---------- placeholders ----------
def _apply_placeholders(text: str, ns: str | None, pod: str | None) -> str:
    if not text:
        return text
    if ns:
        text = re.sub(r'(?i)(<ns>|&lt;ns&gt;)', ns, text)
    if pod:
        text = re.sub(r'(?i)(<pod>|&lt;pod&gt;)', pod, text)
    return text

def _apply_placeholders_list(lst, ns: str | None, pod: str | None):
    if not lst:
        return []
    return [_apply_placeholders(item, ns, pod) for item in lst]

# ---------- ENDPOINTS DO CHATBOT ----------
@app.route("/api/chatbot/build", methods=["POST"])
def api_chatbot_build():
    # Recarrega datasets (novo) e confirma legado (compat)
    global CHAT_DATASET
    CHAT_DATASET = _load_chat_dataset()
    ok = _build_index_if_needed(force=True)
    if not ok:
        return jsonify({"ok": False, "error": _index_error}), 500
    return jsonify({"ok": True, "chat_items": len(CHAT_DATASET)})

@app.route("/api/chatbot/health")
def api_chatbot_health():
    ok = _build_index_if_needed(False)
    status = "ok" if ok else f"index_error: {_index_error}"
    return jsonify({"status": status, "chat_items": len(CHAT_DATASET)})

@app.route("/api/chatbot/ask", methods=["POST"])
def api_chatbot_ask():
    """
    Body esperado:
      {
        question: str  (ou 'q')
        ns?: str, pod?: str
        // logs/events ignorados agora (sem 'usar contexto do pod')
      }
    """
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or payload.get("q") or "").strip()
    if not question:
        return jsonify({"error": "Pergunta vazia."}), 400

    ns = payload.get("ns") or None
    pod = payload.get("pod") or None

    try:
        tags = []
        lang = _detect_lang(question)

        # 1) Busca no dataset rico do chatbot
        chat_hits = _search_chat(question, k=CHAT_MAX_SOURCES)
        if chat_hits:
            # Confiança alta se bateu forte por alias/slug/keywords
            high_conf = chat_hits[0]["score"] >= 8.0
            top_it = chat_hits[0]["item"]

            final_answer = _render_chat_answer(top_it, ns, pod, lang)

            # matches: apenas "veja também" com snippet (overview). Evita duplicar resposta longa.
            matches = []
            if high_conf:
                rel = chat_hits[1:]
            else:
                rel = chat_hits  # baixa confiança: lista todos como opções

            for h in rel:
                it = h["item"]
                matches.append({
                    "rank": h["rank"],
                    "score": h["score"],
                    "name": it.get("name"),
                    "slug": it.get("slug"),
                    "overview": _apply_placeholders(it.get("overview",""), ns, pod),
                    "matched_by": h.get("matched_by") or "aliases/keywords/patterns",
                    "source": "chatbot-dataset"
                })

            # Sugerir texto “veja também” no final da resposta (somente nomes)
            also_names = [m["name"] for m in matches[:5] if m.get("name")]
            if also_names:
                final_answer += "\n\n" + ("**Veja também:** " if lang == "pt" else "**See also:** ")
                final_answer += ", ".join(also_names)

            tags = list((top_it.get("signals",{}) or {}).get("log_keywords") or [])[:8]

            return jsonify({
                "final_answer": final_answer,
                "sources": [ (top_it.get("slug") or top_it.get("name") or "chatbot") ],
                "matches": matches,
                "tags": tags
            })

        # 2) Fallback: dataset legado (“Soluções”)
        hits = _search(question, k=CHAT_MAX_SOURCES)
        if ns or pod:
            for h in hits:
                h["solution"]     = _apply_placeholders(h.get("solution",""), ns, pod)
                h["error"]        = _apply_placeholders(h.get("error",""), ns, pod)
                h["explanation"]  = _apply_placeholders(h.get("explanation",""), ns, pod)
                h["diagnostics"]  = _apply_placeholders_list(h.get("diagnostics"), ns, pod)
                h["fix_steps"]    = _apply_placeholders_list(h.get("fix_steps"), ns, pod)
                h["k8s_commands"] = _apply_placeholders_list(h.get("k8s_commands"), ns, pod)
                h["references"]   = _apply_placeholders_list(h.get("references"), ns, pod)

        final_answer, src_ids = _generate_answer(question, hits, enable_llm=bool(USE_LLM))
        return jsonify({
            "final_answer": final_answer,
            "sources": src_ids,
            "matches": hits,
            "tags": []
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =====================================================================
# ================== FIM: SEÇÃO CHATBOT (DATASET RICO) ================
# =====================================================================

if __name__ == "__main__":
    try:
        _build_index_if_needed(False)
    except Exception as e:
        print("Chatbot index warmup failed:", e)

    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
