from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import time
from urllib.parse import urlparse, quote

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml

# Kubernetes client to read Argo CD repo credentials
from kubernetes import client, config
from kubernetes.client import ApiException


class PluginInputParameters(BaseModel):
    repoURL: str
    revision: str = "master"
    basePath: str = "dev/cm"
    includeGlobs: List[str] | None = None
    excludeFiles: List[str] | None = None
    aggregate: bool = False


class PluginRequest(BaseModel):
    # ApplicationSet controller forwards the `input.parameters` as provided in the generator spec
    input: Dict[str, Any] | None = None


class PluginResponse(BaseModel):
    parameters: List[Dict[str, Any]]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("git-files-plugin")

app = FastAPI()


def load_k8s_config() -> None:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        # Fall back to local kubeconfig for dev
        try:
            config.load_kube_config()
        except Exception:  # noqa: BLE001
            pass


def find_matching_repo_secret(repo_url: str, namespace: str = "argocd") -> Optional[Dict[str, str]]:
    load_k8s_config()
    v1 = client.CoreV1Api()

    parsed = urlparse(repo_url)
    host = parsed.netloc
    try:
        secrets = v1.list_namespaced_secret(
            namespace,
            label_selector="argocd.argoproj.io/secret-type in (repository,repo-creds,repository-credentials)",
        )
    except ApiException as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"k8s error listing secrets: {exc}")

    def decode(data: Dict[str, str], key: str) -> Optional[str]:
        if key not in data:
            return None
        try:
            return base64.b64decode(data[key]).decode()
        except Exception:  # noqa: BLE001
            return None

    best: Optional[Dict[str, str]] = None
    best_score = -1

    for sec in secrets.items:
        data = sec.data or {}
        url = decode(data, "url")
        username = decode(data, "username")
        password = decode(data, "password")
        ssh_key = decode(data, "sshPrivateKey")

        score = 0
        if url:
            p = urlparse(url)
            if p.netloc == host:
                score += 10
            # Prefer longer prefix matches
            if repo_url.startswith(url):
                score += len(url)
        # Prefer exact auth types available
        if password:
            score += 1
        if ssh_key:
            score += 1

        if score > best_score:
            best = {
                "url": url or "",
                "username": username or "",
                "password": password or "",
                "sshPrivateKey": ssh_key or "",
            }
            best_score = score

    if best:
        try:
            logger.info(
                "matched repo secret: url=%s has_username=%s has_password=%s has_ssh_key=%s",
                best.get("url"), bool(best.get("username")), bool(best.get("password")), bool(best.get("sshPrivateKey")),
            )
        except Exception:
            pass
    else:
        logger.info("no matching repo secret found for host: %s", host)
    return best


def run_git_clone(repo_url: str, revision: str, target_dir: Path) -> None:
    # Try to inherit credentials from argocd namespace
    creds = find_matching_repo_secret(repo_url)

    git_env = os.environ.copy()
    ssh_key_path: Optional[Path] = None

    # If we have SSH key, use ssh transport
    if creds and creds.get("sshPrivateKey"):
        # Convert URL to ssh form if https provided but key is present
        parsed = urlparse(repo_url)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            # Example: git@gitlab.rip:group/repo.git
            repo_path = parsed.path.lstrip("/")
            repo_url = f"git@{parsed.netloc}:{repo_path}"
        # Write key to temp file
        ssh_dir = Path(tempfile.mkdtemp(prefix="ssh-"))
        ssh_key_path = ssh_dir / "id_rsa"
        ssh_key_path.write_text(creds["sshPrivateKey"], encoding="utf-8")
        os.chmod(ssh_key_path, 0o600)
        git_env["GIT_SSH_COMMAND"] = f"ssh -i {ssh_key_path} -o StrictHostKeyChecking=accept-new"
        logger.info("git auth: using SSH key for host=%s", parsed.netloc)

    # If we have username/password (or token), inject into https URL
    elif creds and (creds.get("password") or creds.get("username")):
        parsed = urlparse(repo_url)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            user = quote((creds.get("username") or "oauth2"), safe="")
            pwd = quote((creds.get("password") or ""), safe="")
            netloc = f"{user}:{pwd}@{parsed.netloc}"
            repo_url = parsed._replace(netloc=netloc).geturl()
            logger.info("git auth: using HTTPS credentials for host=%s username=%s", parsed.netloc, user)
    else:
        try:
            parsed = urlparse(repo_url)
            logger.info("git auth: no credentials found for host=%s, cloning anonymously", parsed.netloc)
        except Exception:
            pass

    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                revision,
                repo_url,
                str(target_dir),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=git_env,
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"git clone failed: {exc.stderr.decode(errors='ignore')}")
    finally:
        if ssh_key_path:
            try:
                ssh_key_path.unlink(missing_ok=True)
                ssh_key_path.parent.rmdir()
            except Exception:  # noqa: BLE001
                pass


def list_service_dirs(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        return []
    return [p for p in base_dir.iterdir() if p.is_dir()]


def file_matches(name: str, include_globs: List[str] | None, exclude_files: List[str] | None) -> bool:
    from fnmatch import fnmatch

    if exclude_files and name in set(exclude_files):
        return False
    if not include_globs:
        # Default: yaml/yml files
        return name.endswith(".yaml") or name.endswith(".yml")
    return any(fnmatch(name, pat) for pat in include_globs)


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, override_value in (override or {}).items():
        base_value = base.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            base[key] = deep_merge_dicts(dict(base_value), dict(override_value))
        else:
            base[key] = override_value
    return base


def build_parameters(repo_root: Path, params: PluginInputParameters) -> List[Dict[str, Any]]:
    base_dir = (repo_root / params.basePath).resolve()
    results: List[Dict[str, Any]] = []

    logger.info(
        "build_parameters: base_dir=%s includeGlobs=%s excludeFiles=%s",
        str(base_dir), params.includeGlobs, params.excludeFiles,
    )

    # load common values (optional)
    common_path = base_dir / "helm-values-common.yaml"
    common_merged: Dict[str, Any] = {}
    if common_path.exists() and common_path.is_file():
        try:
            docs = list(yaml.safe_load_all(common_path.read_text(encoding="utf-8")))
            for doc in docs:
                if isinstance(doc, dict):
                    common_merged = deep_merge_dicts(common_merged, doc)
        except Exception:
            # ignore invalid common file
            pass

    # If aggregate requested: return single item with all value files across services and base dir
    if params.aggregate:
        aggregated_files: List[str] = []
        # collect from service dirs
        for svc_dir in list_service_dirs(base_dir):
            for f in sorted(svc_dir.iterdir()):
                if f.is_file() and file_matches(f.name, params.includeGlobs, params.excludeFiles):
                    aggregated_files.append(f"{svc_dir.name}/{f.name}")
        # collect flat files in base dir
        try:
            for f in sorted(base_dir.iterdir()):
                if not f.is_file():
                    continue
                if f.name == "helm-values-common.yaml":
                    continue
                if not file_matches(f.name, params.includeGlobs, params.excludeFiles):
                    continue
                aggregated_files.append(f.name)
        except Exception:
            pass

        # deduplicate, preserve order
        seen = set()
        deduped: List[str] = []
        for vf in aggregated_files:
            if vf in seen:
                continue
            seen.add(vf)
            deduped.append(vf)

        logger.info("aggregate: total files=%d", len(deduped))
        if deduped:
            results.append(
                {
                    "name": "aggregate",
                    "path": f"{params.basePath.rstrip('/')}",
                    "valueFiles": deduped,
                }
            )
        logger.info("build_parameters (aggregate): generated %d items", len(results))
        return results

    for svc_dir in list_service_dirs(base_dir):
        service_name = svc_dir.name
        # collect YAML files list and merge into one dict
        value_files_list: List[str] = []
        # collect and merge YAMLs into one dict
        merged: Dict[str, Any] = dict(common_merged)
        for f in sorted(svc_dir.iterdir()):
            if not (f.is_file() and file_matches(f.name, params.includeGlobs, params.excludeFiles)):
                continue
            value_files_list.append(f.name)
            try:
                content = f.read_text(encoding="utf-8")
            except Exception:
                continue
            try:
                # Support multi-doc YAML
                docs = list(yaml.safe_load_all(content))
            except Exception:
                # skip invalid yaml files
                continue
            for doc in docs:
                if isinstance(doc, dict):
                    merged = deep_merge_dicts(merged, doc)
                # non-dict documents are ignored

        if not merged:
            continue

        item = {
            "name": service_name,
            "path": f"{params.basePath.rstrip('/')}/{service_name}",
            "valueFiles": value_files_list,
        }
        results.append(item)

    # Also support flat files directly under basePath (e.g., dev/cm/foo.yaml)
    try:
        for f in sorted(base_dir.iterdir()):
            if not f.is_file():
                continue
            if f.name == "helm-values-common.yaml":
                continue
            if not file_matches(f.name, params.includeGlobs, params.excludeFiles):
                continue
            # service name from filename without extension
            service_name = f.stem
            merged: Dict[str, Any] = dict(common_merged)
            try:
                docs = list(yaml.safe_load_all(f.read_text(encoding="utf-8")))
            except Exception:
                continue
            for doc in docs:
                if isinstance(doc, dict):
                    merged = deep_merge_dicts(merged, doc)
            if not merged:
                continue
            item = {
                "name": service_name,
                "path": f"{params.basePath.rstrip('/')}/{service_name}",
                "valueFiles": [f.name],
            }
            results.append(item)
    except Exception:
        pass

    logger.info("build_parameters: generated %d items: %s", len(results), [r.get("name") for r in results])
    return results


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=PluginResponse)
def generate(req: PluginRequest) -> PluginResponse:
    start = time.time()
    raw_input = req.input or {}
    raw_params = raw_input.get("parameters") or {}

    try:
        params = PluginInputParameters(**raw_params)
    except Exception as exc:  # noqa: BLE001
        logger.error("/generate invalid parameters: %s", exc)
        raise HTTPException(status_code=400, detail=f"invalid parameters: {exc}")

    tmpdir = Path(tempfile.mkdtemp(prefix="appset-plugin-"))
    try:
        repo_dir = tmpdir / "repo"
        logger.info(
            "/generate start: repoURL=%s revision=%s basePath=%s",
            params.repoURL, params.revision, params.basePath,
        )
        run_git_clone(params.repoURL, params.revision, repo_dir)
        items = build_parameters(repo_dir, params)
        dur_ms = int((time.time() - start) * 1000)
        logger.info("/generate done: items=%d durationMs=%d", len(items), dur_ms)
        return PluginResponse(parameters=items)
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


# Compatibility endpoint expected by ApplicationSet plugin generator
def _process_generate(raw_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    t0 = time.time()
    # Fill defaults from env if fields are missing
    raw_params = dict(raw_params or {})
    raw_params.setdefault("repoURL", os.environ.get("DEFAULT_REPO_URL", ""))
    raw_params.setdefault("revision", os.environ.get("DEFAULT_REVISION", "master"))
    raw_params.setdefault("basePath", os.environ.get("DEFAULT_BASE_PATH", "dev/cm"))
    try:
        params = PluginInputParameters(**(raw_params or {}))
    except Exception as exc:  # noqa: BLE001
        logger.error("_process_generate invalid parameters: %s", exc)
        raise HTTPException(status_code=400, detail=f"invalid parameters: {exc}")

    tmpdir = Path(tempfile.mkdtemp(prefix="appset-plugin-"))
    try:
        repo_dir = tmpdir / "repo"
        logger.info(
            "getparams.execute start: repoURL=%s revision=%s basePath=%s",
            params.repoURL, params.revision, params.basePath,
        )
        run_git_clone(params.repoURL, params.revision, repo_dir)
        items = build_parameters(repo_dir, params)
        dur_ms = int((time.time() - t0) * 1000)
        logger.info("getparams.execute done: items=%d durationMs=%d", len(items), dur_ms)
        return items
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


@app.post("/api/v1/getparams.execute")
def getparams_execute(body: Dict[str, Any]) -> Dict[str, Any]:
    # Accept multiple request shapes:
    # 1) {"input": {"parameters": {...}}}
    # 2) {"parameters": {...}}
    # 3) {...} (raw parameters)
    if not isinstance(body, dict):
        body = {}
    candidate = body.get("input") if isinstance(body.get("input"), dict) else body
    raw_params = candidate.get("parameters") if isinstance(candidate.get("parameters"), dict) else candidate

    items = _process_generate(raw_params)
    # Возвращаем во всех популярных ключах для совместимости разных версий контроллера
    return {"output": {"parameters": items}}
