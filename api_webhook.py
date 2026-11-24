"""Webhook receiver: an HTTP call (GitHub push, curl, ...) launches a Prefect deployment.

    GitHub / curl  ->  ngrok  ->  this API (port 8000)  ->  Prefect server  ->  worker
                                                                (deploiement_prefect.py)
"""

import hashlib
import hmac
import json
import os

from fastapi import FastAPI, Header, HTTPException, Request
from prefect.deployments import run_deployment
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="ML pipeline webhook")

# Shared secret: GitHub signs its requests with it, curl sends it in X-Token
SECRET = os.environ.get("WEBHOOK_SECRET", "")
# Flow launched on every push to main (code = checks only, all = full pipeline)
PUSH_FLOW = os.environ.get("PUSH_FLOW", "code")

# flow name -> "<flow>/<deployment>" (names from deploiement_prefect.py)
DEPLOYMENTS = {
    "all": "all/ml-pipeline-all",
    "train": "train/ml-pipeline-train",
    "predict": "predict/ml-pipeline-predict",
    "code": "code/ml-pipeline-code",
    "evaluate": "evaluate/ml-pipeline-evaluate",
}


def require_secret():
    if not SECRET:
        raise HTTPException(status_code=500, detail="WEBHOOK_SECRET is not set")


def launch(flow_name):
    # timeout=0 : return immediately, do not wait for the flow to finish
    flow_run = run_deployment(name=DEPLOYMENTS[flow_name], timeout=0)
    return {"status": "started", "flow": flow_name, "flow_run_id": str(flow_run.id)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/trigger/{flow_name}")
def trigger(flow_name: str, x_token: str = Header(default="")):
    """Manual trigger: curl -X POST .../trigger/train -H 'X-Token: <secret>'"""
    require_secret()
    if not hmac.compare_digest(x_token.encode(), SECRET.encode()):
        raise HTTPException(status_code=401, detail="Invalid token")
    if flow_name not in DEPLOYMENTS:
        raise HTTPException(status_code=404, detail=f"Unknown flow: {flow_name}")
    return launch(flow_name)


@app.post("/webhook/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(default=""),
    x_github_event: str = Header(default=""),
):
    """GitHub webhook: a push to main launches PUSH_FLOW."""
    require_secret()
    body = await request.body()
    expected = "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected.encode(), x_hub_signature_256.encode()):
        raise HTTPException(status_code=401, detail="Invalid signature")

    if x_github_event == "ping":
        return {"status": "pong"}
    if x_github_event != "push":
        return {"status": "ignored", "reason": f"event {x_github_event}"}
    if json.loads(body).get("ref") != "refs/heads/main":
        return {"status": "ignored", "reason": "not the main branch"}

    # run_deployment is synchronous, so keep it off the async event loop
    return await run_in_threadpool(launch, PUSH_FLOW)
