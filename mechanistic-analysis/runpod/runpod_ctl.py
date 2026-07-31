#!/usr/bin/env python3
"""Minimal RunPod control plane for the persona-geometry runs.

Subcommands:
  prices     list price and stock for large-VRAM GPUs
  create     deploy an on-demand pod, wait for SSH, record the pod id
  status     show the recorded pod (or --id) and its SSH endpoint
  ssh        print the ssh command for the recorded pod
  terminate  terminate the recorded pod (or --id) and clear the record

The API key is read from the repo-root .env and never printed. urllib is
blocked by the API edge, so requests go through curl.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
POD_RECORD = Path(__file__).resolve().parent / ".pod_id"
GRAPHQL = "https://api.runpod.io/graphql"

# CUDA 12.9 / torch 2.9.1: required for Blackwell (sm_120) on RTX PRO 6000.
DEFAULT_IMAGE = "runpod/pytorch:1.1.0-rc.154-cu1290-torch291-ubuntu2404"
DEFAULT_GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
DEFAULT_DISK_GB = 200  # raise with --disk-gb for large bases: 235B bf16 needs ~600


def load_env() -> dict[str, str]:
    env_path = ROOT / ".env"
    if not env_path.exists():
        raise SystemExit(f"No .env at {env_path}")
    out: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def api_key() -> str:
    env = load_env()
    key = env.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("RUNPOD_API_KEY missing from .env")
    return key


def gql(query: str) -> dict:
    result = subprocess.run(
        [
            "curl", "-s", "--max-time", "60", "-X", "POST", GRAPHQL,
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {api_key()}",
            "-d", json.dumps({"query": query}),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"curl failed: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"Non-JSON response: {result.stdout[:400]}")
    if "errors" in payload:
        raise SystemExit("API error: " + json.dumps(payload["errors"], indent=2))
    return payload["data"]


def public_key() -> str:
    for name in ("id_ed25519.pub", "id_rsa.pub"):
        path = Path.home() / ".ssh" / name
        if path.exists():
            return path.read_text().strip()
    raise SystemExit("No SSH public key found in ~/.ssh")


def cmd_prices(args: argparse.Namespace) -> None:
    gpus = gql("query { gpuTypes { id memoryInGb } }")["gpuTypes"]
    targets = [g for g in gpus if g["memoryInGb"] >= args.min_vram]
    print(f'{"gpuTypeId":46s} {"VRAM":>5s} {"secure$":>8s} {"stock":>8s} {"comm$":>7s} {"stock":>8s}')
    for gpu in sorted(targets, key=lambda g: g["memoryInGb"]):
        row = [gpu["id"][:46], gpu["memoryInGb"]]
        for cloud in ("true", "false"):
            query = (
                'query { gpuTypes(input:{id:"%s"}) { lowestPrice(input:{gpuCount:%d,secureCloud:%s}) '
                "{ uninterruptablePrice stockStatus } } }" % (gpu["id"], args.gpu_count, cloud)
            )
            try:
                low = gql(query)["gpuTypes"][0]["lowestPrice"]
                row += [low.get("uninterruptablePrice") or "-", low.get("stockStatus") or "none"]
            except SystemExit:
                row += ["err", "err"]
        print(f"{row[0]:46s} {row[1]:5d} {str(row[2]):>8s} {str(row[3]):>8s} {str(row[4]):>7s} {str(row[5]):>8s}")


def pod_fields() -> str:
    return (
        "id name desiredStatus costPerHr machineId "
        "runtime { uptimeInSeconds ports { ip publicPort privatePort isIpPublic } }"
    )


def find_pod(pod_id: str) -> dict | None:
    data = gql("query { myself { pods { %s } } }" % pod_fields())
    for pod in data["myself"]["pods"]:
        if pod["id"] == pod_id:
            return pod
    return None


def ssh_endpoint(pod: dict) -> tuple[str, int] | None:
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return port["ip"], int(port["publicPort"])
    return None


def cmd_create(args: argparse.Namespace) -> None:
    if POD_RECORD.exists() and not args.force:
        raise SystemExit(
            f"A pod is already recorded ({POD_RECORD.read_text().strip()}).\n"
            "Terminate it first, or pass --force to deploy a second one."
        )
    env_entries = [{"key": "PUBLIC_KEY", "value": public_key()}]
    env_literal = ",".join(
        '{key:"%s",value:"%s"}' % (e["key"], e["value"].replace('"', '\\"')) for e in env_entries
    )
    mutation = (
        "mutation { podFindAndDeployOnDemand(input:{"
        "cloudType:SECURE,"
        "gpuCount:%d,"
        'gpuTypeId:"%s",'
        'name:"%s",'
        'imageName:"%s",'
        "containerDiskInGb:%d,"
        "volumeInGb:0,"
        'ports:"22/tcp",'
        "startSsh:true,"
        "supportPublicIp:true,"
        "env:[%s]"
        "}) { id name costPerHr machineId } }"
        % (args.gpu_count, args.gpu_type, args.name, args.image, args.disk_gb, env_literal)
    )
    pod = gql(mutation)["podFindAndDeployOnDemand"]
    if not pod:
        raise SystemExit("Deploy returned null. Usually means no capacity for that GPU right now.")
    POD_RECORD.write_text(pod["id"] + "\n")

    print(f"pod id     : {pod['id']}")
    print(f"name       : {pod['name']}")
    print(f"cost/hr    : ${pod['costPerHr']}")
    print(f"recorded at: {POD_RECORD}")
    print()
    print("!! BILLING IS LIVE. Stop it with:")
    print(f"   python {Path(__file__).relative_to(ROOT)} terminate")
    print()

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        current = find_pod(pod["id"])
        if current:
            endpoint = ssh_endpoint(current)
            if endpoint:
                host, port = endpoint
                print(f"SSH ready  : ssh -p {port} root@{host}")
                return
        print("waiting for SSH...")
        time.sleep(10)
    print(f"Pod deployed but SSH not up after {args.timeout}s. Check `status`.", file=sys.stderr)


def recorded_id(args: argparse.Namespace) -> str:
    if getattr(args, "id", None):
        return args.id
    if not POD_RECORD.exists():
        raise SystemExit("No pod recorded. Pass --id explicitly.")
    return POD_RECORD.read_text().strip()


def cmd_status(args: argparse.Namespace) -> None:
    data = gql("query { myself { pods { %s } } }" % pod_fields())
    pods = data["myself"]["pods"]
    if not pods:
        print("No pods on this account. Nothing is billing.")
        return
    for pod in pods:
        runtime = pod.get("runtime") or {}
        uptime = runtime.get("uptimeInSeconds") or 0
        spent = (pod.get("costPerHr") or 0) * uptime / 3600
        endpoint = ssh_endpoint(pod)
        where = f"{endpoint[0]}:{endpoint[1]}" if endpoint else "no public ssh"
        print(
            f"{pod['id']}  {pod['name']}  {pod['desiredStatus']}  "
            f"${pod['costPerHr']}/hr  up {uptime // 60}m  ~${spent:.2f} so far  {where}"
        )


def cmd_ssh(args: argparse.Namespace) -> None:
    pod = find_pod(recorded_id(args))
    if not pod:
        raise SystemExit("Pod not found.")
    endpoint = ssh_endpoint(pod)
    if not endpoint:
        raise SystemExit("Pod has no public SSH port yet.")
    host, port = endpoint
    print(f"ssh -p {port} root@{host}")


def cmd_terminate(args: argparse.Namespace) -> None:
    pod_id = recorded_id(args)
    gql('mutation { podTerminate(input:{podId:"%s"}) }' % pod_id)
    print(f"terminate sent for {pod_id}")
    time.sleep(5)
    remaining = gql("query { myself { pods { id name desiredStatus } } }")["myself"]["pods"]
    if POD_RECORD.exists():
        POD_RECORD.unlink()
    if remaining:
        print("STILL PRESENT, verify in console:")
        for pod in remaining:
            print(f"  {pod['id']} {pod['name']} {pod['desiredStatus']}")
    else:
        print("No pods remain on the account. Billing stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prices = sub.add_parser("prices", help="list price and stock")
    p_prices.add_argument("--min-vram", type=int, default=80)
    p_prices.add_argument("--gpu-count", type=int, default=1)
    p_prices.set_defaults(func=cmd_prices)

    p_create = sub.add_parser("create", help="deploy an on-demand pod")
    p_create.add_argument("--gpu-type", default=DEFAULT_GPU)
    p_create.add_argument("--gpu-count", type=int, default=1)
    p_create.add_argument("--image", default=DEFAULT_IMAGE)
    p_create.add_argument("--disk-gb", type=int, default=DEFAULT_DISK_GB)
    p_create.add_argument("--name", default="persona-geometry")
    p_create.add_argument("--timeout", type=int, default=600)
    p_create.add_argument("--force", action="store_true")
    p_create.set_defaults(func=cmd_create)

    p_status = sub.add_parser("status", help="list all pods and spend so far")
    p_status.set_defaults(func=cmd_status)

    p_ssh = sub.add_parser("ssh", help="print the ssh command")
    p_ssh.add_argument("--id")
    p_ssh.set_defaults(func=cmd_ssh)

    p_term = sub.add_parser("terminate", help="terminate and verify")
    p_term.add_argument("--id")
    p_term.set_defaults(func=cmd_terminate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
