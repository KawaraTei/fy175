"""Optional standard-library client. The API itself is the complete reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request(url, endpoint="/", body=None):
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    req = Request(url.rstrip("/") + endpoint, data=encoded,
                  headers={"Content-Type": "application/json"} if encoded else {})
    with urlopen(req, timeout=60) as response:
        content = response.read()
        return (json.loads(content) if response.headers.get_content_type() == "application/json" else content,
                dict(response.headers))


def main():
    parser = argparse.ArgumentParser(description="FY175AutoMosaic agent API client; 'describe' discovers all operations")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("describe")
    sub.add_parser("state")
    call = sub.add_parser("call")
    call.add_argument("operation")
    call.add_argument("arguments", nargs="?", default="{}", help="JSON object; @path reads UTF-8 JSON file")
    call.add_argument("--wait", action="store_true", help="Poll job until terminal status")
    get = sub.add_parser("get")
    get.add_argument("endpoint", help="e.g. /preview?workspace=image&view=result")
    get.add_argument("--output", type=Path, help="Save binary image response to this file")
    options = parser.parse_args()
    try:
        endpoint, body = "/", None
        if options.command == "state":
            endpoint = "/state"
        elif options.command == "get":
            endpoint = options.endpoint
        elif options.command == "call":
            raw = Path(options.arguments[1:]).read_text(encoding="utf-8-sig") if options.arguments.startswith("@") else options.arguments
            body = {"operation": options.operation, "arguments": json.loads(raw)}
            endpoint = "/commands"
        result, headers = request(options.url, endpoint, body)
        if options.command == "call" and options.wait and result.get("job"):
            job = result["job"]
            while job["status"] == "running":
                time.sleep(0.2)
                job, _ = request(options.url, "/jobs/" + job["id"])
            result = job
        if isinstance(result, bytes):
            if options.output is None:
                parser.error("Binary response requires --output PATH")
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_bytes(result)
            print(json.dumps({"path": str(options.output.resolve()), "metadata": headers.get("X-Agent-Metadata")}, ensure_ascii=False))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result.get("status") in {"failed", "cancelled"}:
                raise SystemExit(1)
    except HTTPError as error:
        print(error.read().decode("utf-8"))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
