#!/usr/bin/env python3
"""Summarise one day of API gateway access logs for the on-call dashboard.

usage: python process.py INPUT OUTPUT

INPUT is one day's access log, plain text or gzipped. Each line looks like

    2026-09-08T09:14:03.412Z 10.12.4.77 GET /api/v1/users/48213/orders 200 37 5123 rid=3f9c2a7be41d

(timestamp, client, method, path, status, latency in ms, response bytes,
request id). Log shipping is at-least-once, so some lines arrive twice; they are
dropped by request id. OUTPUT gets a plain-text summary: totals, per-route
latency percentiles, status codes, hourly traffic and the busiest clients. The
dashboard parses OUTPUT, so its format must not change.
"""

from __future__ import annotations

import gzip
import re
import sys
from collections import Counter, defaultdict

# The gateway's public routes. Placeholders are filled in from PLACEHOLDERS.
ROUTE_TABLE = """
# method  template
GET     /api/v1/health
GET     /api/v1/users/{id}
PATCH   /api/v1/users/{id}
GET     /api/v1/users/{id}/orders
GET     /api/v1/users/{id}/orders/{id}
GET     /api/v1/users/{id}/addresses
POST    /api/v1/users/{id}/addresses
GET     /api/v1/users/{id}/wishlist
PUT     /api/v1/users/{id}/wishlist/{sku}
DELETE  /api/v1/users/{id}/wishlist/{sku}
GET     /api/v1/products
GET     /api/v1/products/{sku}
GET     /api/v1/products/{sku}/reviews
POST    /api/v1/products/{sku}/reviews
GET     /api/v1/products/{sku}/stock
GET     /api/v1/categories
GET     /api/v1/categories/{slug}
GET     /api/v1/categories/{slug}/products
GET     /api/v1/search
GET     /api/v1/cart
POST    /api/v1/cart/items
PATCH   /api/v1/cart/items/{sku}
DELETE  /api/v1/cart/items/{sku}
POST    /api/v1/checkout
GET     /api/v1/orders/{id}
POST    /api/v1/orders/{id}/cancel
GET     /api/v1/orders/{id}/tracking
POST    /api/v1/payments/{token}/confirm
POST    /api/v1/auth/login
POST    /api/v1/auth/refresh
POST    /api/v1/auth/logout
GET     /api/v1/promotions/{slug}
"""

PLACEHOLDERS = {
    "id": r"[0-9]+",
    "sku": r"[A-Z]{3}-[0-9]{4}",
    "slug": r"[a-z0-9]+(?:-[a-z0-9]+)*",
    "token": r"[0-9a-f]{32}",
}

UNMATCHED = "(unmatched)"
TOP_CLIENTS = 10


def load_routes(table: str) -> list[tuple[str, re.Pattern[str], str]]:
    """Turn the route table into (method, compiled pattern, label) triples."""
    routes = []
    for line in table.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        method, template = line.split()
        parts = []
        for segment in template.strip("/").split("/"):
            if segment.startswith("{") and segment.endswith("}"):
                parts.append(PLACEHOLDERS[segment[1:-1]])
            else:
                parts.append(re.escape(segment))
        pattern = re.compile("/" + "/".join(parts) + "/?")
        routes.append((method, pattern, f"{method} {template}"))
    return routes


def match_route(method: str, path: str) -> str:
    """The route label for a request, ignoring its query string."""
    path = path.split("?", 1)[0]
    for route_method, pattern, label in load_routes(ROUTE_TABLE):
        if route_method == method and pattern.fullmatch(path):
            return label
    return UNMATCHED


def parse_line(line: str) -> tuple[str, str, str, str, int, int, int, str] | None:
    """Split one log line into its fields, or None if it is malformed."""
    fields = line.split()
    if len(fields) != 8 or not fields[7].startswith("rid="):
        return None
    ts, client, method, path, status, latency, size, rid = fields
    try:
        return ts, client, method, path, int(status), int(latency), int(size), rid[4:]
    except ValueError:
        return None


def read_lines(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        yield from f


def percentile(sorted_values: list[int], pct: int) -> int:
    """Nearest-rank percentile of an already sorted list."""
    rank = (pct * len(sorted_values) + 99) // 100
    return sorted_values[max(rank, 1) - 1]


class Summary:
    def __init__(self) -> None:
        self.lines = 0
        self.malformed = 0
        self.duplicates = 0
        self.first_ts: str | None = None
        self.last_ts: str | None = None
        self.route_latency: dict[str, list[int]] = defaultdict(list)
        self.route_bytes: Counter[str] = Counter()
        self.route_class: dict[str, Counter[str]] = defaultdict(Counter)
        self.status: Counter[int] = Counter()
        self.hour_latency: dict[str, list[int]] = defaultdict(list)
        self.hour_4xx: Counter[str] = Counter()
        self.hour_5xx: Counter[str] = Counter()
        self.client_requests: Counter[str] = Counter()
        self.client_errors: Counter[str] = Counter()

    def add(self, ts: str, client: str, route: str, status: int, latency: int, size: int) -> None:
        if self.first_ts is None or ts < self.first_ts:
            self.first_ts = ts
        if self.last_ts is None or ts > self.last_ts:
            self.last_ts = ts
        status_class = f"{status // 100}xx"
        hour = ts[11:13]
        self.route_latency[route].append(latency)
        self.route_bytes[route] += size
        self.route_class[route][status_class] += 1
        self.status[status] += 1
        self.hour_latency[hour].append(latency)
        if status_class == "4xx":
            self.hour_4xx[hour] += 1
        elif status_class == "5xx":
            self.hour_5xx[hour] += 1
            self.client_errors[client] += 1
        self.client_requests[client] += 1

    @property
    def requests(self) -> int:
        return sum(self.status.values())


def summarise(path: str) -> Summary:
    summary = Summary()
    seen: list[str] = []
    for line in read_lines(path):
        summary.lines += 1
        record = parse_line(line)
        if record is None:
            summary.malformed += 1
            continue
        ts, client, method, req_path, status, latency, size, rid = record
        if rid in seen:
            summary.duplicates += 1
            continue
        seen.append(rid)
        summary.add(ts, client, match_route(method, req_path), status, latency, size)
    return summary


def write_summary(summary: Summary, path: str) -> None:
    with open(path, "w", encoding="utf-8") as out:
        print("access log summary", file=out)
        print("==================", file=out)
        print(f"lines read          {summary.lines:>9}", file=out)
        print(f"malformed lines     {summary.malformed:>9}", file=out)
        print(f"duplicate lines     {summary.duplicates:>9}", file=out)
        print(f"requests            {summary.requests:>9}", file=out)
        print(f"unmatched requests  {len(summary.route_latency.get(UNMATCHED, [])):>9}", file=out)
        print(f"first request       {summary.first_ts or '-'}", file=out)
        print(f"last request        {summary.last_ts or '-'}", file=out)

        print("", file=out)
        print("routes", file=out)
        print("------", file=out)
        print(
            f"{'route':<46} {'requests':>8} {'2xx':>7} {'3xx':>6} {'4xx':>6} {'5xx':>5}"
            f" {'p50_ms':>7} {'p95_ms':>7} {'p99_ms':>7} {'avg_bytes':>10}",
            file=out,
        )
        routes = sorted(summary.route_latency, key=lambda r: (-len(summary.route_latency[r]), r))
        for route in routes:
            latencies = sorted(summary.route_latency[route])
            classes = summary.route_class[route]
            avg_bytes = summary.route_bytes[route] / len(latencies)
            print(
                f"{route:<46} {len(latencies):>8} {classes['2xx']:>7} {classes['3xx']:>6}"
                f" {classes['4xx']:>6} {classes['5xx']:>5} {percentile(latencies, 50):>7}"
                f" {percentile(latencies, 95):>7} {percentile(latencies, 99):>7} {avg_bytes:>10.1f}",
                file=out,
            )

        print("", file=out)
        print("status codes", file=out)
        print("------------", file=out)
        for status in sorted(summary.status):
            print(f"{status}  {summary.status[status]:>8}", file=out)

        print("", file=out)
        print("hourly", file=out)
        print("------", file=out)
        print(f"{'hour':<4} {'requests':>8} {'4xx':>6} {'5xx':>5} {'p50_ms':>7} {'p95_ms':>7}", file=out)
        for hour in sorted(summary.hour_latency):
            latencies = sorted(summary.hour_latency[hour])
            print(
                f"{hour:<4} {len(latencies):>8} {summary.hour_4xx[hour]:>6} {summary.hour_5xx[hour]:>5}"
                f" {percentile(latencies, 50):>7} {percentile(latencies, 95):>7}",
                file=out,
            )

        print("", file=out)
        print("top clients", file=out)
        print("-----------", file=out)
        print(f"{'client':<16} {'requests':>8} {'5xx':>5}", file=out)
        top = sorted(summary.client_requests.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_CLIENTS]
        for client, count in top:
            print(f"{client:<16} {count:>8} {summary.client_errors[client]:>5}", file=out)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    write_summary(summarise(argv[1]), argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
