#!/usr/bin/env python3
"""Write the sample workloads: one gzipped API gateway access log per day.

usage: python workloads/make_workloads.py

The logs are synthetic stand-ins for a normal week of gateway traffic. Every day
has its own fixed seed, so the files are byte-identical on every run and the
reference summaries in expected_output/ stay valid. If you change anything here,
rewrite expected_output/ by running the unmodified process.py on each file.
"""

from __future__ import annotations

import gzip
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
SEED = 20260907

# Requests per day before duplicate lines are added. Weekends are quieter.
DAYS = {
    "2026-09-07": 14800,  # Mon
    "2026-09-08": 16200,  # Tue
    "2026-09-09": 15500,  # Wed
    "2026-09-10": 17000,  # Thu
    "2026-09-11": 15000,  # Fri
    "2026-09-12": 10200,  # Sat
    "2026-09-13": 9500,  # Sun
    "2026-09-14": 15800,  # Mon
}

DUPLICATE_RATE = 0.035
MALFORMED_RATE = 0.0006
NOISE_RATE = 0.004

# Share of the day's traffic in each UTC hour.
HOURLY = [2, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 7, 7, 8, 8, 7, 7, 6, 6, 5, 5, 4, 3, 2]

# (method, template, weight, median latency ms, median bytes)
ROUTES = [
    ("GET", "/api/v1/health", 6, 2, 60),
    ("GET", "/api/v1/users/{id}", 9, 18, 1400),
    ("PATCH", "/api/v1/users/{id}", 1, 35, 1400),
    ("GET", "/api/v1/users/{id}/orders", 6, 60, 5200),
    ("GET", "/api/v1/users/{id}/orders/{id}", 4, 30, 2600),
    ("GET", "/api/v1/users/{id}/addresses", 2, 20, 900),
    ("POST", "/api/v1/users/{id}/addresses", 0.4, 45, 400),
    ("GET", "/api/v1/users/{id}/wishlist", 2, 25, 3100),
    ("PUT", "/api/v1/users/{id}/wishlist/{sku}", 0.8, 30, 200),
    ("DELETE", "/api/v1/users/{id}/wishlist/{sku}", 0.3, 28, 120),
    ("GET", "/api/v1/products", 10, 80, 18000),
    ("GET", "/api/v1/products/{sku}", 16, 22, 4200),
    ("GET", "/api/v1/products/{sku}/reviews", 5, 55, 9800),
    ("POST", "/api/v1/products/{sku}/reviews", 0.3, 70, 300),
    ("GET", "/api/v1/products/{sku}/stock", 6, 12, 150),
    ("GET", "/api/v1/categories", 3, 15, 2400),
    ("GET", "/api/v1/categories/{slug}", 3, 20, 1800),
    ("GET", "/api/v1/categories/{slug}/products", 5, 90, 16000),
    ("GET", "/api/v1/search", 8, 140, 12000),
    ("GET", "/api/v1/cart", 5, 25, 2100),
    ("POST", "/api/v1/cart/items", 2.5, 40, 2100),
    ("PATCH", "/api/v1/cart/items/{sku}", 0.8, 38, 2100),
    ("DELETE", "/api/v1/cart/items/{sku}", 0.6, 35, 2000),
    ("POST", "/api/v1/checkout", 1, 420, 1200),
    ("GET", "/api/v1/orders/{id}", 2, 25, 2600),
    ("POST", "/api/v1/orders/{id}/cancel", 0.1, 160, 300),
    ("GET", "/api/v1/orders/{id}/tracking", 1.5, 210, 1700),
    ("POST", "/api/v1/payments/{token}/confirm", 0.9, 650, 500),
    ("POST", "/api/v1/auth/login", 2, 110, 700),
    ("POST", "/api/v1/auth/refresh", 4, 15, 650),
    ("POST", "/api/v1/auth/logout", 0.7, 10, 50),
    ("GET", "/api/v1/promotions/{slug}", 1.5, 30, 3500),
]

# Requests that match no route: scanners, stale clients, typos.
NOISE = [
    ("GET", "/wp-login.php"),
    ("GET", "/.env"),
    ("GET", "/robots.txt"),
    ("GET", "/api/v0/products"),
    ("GET", "/api/v1/users/me"),
    ("POST", "/api/v1/products"),
    ("GET", "/api/v1/checkout"),
    ("GET", "/favicon.ico"),
]

WORDS = ["summer", "sale", "garden", "kitchen", "outdoor", "kids", "audio", "running", "home", "office", "gift", "new"]
QUERIES = ["shoes", "usb-c+cable", "desk+lamp", "coffee", "backpack", "headphones", "tent", "rain+jacket"]


def placeholder(rng: random.Random, name: str) -> str:
    if name == "id":
        return str(rng.randint(1, 99999))
    if name == "sku":
        return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3)) + f"-{rng.randint(0, 9999):04d}"
    if name == "slug":
        return "-".join(rng.sample(WORDS, rng.randint(1, 3)))
    return f"{rng.getrandbits(128):032x}"


def fill(rng: random.Random, template: str) -> str:
    parts = []
    for segment in template.split("/"):
        parts.append(placeholder(rng, segment[1:-1]) if segment.startswith("{") else segment)
    path = "/".join(parts)
    if template in ("/api/v1/products", "/api/v1/categories/{slug}/products", "/api/v1/users/{id}/orders"):
        if rng.random() < 0.3:
            path += f"?page={rng.randint(2, 9)}"
    elif template == "/api/v1/search":
        path += f"?q={rng.choice(QUERIES)}"
    return path


def status_for(rng: random.Random, method: str) -> int:
    x = rng.random()
    if x < 0.006:
        return rng.choice([500, 502, 503, 504])
    if x < 0.04:
        return rng.choice([400, 401, 403, 404, 404, 409, 429])
    if method == "GET" and x < 0.09:
        return 304
    if method == "POST" and x < 0.5:
        return 201
    if method == "DELETE":
        return 204
    return 200


def request_line(rng: random.Random, ts: str, clients: list[str], weights: list[float], rid: str) -> str:
    client = rng.choices(clients, weights)[0]
    if rng.random() < NOISE_RATE:
        method, path = rng.choice(NOISE)
        return f"{ts} {client} {method} {path} 404 {rng.randint(1, 4)} 162 rid={rid}"
    method, template, _, latency_ms, size = rng.choices(ROUTES, ROUTE_WEIGHTS)[0]
    status = status_for(rng, method)
    latency = max(1, round(rng.lognormvariate(0, 0.6) * latency_ms))
    if rng.random() < 0.01:
        latency *= rng.randint(5, 30)
    body = 0 if status in (204, 304) else max(20, round(rng.lognormvariate(0, 0.35) * size))
    return f"{ts} {client} {method} {fill(rng, template)} {status} {latency} {body} rid={rid}"


ROUTE_WEIGHTS = [r[2] for r in ROUTES]


def make_day(day: str, requests: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    clients = [f"10.{rng.randint(0, 31)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}" for _ in range(400)]
    weights = [1 / (i + 1) ** 0.9 for i in range(len(clients))]
    millis = sorted(
        rng.choices(range(24), HOURLY)[0] * 3_600_000 + rng.randrange(3_600_000) for _ in range(requests)
    )
    lines = []
    for ms in millis:
        h, rest = divmod(ms, 3_600_000)
        m, rest = divmod(rest, 60_000)
        s, frac = divmod(rest, 1000)
        ts = f"{day}T{h:02d}:{m:02d}:{s:02d}.{frac:03d}Z"
        rid = f"{rng.getrandbits(48):012x}"
        lines.append(request_line(rng, ts, clients, weights, rid))

    # At-least-once shipping: some lines arrive again a little later.
    ordered = [(float(i), line) for i, line in enumerate(lines)]
    for i, line in enumerate(lines):
        if rng.random() < DUPLICATE_RATE:
            ordered.append((i + rng.randint(1, 600) + 0.5, line))
    ordered.sort(key=lambda item: item[0])
    out = []
    for _, line in ordered:
        if rng.random() < MALFORMED_RATE:
            line = line[: rng.randint(10, line.index(" rid=") - 1)]
        out.append(line)
    return out


def write_gz(path: Path, lines: list[str]) -> None:
    data = ("\n".join(lines) + "\n").encode("utf-8")
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
        gz.write(data)


def main() -> None:
    for i, (day, requests) in enumerate(DAYS.items()):
        path = OUT_DIR / f"access-{day}.log.gz"
        lines = make_day(day, requests, SEED + i)
        write_gz(path, lines)
        print(f"{path.name}: {len(lines)} lines, {path.stat().st_size // 1024} KiB")


if __name__ == "__main__":
    main()
