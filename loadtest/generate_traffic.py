import argparse
import glob
import os
import random
import statistics
import time

import requests


def load_image_pool(images_dir):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(images_dir, e)))
    if not files:
        raise SystemExit("Khong tim thay anh nao trong " + images_dir)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--images", default="./samples")
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--rate", type=float, default=2.0)
    args = ap.parse_args()

    pool = load_image_pool(args.images)
    print("Da nap " + str(len(pool)) + " anh mau tu " + args.images)

    delay = 1.0 / args.rate if args.rate > 0 else 0.0
    latencies = []
    errors = 0
    verdicts = {"OK": 0, "NG": 0, "unknown": 0}

    for i in range(args.requests):
        img_path = random.choice(pool)
        t0 = time.perf_counter()
        try:
            with open(img_path, "rb") as f:
                files = {"file": (os.path.basename(img_path), f, "image/jpeg")}
                r = requests.post(args.url + "/predict", files=files, timeout=15)
            latencies.append(time.perf_counter() - t0)
            if r.status_code == 200:
                v = r.json().get("verdict", "unknown")
                verdicts[v] = verdicts.get(v, 0) + 1
            else:
                errors += 1
        except requests.RequestException as exc:
            errors += 1
            print("  [" + str(i) + "] loi: " + str(exc))

        if (i + 1) % 10 == 0:
            print("  ... " + str(i + 1) + "/" + str(args.requests) + " requests")

        time.sleep(delay)

    print("")
    print("--- summary ---")
    print("requests : " + str(args.requests))
    print("errors   : " + str(errors))
    if latencies:
        mean_ms = statistics.mean(latencies) * 1000
        sorted_lat = sorted(latencies)
        idx = int(len(sorted_lat) * 0.95) - 1
        if idx < 0:
            idx = 0
        p95_ms = sorted_lat[idx] * 1000
        print("latency  : mean %.1f ms, p95 %.1f ms" % (mean_ms, p95_ms))
    print("verdicts : " + str(verdicts))


if __name__ == "__main__":
    main()
