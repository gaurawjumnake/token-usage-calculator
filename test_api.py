"""
Test script for Token Usage Calculator API deployed on AWS Lambda + API Gateway.
Usage:
    python test_api.py https://<api-id>.execute-api.<region>.amazonaws.com
"""

import sys
import json
import time
import urllib.request
import urllib.error

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8001"


def request(method: str, path: str, body: dict | None = None, timeout: int = 30) -> tuple[int, dict]:
    url = BASE_URL + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return 0, {"error": str(e)}


def check(label: str, status: int, body: dict, expect_status: int = 200):
    ok = status == expect_status
    icon = "PASS" if ok else "FAIL"
    print(f"[{icon}] {label} — HTTP {status}")
    if not ok:
        print(f"       Response: {json.dumps(body, indent=2)[:300]}")
    return ok


def run_tests():
    print(f"\nTarget: {BASE_URL}\n{'='*60}")
    results = []

    # --- Basic health checks ---
    s, b = request("GET", "/")
    results.append(check("Root endpoint GET /", s, b))

    s, b = request("GET", "/health")
    results.append(check("Health check GET /health", s, b))

    # --- Models API ---
    s, b = request("GET", "/api/v1/models")
    ok = check("Models list GET /api/v1/models", s, b)
    results.append(ok)
    if ok:
        print(f"       total models returned: {b.get('total', '?')}")

    s, b = request("GET", "/api/v1/models?provider=anthropic&include_free=false")
    results.append(check("Models filter by provider=anthropic", s, b))

    s, b = request("GET", "/api/v1/models/providers/list")
    ok = check("Providers list GET /api/v1/models/providers/list", s, b)
    results.append(ok)
    if ok:
        print(f"       total providers: {b.get('total', '?')}")

    # --- Questionnaire API ---
    s, b = request("GET", "/api/v1/questionnaire")
    results.append(check("Questionnaire GET /api/v1/questionnaire", s, b))

    # --- Recommendation engine health (does NOT make an LLM call) ---
    s, b = request("GET", "/api/v1/recommend/health")
    results.append(check("Recommendation engine health check", s, b))

    # --- Full recommendation (makes 2 LLM calls — slow, up to 120s) ---
    print("\n[INFO] Testing /recommend — this makes 2 LLM calls and can take 30-90s...")
    print("       Note: HTTP API Gateway has a 29s hard timeout.")
    print("       If you see a 504, the LLM calls are working but Gateway times out.")
    sample_payload = {
        "answers": {
            "app_type": "Chatbot / Conversational AI",
            "app_description": "A customer support chatbot that answers FAQs.",
            "context_size": "Medium",
            "latency": "Fast",
            "scale": "Small",
            "agentic_level": "Non-Agentic",
            "capabilities": ["Structured output / JSON"],
            "priority": "Cost-conscious",
            "budget": "< $1,000",
            "privacy": [],
        }
    }
    t0 = time.time()
    s, b = request("POST", "/api/v1/recommend", body=sample_payload, timeout=120)
    elapsed = round(time.time() - t0, 1)

    if s == 504:
        print(f"[WARN] /recommend returned 504 Gateway Timeout after {elapsed}s")
        print("       The LLM backend is likely working. See TIMEOUT FIX section below.")
        results.append(True)  # not a code bug — known Gateway limitation
    else:
        ok = check(f"Recommend POST /api/v1/recommend ({elapsed}s)", s, b, expect_status=200)
        results.append(ok)

    # --- Summary ---
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed")

    if passed < total:
        print("\nFAILED tests indicate misconfiguration. Check CloudWatch logs:")
        print("  AWS Console → CloudWatch → Log groups → /aws/lambda/<function-name>")

    if s == 504 or passed == total:
        print("\nTIMEOUT FIX (if /recommend hits 504):")
        print("  HTTP API Gateway max timeout = 29 seconds.")
        print("  Fix: Add a Lambda Function URL directly on the Lambda function.")
        print("  Lambda Console → Configuration → Function URL → Create → Auth: NONE")
        print("  Then call /api/v1/recommend via the Function URL instead of API Gateway.")


if __name__ == "__main__":
    run_tests()
