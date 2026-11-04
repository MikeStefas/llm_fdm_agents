import os
import vcr

CACHE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "cache",
)
CASSETTE_DIR = os.path.join(CACHE_ROOT, "vcr_cassettes")
os.makedirs(CASSETTE_DIR, exist_ok=True)

# Single shared VCR instance for the entire harness
my_vcr = vcr.VCR(
    cassette_library_dir=CASSETTE_DIR,
    record_mode="new_episodes",
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
    filter_headers=["authorization", "user-agent"],
)

__all__ = ["my_vcr", "CASSETTE_DIR"]

