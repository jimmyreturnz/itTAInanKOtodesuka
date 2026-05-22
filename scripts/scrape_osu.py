"""
scripts/scrape_osu.py

Downloads ranked osu! taiko beatmaps from the osu! API v2.

Steps:
  1. Authenticate with osu! API (client credentials OAuth)
  2. Paginate through ranked taiko beatmapsets
  3. Save star ratings + metadata to star_ratings.json
  4. Download .osz files and extract .osu + audio to data/raw/

Usage:
    python scripts/scrape_osu.py \
        --client-id YOUR_ID \
        --client-secret YOUR_SECRET \
        --output data/raw \
        --limit 10000

Get API credentials at: https://osu.ppy.sh/home/account/edit → OAuth
"""

from __future__ import annotations
import argparse
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not installed. Run: pip install requests")


# ---------------------------------------------------------------------------
# osu! API v2 client
# ---------------------------------------------------------------------------

API_BASE = "https://osu.ppy.sh/api/v2"
TOKEN_URL = "https://osu.ppy.sh/oauth/token"


class OsuApiClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id     = client_id
        self.client_secret = client_secret
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0
        self.session = requests.Session() if REQUESTS_AVAILABLE else None

    def _ensure_token(self):
        if time.time() < self._token_expiry - 60:
            return
        resp = self.session.post(TOKEN_URL, json={
            "client_id":     self.client_id,
            "client_secret": self.client_secret,
            "grant_type":    "client_credentials",
            "scope":         "public",
        })
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expiry = time.time() + data["expires_in"]
        self.session.headers.update({"Authorization": f"Bearer {self._token}"})

    def get(self, endpoint: str, params: dict = None, retries: int = 3) -> dict:
        self._ensure_token()
        for attempt in range(retries):
            try:
                resp = self.session.get(f"{API_BASE}{endpoint}", params=params, timeout=30)
                if resp.status_code == 429:
                    # Rate limited
                    wait = int(resp.headers.get("Retry-After", 60))
                    print(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return {}

    def download(self, url: str, dest: Path) -> bool:
        self._ensure_token()
        try:
            resp = self.session.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class TaikoScraper:
    def __init__(
        self,
        client: OsuApiClient,
        output_dir: str | Path,
        limit: int = 10_000,
        min_star: float = 1.0,
        max_star: float = 12.0,
    ):
        self.client     = client
        self.output_dir = Path(output_dir)
        self.limit      = limit
        self.min_star   = min_star
        self.max_star   = max_star

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sr_db_path = self.output_dir.parent / "star_ratings.json"

    def scrape_metadata(self) -> dict[int, float]:
        """
        Paginate through ranked taiko beatmapsets and collect star ratings.
        Returns dict: {beatmap_id: star_rating}
        """
        # Load existing DB
        sr_db: dict[int, float] = {}
        if self.sr_db_path.exists():
            sr_db = json.loads(self.sr_db_path.read_text())
            sr_db = {int(k): v for k, v in sr_db.items()}
            print(f"Loaded {len(sr_db)} existing star ratings.")

        total      = 0
        cursor_str = None

        while total < self.limit:
            params = {
                "m":           "taiko",    # mode = taiko
                "s":           "ranked",   # status = ranked
                "nsfw":        "false",
                "sort":        "ranked_desc",
            }
            if cursor_str:
                params["cursor_string"] = cursor_str

            data = self.client.get("/beatmapsets/search", params=params)

            beatmapsets = data.get("beatmapsets", [])
            if not beatmapsets:
                print("No more beatmapsets.")
                break

            for bms in beatmapsets:
                for bm in bms.get("beatmaps", []):
                    if bm.get("mode") != "taiko":
                        continue
                    bid = bm["id"]
                    sr  = bm.get("difficulty_rating", 0.0)
                    if self.min_star <= sr <= self.max_star:
                        sr_db[bid] = sr
                        total += 1

            cursor_str = data.get("cursor_string")
            if not cursor_str:
                break

            print(f"Scraped {total} beatmaps so far...")
            time.sleep(0.5)  # be polite to the API

        # Save updated DB
        self.sr_db_path.write_text(json.dumps(sr_db, indent=2))
        print(f"Saved {len(sr_db)} star ratings to {self.sr_db_path}")
        return sr_db

    def download_beatmapsets(self, sr_db: dict[int, float], max_sets: int = 5000):
        """
        Download .osz files for beatmapsets that have taiko difficulties.
        Extracts audio + .osu files into data/raw/<beatmapset_id>/
        """
        # Collect unique beatmapset IDs we need
        # We need to re-fetch beatmapset IDs from the star rating DB
        # For this we use beatmap lookup in batches

        beatmap_ids = list(sr_db.keys())
        print(f"Downloading beatmapsets for {len(beatmap_ids)} beatmaps...")

        # Map beatmap_id → beatmapset_id (fetch in batches of 50)
        bm_to_bms: dict[int, int] = {}
        for i in range(0, min(len(beatmap_ids), max_sets * 5), 50):
            batch = beatmap_ids[i:i+50]
            params = {"ids[]": batch}
            data = self.client.get("/beatmaps", params=params)
            for bm in data.get("beatmaps", []):
                bm_to_bms[bm["id"]] = bm["beatmapset_id"]
            time.sleep(0.3)

        # Unique beatmapset IDs
        beatmapset_ids = list(set(bm_to_bms.values()))[:max_sets]
        print(f"Downloading {len(beatmapset_ids)} beatmapsets...")

        for i, bms_id in enumerate(beatmapset_ids):
            dest_dir = self.output_dir / str(bms_id)
            if dest_dir.exists():
                continue  # already downloaded

            osz_path = self.output_dir / f"{bms_id}.osz"
            url = f"https://osu.ppy.sh/beatmapsets/{bms_id}/download?noVideo=1"

            print(f"[{i+1}/{len(beatmapset_ids)}] Downloading {bms_id}...")
            if not self.client.download(url, osz_path):
                continue

            # Extract .osz (it's a zip)
            try:
                self._extract_osz(osz_path, dest_dir)
                osz_path.unlink()  # remove .osz after extraction
            except zipfile.BadZipFile:
                print(f"Bad zip: {osz_path}")
                osz_path.unlink()
                continue

            time.sleep(1.0)  # rate limiting: 1 download per second

    def _extract_osz(self, osz_path: Path, dest_dir: Path):
        """Extract only .osu files and audio from an .osz archive."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(osz_path, "r") as z:
            for name in z.namelist():
                lower = name.lower()
                if lower.endswith(".osu") or any(
                    lower.endswith(ext) for ext in (".mp3", ".ogg", ".wav", ".flac")
                ):
                    z.extract(name, dest_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape osu! taiko beatmaps")
    parser.add_argument("--client-id",     required=True, help="osu! API client ID")
    parser.add_argument("--client-secret", required=True, help="osu! API client secret")
    parser.add_argument("--output",        default="data/raw",  help="Output directory")
    parser.add_argument("--limit",         type=int, default=10_000, help="Max beatmaps to scrape metadata for")
    parser.add_argument("--max-sets",      type=int, default=5_000,  help="Max beatmapsets to download")
    parser.add_argument("--min-star",      type=float, default=1.0)
    parser.add_argument("--max-star",      type=float, default=12.0)
    parser.add_argument("--metadata-only", action="store_true", help="Only scrape metadata, don't download")
    args = parser.parse_args()

    if not REQUESTS_AVAILABLE:
        print("Install requests: pip install requests")
        return

    client  = OsuApiClient(args.client_id, args.client_secret)
    scraper = TaikoScraper(
        client=client,
        output_dir=args.output,
        limit=args.limit,
        min_star=args.min_star,
        max_star=args.max_star,
    )

    print("=== Phase 1: Scraping metadata + star ratings ===")
    sr_db = scraper.scrape_metadata()

    if not args.metadata_only:
        print("=== Phase 2: Downloading beatmapsets ===")
        scraper.download_beatmapsets(sr_db, max_sets=args.max_sets)

    print("Done.")


if __name__ == "__main__":
    main()
