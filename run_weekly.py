"""
run_weekly.py - weekly Flipkart phone pipeline (Realme + Samsung)
==================================================================
scrape -> clean -> VALIDATE -> save dated snapshot  (Power BI reads the snapshots folder)

Folder layout created next to this script:
    snapshots/   Mobile_Phones_YYYY-MM-DD.csv   <- Power BI "Folder" connector reads this
    raw/         <brand>_raw_YYYY-MM-DD.csv     <- untouched scrape, for debugging
    rejected/    Mobile_Phones_YYYY-MM-DD_REJECTED.csv  <- data that FAILED validation
    logs/        pipeline.log

Exit code 0 = snapshot saved, 1 = nothing saved (see logs/pipeline.log).
Re-running on the same day overwrites that day's snapshot (idempotent, no duplicates).

Install:  pip install selenium beautifulsoup4 pandas numpy
Run:      python run_weekly.py
"""
import logging
import random
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
BRANDS = {
    # Tip: tick the Brand filter on Flipkart and paste that URL here (keep page={page}).
    "Realme":  "https://www.flipkart.com/search?q=realme+mobile&page={page}",
    "SAMSUNG": "https://www.flipkart.com/search?q=samsung+phone&page={page}",
}
MAX_PAGES = 80
EMPTY_LIMIT = 4            # stop after this many pages in a row with no phones of the brand
HEADLESS = False           # Flipkart can block headless Chrome; keep False unless it works for you

# validation thresholds
MIN_ROWS_PER_BRAND = 30
MAX_DROP_VS_LAST = 0.40    # fail if a brand loses more than 40% of rows vs the previous snapshot
MAX_MISSING = {            # fail if more than this share of rows is blank in a column
    "Battery_in_mAh": 0.30, "Display_size_cm": 0.30, "RAM": 0.30, "ROM": 0.30,
    "Camera": 0.50, "Rating": 0.60,
}

# Class names from the Flipkart search page (list layout). Only edit here if Flipkart changes them.
CLS = {
    "card": "ZFwe0M", "title": "RG5Slk", "rating": "MKiFS6", "rr": "PvbNMB",
    "specs": "DTBslk", "price": "hZ3P6w", "mrp": "kRYCnD", "disc": "HQe8jr",
}

BASE = Path(__file__).resolve().parent
SNAP_DIR, RAW_DIR, REJ_DIR, LOG_DIR = (BASE / d for d in ("snapshots", "raw", "rejected", "logs"))
log = logging.getLogger("pipeline")
driver = None


# ----------------------------------------------------------------------------
# EXTRACT
# ----------------------------------------------------------------------------
def start_driver():
    """(Re)start Chrome. Called again automatically if the window closes or crashes."""
    global driver
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    try:
        driver.quit()
    except Exception:
        pass
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1400,1000")
    driver = webdriver.Chrome(options=opts)


def load(url, wait=(4, 6), tries=3):
    from selenium.common.exceptions import WebDriverException
    for attempt in range(1, tries + 1):
        try:
            driver.get(url)
            time.sleep(random.uniform(*wait))
            for y in (800, 1600, 2400, 3200, 4000):      # scroll so lazy-loaded cards/images render
                driver.execute_script(f"window.scrollTo(0, {y});")
                time.sleep(0.4)
            return BeautifulSoup(driver.page_source, "html.parser")
        except WebDriverException as e:
            log.warning("Chrome problem (%s) - restarting, try %d/%d", type(e).__name__, attempt, tries)
            start_driver()
    raise RuntimeError("Chrome kept failing")


def txt(card, tag, cls, sep=" "):
    el = card.find(tag, class_=cls)
    return el.get_text(sep, strip=True) if el else np.nan


def find_image(card):
    """Phone photo sits just outside the row: climb until found (never past one phone)."""
    el = card
    for _ in range(6):
        el = el.parent
        if el is None or len(el.find_all("div", class_=CLS["card"])) != 1:
            break
        img = el.find("img", src=re.compile("rukminim"))
        if img:
            return img["src"]
    return np.nan


def parse_page(soup):
    rows = []
    for card in soup.find_all("div", class_=CLS["card"]):
        specs = [li.get_text(" ", strip=True) for li in card.find_all("li", class_=CLS["specs"])]
        if not specs:                       # banner / non-phone card
            continue
        pick = lambda *kws: next((s for s in specs if any(k in s for k in kws)), np.nan)
        rows.append({
            "Model": txt(card, "div", CLS["title"]),
            "Number of rating and reviews": txt(card, "span", CLS["rr"]),
            "original_price": txt(card, "div", CLS["mrp"], sep=""),
            "Discounted_Price": txt(card, "div", CLS["price"]),
            "Discount": txt(card, "div", CLS["disc"]),
            "Rating": txt(card, "div", CLS["rating"]),
            "RAM_ROM": pick("RAM", "ROM"),
            "Display": pick("Display"),
            "Camera": pick("Camera"),
            "Battery": pick("mAh"),
            "product_image": find_image(card),
        })
    return rows


def scrape_brand(brand, url):
    rows, empty = [], 0
    for page in range(1, MAX_PAGES + 1):
        cards = parse_page(load(url.format(page=page)))
        mine = [r for r in cards if brand.lower() in str(r["Model"]).lower()]
        log.info("[%s] page %d: %d phones (%d other-brand skipped) | total %d",
                 brand, page, len(mine), len(cards) - len(mine), len(rows) + len(mine))
        if mine:
            rows += mine
            empty = 0
        else:
            empty += 1
            if empty >= EMPTY_LIMIT:
                break
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# TRANSFORM
# ----------------------------------------------------------------------------
def num(s):
    if pd.isna(s):
        return np.nan
    m = re.sub(r"[^\d.]", "", str(s))
    return float(m) if m else np.nan


def clean(raw, brand, scraped_on):
    df = raw.copy()
    df["brand"] = brand
    df["Discounted_Price"] = df["Discounted_Price"].map(num)
    df["original_price"] = df["original_price"].map(num)
    df = df.dropna(subset=["Discounted_Price"])
    df["original_price"] = df["original_price"].fillna(df["Discounted_Price"])   # no discount -> same price

    d = df["Discount"].astype(str).str.extract(r"(\d+)\s*%")[0].astype(float)
    calc = ((df["original_price"] - df["Discounted_Price"]) / df["original_price"] * 100).round()
    df["Discount"] = d.fillna(calc).fillna(0).astype(int)

    rr = df["Number of rating and reviews"].astype(str)
    df["No_of_Ratings"] = rr.str.extract(r"([\d,]+)\s*Ratings?")[0].str.replace(",", "").astype(float)
    df["No_of_Reviews"] = rr.str.extract(r"([\d,]+)\s*Reviews?")[0].str.replace(",", "").astype(float)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    df["Battery_in_mAh"] = df["Battery"].astype(str).str.extract(r"(\d+)\s*mAh")[0].astype(float)
    df["Display_size_cm"] = df["Display"].astype(str).str.extract(r"([\d.]+)\s*cm")[0].astype(float)
    mp = df["Camera"].astype(str).str.extract(r"(\d+(?:\.\d+)?)\s*MP")[0]       # first (rear) camera
    df["Camera"] = mp + "MP"
    df["RAM"] = df["RAM_ROM"].astype(str).str.extract(r"(\d+\s*(?:GB|MB))\s*RAM")[0]
    df["ROM"] = df["RAM_ROM"].astype(str).str.extract(r"(\d+\s*(?:GB|TB|MB))\s*ROM")[0]

    m = df["Model"].astype(str)
    df["model"] = m.str.split("(").str[0].str.strip()
    df["Color"] = m.str.extract(r"\(([^,)]+)")[0].str.strip()

    cols = ["model", "brand", "Color", "RAM", "ROM", "original_price", "Discount", "Discounted_Price",
            "Display_size_cm", "Battery_in_mAh", "Camera", "Rating", "No_of_Ratings", "No_of_Reviews",
            "product_image"]
    out = df[cols].drop_duplicates(subset=cols[:-1]).reset_index(drop=True)
    out["original_price"] = out["original_price"].astype(int)
    out["Discounted_Price"] = out["Discounted_Price"].astype(int)
    out["scraped_on"] = scraped_on
    return out


# ----------------------------------------------------------------------------
# VALIDATE
# ----------------------------------------------------------------------------
def previous_snapshot(today):
    files = sorted(p for p in SNAP_DIR.glob("Mobile_Phones_*.csv") if p.stem[-10:] < today)
    return pd.read_csv(files[-1]) if files else None


def validate(df, prev):
    """Return (errors, warnings). Any error means the snapshot is NOT saved."""
    errors, warnings = [], []
    if df["model"].isna().any() or (df["model"].astype(str).str.strip() == "").any():
        errors.append("empty model names found")
    if (df["Discounted_Price"] <= 0).any():
        errors.append("price <= 0 found")
    if (df["original_price"] < df["Discounted_Price"]).any():
        errors.append("original price lower than selling price found")
    if not df["Discount"].between(0, 100).all():
        errors.append("discount outside 0-100")
    if not df["Rating"].dropna().between(1, 5).all():
        errors.append("rating outside 1-5")

    for brand, g in df.groupby("brand"):
        if len(g) < MIN_ROWS_PER_BRAND:
            errors.append(f"{brand}: only {len(g)} rows (minimum {MIN_ROWS_PER_BRAND})")
        for col, limit in MAX_MISSING.items():     # blank columns usually mean Flipkart changed its HTML
            share = g[col].isna().mean()
            if share > limit:
                errors.append(f"{brand}: {col} is blank in {share:.0%} of rows (limit {limit:.0%}) - page layout changed?")
        if prev is not None:
            before = (prev["brand"] == brand).sum()
            if before:
                change = len(g) / before - 1
                if change < -MAX_DROP_VS_LAST:
                    errors.append(f"{brand}: rows fell {change:.0%} vs last snapshot ({before} -> {len(g)})")
                elif change > 0.5:
                    warnings.append(f"{brand}: rows grew {change:.0%} vs last snapshot ({before} -> {len(g)})")
    return errors, warnings


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)


def main():
    setup_logging()
    today = date.today().isoformat()
    log.info("===== run started %s =====", today)
    for d in (SNAP_DIR, RAW_DIR, REJ_DIR):
        d.mkdir(exist_ok=True)

    frames = []
    try:
        start_driver()
        for brand, url in BRANDS.items():
            raw = scrape_brand(brand, url)
            raw.to_csv(RAW_DIR / f"{brand.lower()}_raw_{today}.csv", index=False)
            if raw.empty:
                log.error("[%s] scraped 0 phones - skipping (selectors stale or blocked?)", brand)
                continue
            frames.append(clean(raw, brand, today))
    except Exception:
        log.exception("scrape failed")
        return 1
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    if not frames:
        log.error("nothing scraped - no snapshot written")
        return 1
    phones = pd.concat(frames, ignore_index=True)

    errors, warnings = validate(phones, previous_snapshot(today))
    for w in warnings:
        log.warning("VALIDATION WARNING: %s", w)
    if errors:
        for e in errors:
            log.error("VALIDATION FAILED: %s", e)
        phones.to_csv(REJ_DIR / f"Mobile_Phones_{today}_REJECTED.csv", index=False)
        log.error("snapshot NOT saved; data kept in rejected/ for inspection")
        return 1

    out = SNAP_DIR / f"Mobile_Phones_{today}.csv"
    phones.to_csv(out, index=False)
    log.info("saved %s (%d rows)", out.name, len(phones))
    for brand, g in phones.groupby("brand"):
        log.info("  %s: %d rows | avg price %.0f | avg battery %.0f mAh | avg rating %.2f",
                 brand, len(g), g["Discounted_Price"].mean(), g["Battery_in_mAh"].mean(), g["Rating"].mean())
    return 0


if __name__ == "__main__":
    sys.exit(main())