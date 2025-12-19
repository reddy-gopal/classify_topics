import os
import re
import json
import time
import html
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse, parse_qs

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gspread
from google.genai import Client


# ------------------ DEFAULTS ------------------
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_BATCH_SIZE = 18 


# ------------------ URL PARSING ------------------
def parse_google_sheet_url(url: str) -> Tuple[str, int]:
    """
    Extract spreadsheetId and gid from a Google Sheet URL.
    """
    url = url.strip()

    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError("Could not find spreadsheetId in the URL.")
    spreadsheet_id = m.group(1)

    gid = None
    parsed = urlparse(url)

    if parsed.fragment:
        frag_qs = parse_qs(parsed.fragment)
        if "gid" in frag_qs and frag_qs["gid"]:
            gid = frag_qs["gid"][0]
        if gid is None:
            m2 = re.search(r"gid=(\d+)", parsed.fragment)
            if m2:
                gid = m2.group(1)

    if gid is None and parsed.query:
        q = parse_qs(parsed.query)
        if "gid" in q and q["gid"]:
            gid = q["gid"][0]

    if gid is None:
        raise ValueError("Could not find gid in the URL. Make sure the URL contains #gid=XXXX.")

    return spreadsheet_id, int(gid)


# ------------------ SHEET HELPERS ------------------
def colnum_to_a1(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def get_header_map(ws) -> Dict[str, int]:
    headers = ws.row_values(1)
    return {h: i + 1 for i, h in enumerate(headers) if h}


def get_last_data_row(ws) -> int:
    values = ws.get_all_values()
    last = 1
    for i, row in enumerate(values, start=1):
        if any(cell.strip() for cell in row):
            last = i
    return last


def read_batch(ws, header_to_col: Dict[str, int], start_row: int, end_row: int, cols: List[str]) -> List[Dict[str, Any]]:
    for c in cols:
        if c not in header_to_col:
            raise RuntimeError(f"Missing required column in sheet header row: '{c}'")

    col_idxs = [header_to_col[c] for c in cols]
    left_c, right_c = min(col_idxs), max(col_idxs)

    rng = f"{colnum_to_a1(left_c)}{start_row}:{colnum_to_a1(right_c)}{end_row}"
    values = ws.get(rng)

    expected = end_row - start_row + 1
    while len(values) < expected:
        values.append([])

    items = []
    for i, row_vals in enumerate(values):
        row_id = start_row + i
        row = {"row_id": row_id}
        for c in cols:
            idx = header_to_col[c] - left_c
            row[c] = row_vals[idx] if idx < len(row_vals) else ""
        items.append(row)

    return items


def batch_update_column(ws, header_to_col: Dict[str, int], start_row: int, end_row: int,
                        col_name: str, values: List[str]) -> Dict[str, Any]:
    col = header_to_col[col_name]
    a1 = f"{colnum_to_a1(col)}{start_row}:{colnum_to_a1(col)}{end_row}"
    return {"range": a1, "values": [[v] for v in values]}


# ------------------ INLINE IMAGE HTML ------------------
def img_tag(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    safe = html.escape(url, quote=True)
    return f"<p><img src=\"{safe}\" /></p>"


def append_image_html(text: str, image_url: str) -> str:
    """
    Put image inline as HTML at the end (so it remains in same cell).
    If the text already has an <img ...> tag, we don't add another.
    """
    t = text or ""
    u = (image_url or "").strip()
    if not u:
        return t
    if re.search(r"<\s*img\b", t, flags=re.IGNORECASE):
        return t
    return (t.rstrip() + "\n" + img_tag(u)).strip()


# ------------------ CLEANUP + GEMINI ------------------
def extract_json(text: str) -> Any:
    if not text:
        raise ValueError("Empty model response")

    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    try:
        return json.loads(t)
    except Exception:
        pass

    m = re.search(r"(\[\s*{.*}\s*\]|\{\s*\".*\"\s*:.*\})", t, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Could not locate JSON in response. First 300 chars:\n{t[:300]}")
    return json.loads(m.group(1))


def light_post_clean(s: str) -> str:
    """
    Important: keep HTML tags intact; only normalize whitespace and unescape entities.
    """
    if s is None:
        return ""
    s = str(s)

    # Unescape HTML entities (keeps tags)
    s = html.unescape(s)

    # Fix split words
    s = re.sub(r"([A-Za-z])[\t\r\n]+([A-Za-z])", r"\1\2", s)
    s = re.sub(r"([A-Za-z]) {2,}([A-Za-z])", r"\1\2", s)

    # Normalize whitespace but keep newlines
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def build_prompt(items: List[Dict[str, Any]]) -> str:
    payload = []
    for it in items:
        payload.append({
            "row_id": it["row_id"],
            "question_text": it.get("question_text", "") or "",
            "option_a": it.get("option_a", "") or "",
            "option_b": it.get("option_b", "") or "",
            "option_c": it.get("option_c", "") or "",
            "option_d": it.get("option_d", "") or "",
            "explanation": it.get("explanation", "") or "",
        })

    return f"""
You are an educational question processing engine.

DO BOTH TASKS:

1) CLEANING / FORMATTING
- Convert LaTeX/KaTeX tokens into clean, human-readable Unicode where appropriate.
  Examples:
  \\alpha→α, \\theta→θ, \\pi→π, \\rightarrow→→, \\Rightarrow→⇒, \\leq→≤, \\geq→≥,
  \\times→×, \\div→÷, x^2→x², x^3→x³, x_1→x₁,
  \\sqrt{{x}}→√x, \\sqrt{{a+b}}→√(a+b),
  \\frac{{a}}{{b}}→a/b (or a⁄b),
  \\degree or ^\\circ→°.
- Remove math wrappers like \\( \\), \\[ \\], $$ $$.
- Remove/convert raw HTML entities, BUT:
  IMPORTANT: If the input contains HTML tags like <p>...</p> or <img src="..."/>,
  KEEP them as HTML (do not delete tags). You may normalize spacing inside tags if needed.
- Fix split words caused by tabs/newlines like "resis\\tance" -> "resistance".
- Keep meaning unchanged.

2) CLASSIFICATION
- subject: short label
- topic: short + specific
- difficulty_level: exactly one of ["Easy","Medium","Hard"]

OUTPUT FORMAT (STRICT JSON ONLY):
Return ONLY valid JSON: an array of objects, same order as input.
Each object MUST be:
{{
  "row_id": <int>,
  "question_text": <string>,
  "option_a": <string>,
  "option_b": <string>,
  "option_c": <string>,
  "option_d": <string>,
  "explanation": <string>,
  "subject": <string>,
  "topic": <string>,
  "difficulty_level": "Easy"|"Medium"|"Hard"
}}

INPUT:
{json.dumps(payload, ensure_ascii=False)}
""".strip()


def gemini_call_json(client: Client, model: str, prompt: str, max_retries: int = 6) -> Any:
    base = 1.6
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            text = getattr(resp, "text", None) or ""
            return extract_json(text)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            sleep_s = (base ** attempt) + (0.25 * attempt)
            print(f"[WARN] Gemini error (attempt {attempt+1}/{max_retries}): {e}. Sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)


# ------------------ MAIN ------------------
def main():
    print("=== Gemini: In-place Clean + Classify (Google Sheet) ===\n")

    sheet_url = input("Paste Google Sheet URL (must include #gid=...):\n> ").strip()
    spreadsheet_id, gid = parse_google_sheet_url(sheet_url)
    print(f"\n[INFO] spreadsheetId: {spreadsheet_id}")
    print(f"[INFO] gid: {gid}\n")

    # Credentials auto-detect
    cred_path = None
    if os.path.exists("credentials.json"):
        cred_path = "credentials.json"
    elif os.path.exists("credential.json"):
        cred_path = "credential.json"

    if not cred_path:
        raise RuntimeError("Could not find credentials.json or credential.json in current folder.")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found. Set it in .env or environment variables.")

    # Fixed defaults (no prompts)
    model = DEFAULT_MODEL
    batch_size = DEFAULT_BATCH_SIZE

    # Sheets auth
    gc = gspread.service_account(filename=cred_path)
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.get_worksheet_by_id(gid)
    if ws is None:
        raise RuntimeError(f"Could not open worksheet with gid={gid}. Check the URL and permissions.")

    print(f"[INFO] Using credentials: {cred_path}")
    print(f"[INFO] Worksheet: {ws.title}")
    print(f"[INFO] Model: {model}")
    print(f"[INFO] Batch size: {batch_size}\n")

    header_to_col = get_header_map(ws)

    # Required text columns to update in-place
    text_cols = ["question_text", "option_a", "option_b", "option_c", "option_d", "explanation"]
    missing_text = [c for c in text_cols if c not in header_to_col]
    if missing_text:
        raise RuntimeError(f"Missing required columns: {missing_text}")

    # Classification columns must exist to write back (since you want in-place update)
    class_cols = ["subject", "topic", "difficulty_level"]
    missing_class = [c for c in class_cols if c not in header_to_col]
    if missing_class:
        raise RuntimeError(
            f"Missing classification columns in sheet header: {missing_class}. "
            f"Please add them once in the sheet header row."
        )

    # Optional image URL columns
    img_cols = {
        "question_text": "question_image_url",
        "option_a": "option_a_image_url",
        "option_b": "option_b_image_url",
        "option_c": "option_c_image_url",
        "option_d": "option_d_image_url",
        "explanation": "explanation_image_url",
    }

    # Which of those exist?
    existing_img_cols = {k: v for k, v in img_cols.items() if v in header_to_col}

    gemini = Client(api_key=api_key)

    last_row = get_last_data_row(ws)
    if last_row < 2:
        print("[DONE] No data rows found.")
        return

    start_row, end_row = 2, last_row
    print(f"[INFO] Processing rows {start_row} to {end_row} (total {end_row - start_row + 1})\n")

    # We will read text cols + any existing image url cols (for inline HTML injection)
    read_cols = text_cols + list(set(existing_img_cols.values()))

    cur = start_row
    while cur <= end_row:
        batch_end = min(end_row, cur + batch_size - 1)

        rows = read_batch(ws, header_to_col, cur, batch_end, read_cols)

        # Make items for Gemini with images injected into the corresponding text fields
        items_for_gemini = []
        for r in rows:
            # skip totally empty row (all text cols empty)
            if not any((r.get(c, "") or "").strip() for c in text_cols):
                continue

            enriched = {"row_id": r["row_id"]}
            for c in text_cols:
                val = r.get(c, "") or ""
                img_col = existing_img_cols.get(c)
                if img_col:
                    val = append_image_html(val, r.get(img_col, "") or "")
                enriched[c] = val
            items_for_gemini.append(enriched)

        if not items_for_gemini:
            print(f"[SKIP] Rows {cur}-{batch_end} empty.")
            cur = batch_end + 1
            continue

        prompt = build_prompt(items_for_gemini)
        parsed = gemini_call_json(gemini, model, prompt)

        if not isinstance(parsed, list) or len(parsed) != len(items_for_gemini):
            raise ValueError(
                f"Gemini returned unexpected JSON. Expected array length {len(items_for_gemini)}, "
                f"got {type(parsed)} length {len(parsed) if isinstance(parsed, list) else 'N/A'}"
            )

        by_row = {obj["row_id"]: obj for obj in parsed if isinstance(obj, dict) and "row_id" in obj}

        # Build aligned outputs for the FULL batch
        out_question = []
        out_a = []
        out_b = []
        out_c = []
        out_d = []
        out_expl = []
        out_subject = []
        out_topic = []
        out_diff = []

        for rid in range(cur, batch_end + 1):
            obj = by_row.get(rid)

            if not obj:
                # keep originals unchanged for rows we skipped (empty)
                out_question.append(rows[rid - cur].get("question_text", "") or "")
                out_a.append(rows[rid - cur].get("option_a", "") or "")
                out_b.append(rows[rid - cur].get("option_b", "") or "")
                out_c.append(rows[rid - cur].get("option_c", "") or "")
                out_d.append(rows[rid - cur].get("option_d", "") or "")
                out_expl.append(rows[rid - cur].get("explanation", "") or "")
                out_subject.append("")  # leave blank for empty row
                out_topic.append("")
                out_diff.append("")
                continue

            out_question.append(light_post_clean(obj.get("question_text", "")))
            out_a.append(light_post_clean(obj.get("option_a", "")))
            out_b.append(light_post_clean(obj.get("option_b", "")))
            out_c.append(light_post_clean(obj.get("option_c", "")))
            out_d.append(light_post_clean(obj.get("option_d", "")))
            out_expl.append(light_post_clean(obj.get("explanation", "")))
            out_subject.append(light_post_clean(obj.get("subject", "")))
            out_topic.append(light_post_clean(obj.get("topic", "")))
            out_diff.append(light_post_clean(obj.get("difficulty_level", "")))

        # Batch update in-place
        requests = [
            batch_update_column(ws, header_to_col, cur, batch_end, "question_text", out_question),
            batch_update_column(ws, header_to_col, cur, batch_end, "option_a", out_a),
            batch_update_column(ws, header_to_col, cur, batch_end, "option_b", out_b),
            batch_update_column(ws, header_to_col, cur, batch_end, "option_c", out_c),
            batch_update_column(ws, header_to_col, cur, batch_end, "option_d", out_d),
            batch_update_column(ws, header_to_col, cur, batch_end, "explanation", out_expl),
            batch_update_column(ws, header_to_col, cur, batch_end, "subject", out_subject),
            batch_update_column(ws, header_to_col, cur, batch_end, "topic", out_topic),
            batch_update_column(ws, header_to_col, cur, batch_end, "difficulty_level", out_diff),
        ]
        ws.batch_update(requests)

        print(f"[OK] Updated rows {cur}-{batch_end}")
        cur = batch_end + 1

    print("\n[DONE] Finished processing and updating the sheet in-place.")


if __name__ == "__main__":
    main()
