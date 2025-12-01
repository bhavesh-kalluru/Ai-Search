# AI/ML Job Finder — Streamlit App
# Author: Bhavesh
# Description: Aggregates fresh AI/ML job openings (remote/onsite/intern/contract/part-time/full-time)
# by querying the Perplexity API for web search + OpenAI (optional) for robust JSON extraction.
# Works on macOS. No scraping of job sites; relies on Perplexity's browsing to collect current listings.

import os
import json
import time
import math
import textwrap
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pandas as pd
import requests
import streamlit as st

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="AI/ML Job Finder",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------
# Constants & Choices
# ----------------------------
DEFAULT_SITES = [
    "LinkedIn",
    "ZipRecruiter",
    "Dice",
    "Naukri",
    "Hired",
    "Wellfound (AngelList Talent)",
    "Indeed",
    "Glassdoor",
    "Greenhouse (company careers)",
    "Lever (company careers)",
    "Ashby (company careers)",
]

DOMAINS = [
    "General AI / GenAI",
    "Applied AI",
    "Machine Learning",
    "Deep Learning",
    "Data Engineering",
    "Data Analysis / Analytics",
]

EMPLOYMENT_TYPES = [
    "Full-time",
    "Part-time",
    "Contract",
    "Internship",
    "Temporary",
]

WORK_MODES = ["Remote", "Onsite", "Hybrid"]

EXPERIENCE_YEARS = [2, 3, 4, 5]

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    return url.strip().replace(" ", "%20")


def safe_get(d: Dict[str, Any], key: str, default: Any = "") -> Any:
    v = d.get(key, default)
    return v if v is not None else default


def build_query(
    years: List[int],
    domains: List[str],
    work_modes: List[str],
    emp_types: List[str],
    sites: List[str],
    extra_keywords: str,
    days_back: int,
) -> str:
    # Compose a rich, reproducible search instruction for Perplexity.
    # We ask for structured JSON back for easy parsing.
    since = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    yrs_txt = ", ".join([str(y) for y in years]) or "2, 3, 4, 5"
    dom_txt = ", ".join(domains) or ", ".join(DOMAINS)
    mode_txt = ", ".join(work_modes) or ", ".join(WORK_MODES)
    emp_txt = ", ".join(emp_types) or ", ".join(EMPLOYMENT_TYPES)
    site_txt = ", ".join(sites) or ", ".join(DEFAULT_SITES)

    extra = extra_keywords.strip()

    instruction = f"""
You are a job-search meta engine. Search the live web and list current, actively hiring job postings
for the AI/ML space. Focus on postings published ON or AFTER {since}.

CONSTRAINTS
- Experience required should include any of these exact years: {yrs_txt}.
- Relevant domains: {dom_txt}.
- Allowed work modes: {mode_txt}.
- Allowed employment types: {emp_txt}.
- Prioritize results from these sites and company career systems: {site_txt}.
- Include internships if available.
- Include REMOTE options alongside onsite/hybrid when available.
- Avoid duplicates; prefer canonical company career links when possible.

If helpful, you may also include strong company career pages directly.
{('Extra user keywords to match strictly: ' + extra) if extra else ''}

RESPONSE FORMAT (JSON ONLY)
Return a JSON object with key "results" mapping to a list of objects with EXACT keys:
[
  {{
    "company": str,               # canonical company name
    "role": str,                  # job title
    "experience_required": str,   # short text like "2+ years" or "3 years"
    "domain": str,                # one of: {', '.join(DOMAINS)}
    "work_mode": str,             # one of: Remote | Onsite | Hybrid
    "employment_type": str,       # one of: {', '.join(EMPLOYMENT_TYPES)}
    "location": str,              # city/state/country (if remote, note it)
    "posted_date": str,           # ISO 8601 or YYYY-MM-DD if known; else "unknown"
    "source": str,                # site or career system, e.g., "LinkedIn", "Greenhouse"
    "url": str                    # direct job link
  }}
]
Only JSON. No prose. 30-80 high-quality, diverse entries if possible.
"""
    return textwrap.dedent(instruction).strip()


def call_perplexity(prompt: str, api_key: str, model: str = "sonar-pro") -> Dict[str, Any]:

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "You are a precise web researcher that returns only valid JSON when asked."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=90)

    # <<< Add this block >>>
    if resp.status_code != 200:
        print("Status:", resp.status_code)
        print("Response text:", resp.text)
        raise requests.HTTPError(f"Perplexity error {resp.status_code}: {resp.text}")
    # <<< End block >>>

    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {"raw": data, "text": content}



def try_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        # Try to salvage a JSON block if the model added extra text accidentally.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
    return {}


def normalize_rows(obj: Dict[str, Any]) -> pd.DataFrame:
    results = obj.get("results", []) if isinstance(obj, dict) else []
    if not isinstance(results, list):
        results = []
    rows = []
    for r in results:
        if not isinstance(r, dict):
            continue
        rows.append({
            "Company": safe_get(r, "company"),
            "Role": safe_get(r, "role"),
            "Experience": safe_get(r, "experience_required"),
            "Domain": safe_get(r, "domain"),
            "Work Mode": safe_get(r, "work_mode"),
            "Employment Type": safe_get(r, "employment_type"),
            "Location": safe_get(r, "location"),
            "Posted": safe_get(r, "posted_date"),
            "Source": safe_get(r, "source"),
            "URL": normalize_url(safe_get(r, "url")),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        # Basic cleanup
        df = df.drop_duplicates(subset=["Company", "Role", "URL"], keep="first")
        # Move Remote roles to the top by default
        df["_remote_rank"] = df["Work Mode"].fillna("").str.lower().str.contains("remote").astype(int) * -1
        df = df.sort_values(["_remote_rank", "Company", "Role"]).drop(columns=["_remote_rank"])
    return df


def filter_dataframe(
    df: pd.DataFrame,
    years: List[int],
    domains: List[str],
    work_modes: List[str],
    emp_types: List[str],
    only_active_since_days: int,
) -> pd.DataFrame:
    if df.empty:
        return df

    # --- Experience filter (still strict) ---
    if years:
        pattern_parts = [fr"\b{y}\+?\s*year" for y in years]
        pattern = r"|".join(pattern_parts)
        df = df[df["Experience"].fillna("").str.lower().str.contains(pattern, regex=True)]

    # --- Domain filter (only if user selected a subset) ---
    if domains and len(domains) < len(DOMAINS):
        dom_pattern = r"|".join([d.split("/")[0].strip().lower() for d in domains])
        combined = (
            df["Domain"].fillna("").str.lower()
            + " " +
            df["Role"].fillna("").str.lower()
        )
        df = df[combined.str.contains(dom_pattern, regex=True)]

    # --- Work mode filter (only if subset) ---
    if work_modes and len(work_modes) < len(WORK_MODES):
        wm_set = {w.lower() for w in work_modes}
        df = df[df["Work Mode"].fillna("").str.lower().isin(wm_set)]

    # --- Employment type filter (only if subset) ---
    if emp_types and len(emp_types) < len(EMPLOYMENT_TYPES):
        et_set = {e.lower() for e in emp_types}
        df = df[df["Employment Type"].fillna("").str.lower().isin(et_set)]

    # --- Date filter ---
    if only_active_since_days > 0:
        cutoff = datetime.utcnow().date() - timedelta(days=only_active_since_days)

        def ok_date(s: str) -> bool:
            if not s or s == "unknown":
                return True  # keep unknowns
            try:
                d = datetime.fromisoformat(s.replace("Z", "")).date()
                return d >= cutoff
            except Exception:
                return True

        df = df[df["Posted"].apply(ok_date)]

    return df


# ----------------------------
# Sidebar — Settings
# ----------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("Store keys in **.streamlit/secrets.toml** for convenience.")

    perplexity_key = st.text_input(
        "Perplexity API Key",
        value=os.getenv("PERPLEXITY_API_KEY", ""),
        type="password",
        help="Required. Used to search the live web for job postings.",
    )
    openai_key = st.text_input(
        "OpenAI API Key (optional)",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Optional. Used only if we need help fixing/structuring JSON.",
    )

    st.divider()
    st.caption("Result controls")
    days_back = st.slider("Look back (days)", 1, 60, 14)
    batch_size = st.slider("Target results per search", 20, 100, 50, step=10)

    st.divider()
    st.caption("Default sites to prioritize")
    sites = st.multiselect("Sites", options=DEFAULT_SITES, default=DEFAULT_SITES)


# ----------------------------
# Main — Query Builder
# ----------------------------
st.title("🧠 AI/ML Job Finder")
st.write("Find current openings across AI/ML/GenAI, filtered by experience, work mode, and employment type.")

c1, c2, c3 = st.columns([1,1,1])
with c1:
    years = st.multiselect("Experience (years)", options=EXPERIENCE_YEARS, default=[2,3,4,5])
with c2:
    domains = st.multiselect("Domains", options=DOMAINS, default=DOMAINS)
with c3:
    work_modes = st.multiselect("Work mode", options=WORK_MODES, default=WORK_MODES)

c4, c5 = st.columns([1,1])
with c4:
    emp_types = st.multiselect("Employment type", options=EMPLOYMENT_TYPES, default=EMPLOYMENT_TYPES)
with c5:
    extra_keywords = st.text_input("Extra required keywords (optional)", placeholder="LLMs, LangChain, RAG, PyTorch, Databricks, etc.")

st.divider()

run = st.button("🔎 Search now", type="primary")

# ----------------------------
# Search & Results
# ----------------------------
if run:
    if not perplexity_key:
        st.error("Please provide your Perplexity API key in the sidebar.")
        st.stop()

    with st.spinner("Searching the live web via Perplexity..."):
        prompt = build_query(
            years=years,
            domains=domains,
            work_modes=work_modes,
            emp_types=emp_types,
            sites=sites,
            extra_keywords=extra_keywords,
            days_back=days_back,
        )
        try:
            resp = call_perplexity(prompt, perplexity_key)
            parsed = try_parse_json(resp.get("text", ""))
            if not parsed:
                st.warning("The response was not valid JSON. Attempting to fix...")
                # Optional: try to repair via OpenAI if key present
                if openai_key:
                    try:
                        fixed = requests.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {openai_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": "gpt-4o-mini",
                                "temperature": 0,
                                "messages": [
                                    {"role": "system", "content": "You convert messy text into exactly valid JSON, preserving data."},
                                    {"role": "user", "content": f"Fix the following into valid JSON only, no prose.\n\n{textwrap.shorten(resp.get('text',''), width=12000, placeholder='...')}"},
                                ],
                            },
                            timeout=60,
                        )
                        fixed.raise_for_status()
                        fixed_text = fixed.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                        parsed = try_parse_json(fixed_text)
                    except Exception as e:
                        st.info("Couldn't auto-fix JSON; showing raw output below.")
                        parsed = {}
                else:
                    parsed = {}
        except Exception as e:
            st.exception(e)
            st.stop()

    # Normalize into a table
    df = normalize_rows(parsed)

    # Apply in-app filters as a second pass
    df_filtered = filter_dataframe(
        df,
        years=years,
        domains=domains,
        work_modes=work_modes,
        emp_types=emp_types,
        only_active_since_days=days_back,
    )

    total = len(df)
    shown = len(df_filtered)
    st.subheader(f"Results ({shown}/{total})")

    if df_filtered.empty:
        st.info("No results matched your filters. Try broadening the query or increasing the lookback window.")
    else:
        # Render as clickable table
        def make_link(url):
            if not url:
                return ""
            return f"<a href='{url}' target='_blank'>Open</a>"

        out = df_filtered.copy()
        out["Link"] = out["URL"].apply(make_link)
        out = out.drop(columns=["URL"])  # keep a clean table, with a separate Open link

        # Reorder for readability
        cols = [
            "Company", "Role", "Experience", "Domain", "Work Mode", "Employment Type",
            "Location", "Posted", "Source", "Link"
        ]
        out = out.reindex(columns=cols)

        st.dataframe(out, use_container_width=True, hide_index=True)

        # Download
        csv = df_filtered.to_csv(index=False)
        st.download_button("⬇️ Download CSV", data=csv, file_name=f"aiml_jobs_{datetime.utcnow().date()}.csv", mime="text/csv")

        # Tips
        with st.expander("Tips to get richer results"):
            st.markdown(
                "- Add role-specific keywords (e.g., *RAG, LLMOps, MLOps, PyTorch, TensorFlow, Databricks, Snowflake*).\n"
                "- Keep both *Remote* and *Onsite* enabled to see hybrid options.\n"
                "- Increase *Look back (days)* for a wider net.\n"
                "- Prefer canonical company career links (Greenhouse/Lever/Ashby) when available."
            )

# ----------------------------
# Footer
# ----------------------------
st.caption(
    f"Last run: {_now_iso()} · This tool uses Perplexity for live web search and (optionally) OpenAI to clean JSON. "
)
