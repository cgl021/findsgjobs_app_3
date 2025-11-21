"""Job Search and Table page."""
import re
from typing import Dict

import pandas as pd
import streamlit as st

from api_client import fetch_jobs_from_endpoint


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Job Search - FindSGJobs",
    page_icon="🔍",
    layout="wide",
)

# ---------------- SESSION STATE INIT ----------------
for key, default in [
    ("full_jobs", []),           # list of full backend job dicts (inner "job")
    ("flat_jobs", []),           # simplified rows for UI table
    ("selected_job_idx", None),  # index into full_jobs / flat_jobs
    ("raw_response", None),      # raw API response for debugging
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------- HELPER FUNCTIONS ----------------
def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def get_job_description_text(job: Dict) -> str:
    """Try multiple variants that might hold the job description."""
    candidates = ["JobDescription", "Description", "job_description", "jobDesc"]
    for key in candidates:
        val = job.get(key)
        if val:
            if isinstance(val, dict):
                for sub in ["caption", "value", "text", "description"]:
                    subval = val.get(sub)
                    if isinstance(subval, str) and subval.strip():
                        return subval
            elif isinstance(val, str):
                return val
    return ""


# ---------------- GLOBAL STYLES ----------------
st.markdown(
    """
    <style>
    .big-title {font-size: 40px; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="big-title">🔍 Job Search</div>',
    unsafe_allow_html=True,
)


# ---------------- SIDEBAR – JOB SEARCH ----------------
st.sidebar.markdown("### Job Search Filters")

sidebar_job_title = st.sidebar.text_input("Keyword for Job Title", value="support")
sidebar_company = st.sidebar.text_input("Company", value="")
sidebar_min_salary = st.sidebar.number_input("Min Salary", min_value=0, value=0, step=100)
sidebar_mrt = st.sidebar.text_input("Nearest MRT Station", value="")
sidebar_emp_type = st.sidebar.text_input("Min Employment Type", value="")
sidebar_education = st.sidebar.text_input("Education", value="")


if st.sidebar.button("Fetch Jobs"):
    with st.spinner("Calling FindSGJobs backend…"):
        # backend still only takes a keyword; other filters applied client-side
        raw = fetch_jobs_from_endpoint(
            page=1,
            per_page=50,
            keywords=sidebar_job_title,
        )
    
    # Store raw response in session state for debugging
    st.session_state["raw_response"] = raw

    wrapped = raw.get("data", {}).get("result", []) if raw else []

    if not wrapped:
        st.warning("No jobs found in backend response.")
    else:
        full_jobs = []
        flat_jobs = []

        for row_idx, item in enumerate(wrapped):
            job = item.get("job", {}) or {}
            company_obj = item.get("company", {}) or {}
            full_jobs.append(job)

            title = job.get("Title", "") or ""
            company_name = company_obj.get("CompanyName", "") or ""

            # Nearest MRT
            mrt_list = job.get("id_Job_NearestMRTStation", []) or []
            mrt_caps = []
            if isinstance(mrt_list, list):
                for m in mrt_list:
                    if isinstance(m, dict):
                        cap = m.get("caption", "")
                        if cap:
                            mrt_caps.append(cap)
            nearest_mrt = ", ".join(mrt_caps)

            # Salary
            salary_not_display = job.get("id_Job_Donotdisplaysalary", 0)
            salary_range = ""
            min_salary_numeric = None

            if not salary_not_display:
                sr = job.get("Salaryrange") or {}
                currency_obj = job.get("id_Job_Currency") or {}
                interval_obj = job.get("id_Job_Interval") or {}
                currency = currency_obj.get("caption", "SGD")
                interval = interval_obj.get("caption", "Month")
                if isinstance(sr, dict) and sr.get("caption"):
                    salary_range = f"{currency} {sr['caption']} per {interval}"
                    nums = re.findall(r"\d[\d,]*", sr["caption"])
                    if nums:
                        min_salary_numeric = int(nums[0].replace(",", ""))
                else:
                    min_sal = job.get("id_Job_Salary")
                    max_sal = job.get("id_Job_MaxSalary")
                    if min_sal:
                        try:
                            min_salary_numeric = int(min_sal)
                        except Exception:
                            pass
                    if min_sal and max_sal:
                        salary_range = f"{currency} {min_sal}–{max_sal} per {interval}"
                    elif min_sal:
                        salary_range = f"{currency} {min_sal}+ per {interval}"
                    elif max_sal:
                        salary_range = f"{currency} up to {max_sal} per {interval}"

            # Employment type
            emp_list = job.get("EmploymentType", []) or []
            emp_caps = []
            if isinstance(emp_list, list):
                for et in emp_list:
                    if isinstance(et, dict):
                        cap = et.get("caption", "")
                        if cap:
                            emp_caps.append(cap)
            employment_type = ", ".join(emp_caps)

            # Min education & experience
            min_edu_obj = job.get("MinimumEducationLevel") or {}
            min_exp_obj = job.get("MinimumYearsofExperience") or {}
            min_edu = min_edu_obj.get("caption", "") or ""
            min_exp = min_exp_obj.get("caption", "") or ""

            # Separate JD only (API may not expose Requirements/Skills)
            desc_html = get_job_description_text(job)
            desc_plain = strip_html(desc_html)

            flat_jobs.append(
                {
                    "job_id": str(job.get("sid") or job.get("id") or f"job-{row_idx}"),
                    "Title": title,
                    "Company": company_name,
                    "Nearest MRT": nearest_mrt,
                    "Salary Range": salary_range,
                    "Employment Type": employment_type,
                    "Min Education": min_edu,
                    "Min Experience": min_exp,
                    "Job Description": desc_plain,
                    "Min Salary (numeric)": min_salary_numeric,
                }
            )

        # Apply sidebar filters client-side
        df_all = pd.DataFrame(flat_jobs)

        if sidebar_company:
            df_all = df_all[
                df_all["Company"].str.contains(sidebar_company, case=False, na=False)
            ]
        if sidebar_min_salary > 0 and "Min Salary (numeric)" in df_all.columns:
            df_all = df_all[
                df_all["Min Salary (numeric)"].fillna(0) >= sidebar_min_salary
            ]
        if sidebar_mrt:
            df_all = df_all[
                df_all["Nearest MRT"].str.contains(sidebar_mrt, case=False, na=False)
            ]
        if sidebar_emp_type:
            df_all = df_all[
                df_all["Employment Type"].str.contains(sidebar_emp_type, case=False, na=False)
            ]
        if sidebar_education:
            df_all = df_all[
                df_all["Min Education"].str.contains(sidebar_education, case=False, na=False)
            ]

        # Push filtered result into session state
        st.session_state["full_jobs"] = full_jobs
        st.session_state["flat_jobs"] = df_all.to_dict(orient="records")
        st.session_state["selected_job_idx"] = None

        st.success(f"Fetched {len(full_jobs)} jobs (displaying {len(df_all)} after filters).")


# ---------------- MAIN CONTENT: JOB TABLE ----------------
st.subheader("Fetched Jobs")

flat_jobs = st.session_state["flat_jobs"]

if flat_jobs:
    df = pd.DataFrame(flat_jobs)
    display_cols = [
        "Title",
        "Company",
        "Nearest MRT",
        "Salary Range",
        "Employment Type",
        "Min Education",
        "Min Experience",
        "Job Description",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    st.markdown("**Select a role that interest you via buttons below (carried into Gap Analysis by default):**")
    for i, row in df.iterrows():
        btn_col, info_col = st.columns([1, 6])
        with btn_col:
            if st.button("Select", key=f"select-row-{i}"):
                st.session_state["selected_job_idx"] = int(i)
                st.success(f"Selected: {row['Title']} — {row['Company']}")
        with info_col:
            desc = row.get("Job Description", "") or "-"
            st.markdown(
                f"**{row['Title']}** — {row['Company']}  \n"
                f"Nearest MRT: {row.get('Nearest MRT', '') or '-'} | "
                f"Salary: {row.get('Salary Range', '') or '-'} | "
                f"Employment Type: {row.get('Employment Type', '') or '-'} | "
                f"Min Education: {row.get('Min Education', '') or '-'} | "
                f"Min Experience: {row.get('Min Experience', '') or '-'}"
            )
            st.markdown(f"**Job Description:** {desc}")
        st.markdown("---")
else:
    st.info("Use the filters in the sidebar to search for jobs.")

# --- Debug expanders moved to bottom ---
if "raw_response" in st.session_state and st.session_state["raw_response"]:
    with st.expander("🔍 Debug: Raw backend response"):
        st.json(st.session_state["raw_response"])
    
    wrapped = st.session_state["raw_response"].get("data", {}).get("result", [])
    if wrapped:
        first_job_full = wrapped[0]               # the whole wrapper (item)
        first_job = first_job_full.get("job", {}) # the "job" dictionary inside

        with st.expander("🔍 Debug: Job fields that may contain requirements"):
            st.write("Job-level keys:", list(first_job.keys()))

            st.write("Possible requirement-like fields inside job:")
            for k, v in first_job.items():
                if any(tok in k.lower() for tok in ["require", "qualif", "about", "responsib"]):
                    st.write(f"• {k} → {type(v).__name__}")
                    st.json(v)

            st.write("Wrapper-level keys:", list(first_job_full.keys()))
            st.write("Possible requirement-like fields in wrapper:")
            for k, v in first_job_full.items():
                if any(tok in k.lower() for tok in ["require", "qualif", "about", "responsib"]):
                    st.write(f"• {k} → {type(v).__name__}")
                    st.json(v)
