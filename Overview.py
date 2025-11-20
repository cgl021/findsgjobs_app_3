import re
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

from api_client import fetch_jobs_from_endpoint


# ---------------- PAGE CONFIG ----------------
import streamlit as st


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="FindSGJobs - Home",
    page_icon="🏠",
    layout="wide",
)

# ---------------- SESSION STATE INIT ----------------
for key, default in [
    ("full_jobs", []),           # list of full backend job dicts (inner "job")
    ("flat_jobs", []),           # simplified rows for UI table
    ("selected_job_idx", None),  # index into full_jobs / flat_jobs
    ("resume_text", ""),         # extracted text from uploaded resume
    ("analysis_text", ""),       # gap analysis + course recommendation
    ("job_match_pct", None),     # overall job match %
    ("keyword_coverage_pct", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------- GLOBAL STYLES ----------------
st.markdown(
    """
    <style>
    .big-title {font-size: 48px; font-weight: 700; color: #1f77b4;}
    .subtitle {font-size: 20px; color: #666; margin-bottom: 2rem;}
    .feature-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1.5rem;
    }
    .feature-title {font-size: 24px; font-weight: 600; margin-bottom: 0.5rem;}
    .feature-desc {font-size: 16px; color: #555;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="big-title">🏠 Welcome to FindSGJobs</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Your intelligent job search and resume matching assistant</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------- FEATURES ----------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
        <div class="feature-box">
            <div class="feature-title">🔍 Job Search</div>
            <div class="feature-desc">
                Browse and filter jobs from the FindSGJobs backend API. Search by:
                <ul>
                    <li>Job title keywords</li>
                    <li>Company name</li>
                    <li>Salary range</li>
                    <li>Nearest MRT station</li>
                    <li>Employment type</li>
                    <li>Education requirements</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature-box">
            <div class="feature-title">📊 Gap Analysis</div>
            <div class="feature-desc">
                Upload your resume and get instant analysis:
                <ul>
                    <li>Job match percentage</li>
                    <li>Keyword coverage analysis</li>
                    <li>Strengths and gaps identification</li>
                    <li>Personalized course recommendations</li>
                    <li>Tips to improve your fit</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------- GETTING STARTED ----------------
st.subheader("🚀 Getting Started")

st.markdown(
    """
    **Follow these simple steps:**
    
    1. **Navigate to Job Search** (in the sidebar) → Use filters to find relevant jobs
    2. **Select a job** from the table that interests you
    3. **Go to Gap Analysis** (in the sidebar) → Upload your resume
    4. **Click "Run Gap Analysis"** to see how well you match the job
    5. **Review recommendations** and take action to improve your profile
    """,
)

st.info("💡 **Tip:** Jobs are stored in session state, so you can switch between pages without losing your search results!")

st.markdown("---")

# ---------------- FOOTER ----------------
st.caption("Built with Streamlit • Powered by FindSGJobs API • Singapore Job Market 2025")

# ---------------- SESSION STATE INIT ----------------
for key, default in [
    ("full_jobs", []),           # list of full backend job dicts (inner "job")
    ("flat_jobs", []),           # simplified rows for UI table
    ("selected_job_idx", None),  # index into full_jobs / flat_jobs
    ("resume_text", ""),         # extracted text from uploaded resume
    ("analysis_text", ""),       # gap analysis + course recommendation
    ("job_match_pct", None),     # overall job match %
    ("keyword_coverage_pct", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------- HELPER FUNCTIONS ----------------
STOPWORDS = {
    "and", "the", "with", "for", "to", "of", "in", "on", "a", "an", "or",
    "be", "as", "by", "is", "are", "will", "able", "etc", "any", "all",
    "job", "role", "responsible", "responsibilities", "requirement",
    "requirements", "candidate", "candidates", "ability", "strong", "good",
    "skills", "experience", "experiences", "year", "years"
}


def extract_keywords(text: str, min_len: int = 3) -> set:
    tokens = re.findall(r"[A-Za-z]{%d,}" % min_len, text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def strip_html(text: str) -> str:
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


def extract_requirements_from_description(description: str) -> str:
    """Heuristic: pull out the 'Requirements' / 'Qualifications' section from a blob JD."""
    if not description:
        return ""

    text = description.replace("\r", "\n")

    # headings commonly used in SG JDs
    heading_pattern = r"(requirements|requirement|qualifications|about you|what you bring|who you are)"
    parts = re.split(heading_pattern, text, flags=re.IGNORECASE)

    # parts = [before, heading1, after1, heading2, after2, ...]
    if len(parts) < 3:
        return ""

    # take the first heading's content
    requirements_section = parts[2]

    # stop at next all-caps / colon / obvious header style if present
    requirements_section = re.split(r"\n[A-Z][A-Za-z0-9 /&]{3,}:\s*\n", requirements_section)[0]

    return requirements_section.strip()


def get_job_requirement_text(job: Dict) -> str:
    """
    Extract job requirements, prioritising known keys,
    then fuzzy match on any 'require*' / 'qualif*' keys (recursively),
    then fallback to carving them out of JobDescription.
    """
    if not job:
        return ""

    # 1) Prioritise known keys used previously
    for key in ["id_Job_Requirement", "id_Job_Requirements", "Job_Requirement", "Job_Requirements"]:
        val = job.get(key)
        if val:
            if isinstance(val, list):
                parts = []
                for item in val:
                    if isinstance(item, dict):
                        # try common subfields
                        for sub in [
                            "caption", "value", "text", "requirements",
                            "description", "JobRequirement", "JobRequirements"
                        ]:
                            subval = item.get(sub)
                            if isinstance(subval, str) and subval.strip():
                                parts.append(subval.strip())
                                break
                    elif isinstance(item, str) and item.strip():
                        parts.append(item.strip())
                if parts:
                    return "\n".join(parts)
            elif isinstance(val, dict):
                for sub in [
                    "caption", "value", "text", "requirements",
                    "description", "JobRequirement", "JobRequirements"
                ]:
                    subval = val.get(sub)
                    if isinstance(subval, str) and subval.strip():
                        return subval
            elif isinstance(val, str):
                return val

    # 2) Legacy string-style fields
    candidates = ["JobRequirement", "JobRequirements", "Requirement", "Requirements", "job_requirement"]
    for key in candidates:
        val = job.get(key)
        if val:
            if isinstance(val, dict):
                for sub in ["caption", "value", "text", "requirements"]:
                    subval = val.get(sub)
                    if isinstance(subval, str) and subval.strip():
                        return subval
            elif isinstance(val, str):
                return val

    # 3) Fuzzy recursive search: any key that *looks like* requirements / qualifications
    texts: List[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                k_low = k.lower()
                # keys like: id_Job_Requirement, RequirementDetail, candidateRequirements, minQualifications, AboutYou, etc.
                if any(tok in k_low for tok in ["require", "qualif", "about you", "what you bring", "who you are"]):
                    if isinstance(v, str) and v.strip():
                        texts.append(v.strip())
                    elif isinstance(v, (dict, list)):
                        walk(v)
                else:
                    if isinstance(v, (dict, list)):
                        walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(job)

    if texts:
        # de-duplicate while preserving order
        seen = set()
        uniq = []
        for t in texts:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return "\n".join(uniq)

    # 4) Fallback: carve from JobDescription blob
    desc = get_job_description_text(job)
    return extract_requirements_from_description(desc) or ""


def normalise_skill_text(s: str) -> str:
    """Lowercase and strip most punctuation for matching."""
    return re.sub(r"[^a-z0-9+.# ]", " ", s.lower()).strip()


def match_job_resume_skills(job_skills: List[str], resume_text: str) -> Tuple[List[str], List[str]]:
    """
    Match job skills (from id_Job_Skills) against resume text.
    Returns (matched_skills, missing_skills).
    """
    resume_norm = resume_text.lower()
    matched = []
    missing = []

    for raw_skill in job_skills:
        skill_str = str(raw_skill).strip()
        if not skill_str:
            continue
        skill_norm = normalise_skill_text(skill_str)
        if not skill_norm:
            continue

        # direct phrase match first
        is_match = skill_norm in resume_norm

        # fallback: any token > 2 chars appears
        if not is_match:
            tokens = [t for t in skill_norm.split() if len(t) > 2]
            if any(t in resume_norm for t in tokens):
                is_match = True

        if is_match:
            matched.append(skill_str)
        else:
            missing.append(skill_str)

    return matched, missing


def build_job_resume_overlap(job_text: str, resume_text: str):
    job_kw = extract_keywords(job_text)
    cv_kw = extract_keywords(resume_text)

    overlap = sorted(job_kw & cv_kw)
    gaps = sorted(job_kw - cv_kw)

    return job_kw, overlap, gaps


def generate_keyword_only_analysis_text(
    job: Dict,
    resume_text: str,
    keyword_overlap: List[str],
    keyword_gaps: List[str],
    job_kw_total: int,
    keyword_coverage: int,
) -> str:
    """Generate a narrative analysis that focuses purely on
    keyword overlap between the job description and the resume.

    This avoids referencing skills or requirement fields which may
    not be present in the API response.
    """

    title = job.get("Title", "this role")
    core_strengths = keyword_overlap[:5]
    core_gaps = keyword_gaps[:5]

    strengths_str = ", ".join(core_strengths) if core_strengths else "related experience"
    gaps_str = ", ".join(core_gaps) if core_gaps else "no obvious gaps based on text"

    lines: List[str] = []

    # Overview
    lines.append(
        f"For the **{title}** role, your resume covers about **{keyword_coverage}%** "
        f"of the prominent keywords found in the job description."
    )

    # Strengths
    lines.append("")
    lines.append("**Where you are aligned**")
    if core_strengths:
        lines.append(f"- Strong overlap on: **{strengths_str}**.")
    else:
        lines.append("- Limited direct keyword overlap detected. Consider mirroring key terms from the JD where truthful.")

    # Gaps
    lines.append("")
    lines.append("**Potential gaps**")
    if core_gaps:
        lines.append(f"- Missing or under-emphasised: **{gaps_str}**.")
    else:
        lines.append("- No clear gaps from keywords alone. Focus on clearer impact statements and outcomes.")

    # Tips
    lines.append("")
    lines.append("**How to strengthen your fit**")
    if core_gaps:
        lines.append(
            "- If you have experience in the above, bring them forward explicitly with quantifiable examples."
        )
    lines.append(
        "- Mirror relevant phrasing from the JD (truthfully) so that ATS and reviewers can recognise the match."
    )

    return "\n".join(lines)


def recommend_course(job: Dict, gaps: List[str]) -> str:
    """Return one simple Singapore course recommendation based on title/skills."""
    title = job.get("Title", "").lower()
    skills_val = job.get("id_Job_Skills", []) or []
    if isinstance(skills_val, list):
        skills_text = " ".join(str(s) for s in skills_val)
    else:
        skills_text = str(skills_val)

    req_text = get_job_requirement_text(job)
    text = f"{title} {skills_text} {req_text} {' '.join(gaps)}".lower()

    if any(k in text for k in ["support", "helpdesk", "customer service", "call centre"]):
        return "‘Customer Service Excellence’ — NTUC LearningHub (Singapore, classroom/online)"
    if any(k in text for k in ["data", "analytics", "excel"]):
        return "‘Excel Skills for Business’ — Coursera (online, SkillsFuture claimable)"
    if any(k in text for k in ["admin", "executive", "coordinator"]):
        return "‘Digital Office Skills with Microsoft 365’ — Singapore Polytechnic PACE (short course)"
    if any(k in text for k in ["it", "network", "technician"]):
        return "‘CompTIA A+ Certification Training’ — NTUC LearningHub (Singapore, blended)"
    if any(k in text for k in ["sales", "marketing", "account manager"]):
        return "‘Professional Selling Skills’ — SMU Academy (short executive programme)"

    return "‘Career Resilience & Future Skills’ — SkillsFuture Singapore (online options available)"


def extract_resume_text(uploaded_file) -> str:
    """Extract text from PDF, DOCX, or TXT resume."""
    if uploaded_file is None:
        return ""

    suffix = uploaded_file.name.lower().split(".")[-1]

    if suffix == "txt":
        return uploaded_file.read().decode("utf-8", errors="ignore").strip()

    if suffix in ("docx", "doc"):
        try:
            from docx import Document  # pip install python-docx

            doc = Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs).strip()
        except Exception as e:
            st.error(f"Error reading DOC/DOCX file: {e}")
            return ""

    if suffix == "pdf":
        try:
            from PyPDF2 import PdfReader  # pip install pypdf2

            reader = PdfReader(uploaded_file)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages).strip()
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return ""

    st.warning("Unsupported file type. Please upload PDF, DOCX, DOC, or TXT.")
    return ""


# ---------------- SIDEBAR – JOB SEARCH ----------------
st.sidebar.markdown("### Job Search")

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

    with st.expander("🔍 Debug: Raw backend response"):
        st.json(raw)

    wrapped = raw.get("data", {}).get("result", []) if raw else []

    # Optional debug to inspect requirement-like fields in first job
    if wrapped:
        first_job_full = wrapped[0]               # the whole wrapper (item)
        first_job = first_job_full.get("job", {}) # the “job” dictionary inside

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
        st.session_state["analysis_text"] = ""
        st.session_state["job_match_pct"] = None
        st.session_state["keyword_coverage_pct"] = None

        st.success(f"Fetched {len(full_jobs)} jobs (displaying {len(df_all)} after filters).")


# ---------------- GLOBAL STYLES ----------------
st.markdown(
    """
    <style>
    .big-title {font-size: 40px; font-weight: 700;}
    .job-card {padding: 1rem 1.2rem; border-radius: 0.8rem;
               border: 1px solid #eee; margin-bottom: 0.8rem;
               background-color: #fafafa;}
    .job-title {font-size: 20px; font-weight: 600; margin-bottom: 0.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
