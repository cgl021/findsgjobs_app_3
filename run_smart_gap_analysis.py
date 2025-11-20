#!/usr/bin/env python3
"""
Standalone runner for the Smart Gap Analysis (smart_gap_analysis.get_smart_gap_analysis).

Usage:
    python run_smart_gap_analysis.py --resume path/to/resume.pdf --job path/to/job.json --out results.txt

If no --job is provided, a sample job is used.
If GEMINI_API_KEY / TAVILY_API_KEY are set in .env, the script will attempt to use them.

The script attempts to parse PDF/DOCX/TXT resumes and falls back to simple text input.
"""
import argparse
import json
import os
import re
from typing import List, Tuple

from dotenv import load_dotenv

# Import the analysis function
from smart_gap_analysis import get_smart_gap_analysis

# Minimal stopwords set to extract keywords (similar to the app)
STOPWORDS = {
    "and", "the", "with", "for", "to", "of", "in", "on", "a", "an", "or",
    "be", "as", "by", "is", "are", "will", "able", "etc", "any", "all",
    "job", "role", "responsible", "responsibilities", "requirement",
    "requirements", "candidate", "candidates", "ability", "strong", "good",
    "skills", "experience", "experiences", "year", "years"
}


def extract_keywords(text: str, min_len: int = 3) -> set:
    if not text:
        return set()
    tokens = re.findall(r"[A-Za-z]{%d,}" % min_len, text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def build_job_resume_overlap(job_text: str, resume_text: str) -> Tuple[List[str], List[str], List[str]]:
    job_kw = extract_keywords(job_text)
    cv_kw = extract_keywords(resume_text)
    overlap = sorted(job_kw & cv_kw)
    gaps = sorted(job_kw - cv_kw)
    return sorted(job_kw), overlap, gaps


def extract_resume_text(path: str) -> str:
    if not path:
        return ""
    suffix = path.lower().split('.')[-1]
    if suffix == 'txt':
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    if suffix in ('docx', 'doc'):
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            print(f"Warning: python-docx not available or failed to read file: {e}")
            return ""
    if suffix == 'pdf':
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(path)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        except Exception as e:
            print(f"Warning: PyPDF2 not available or failed to read PDF: {e}")
            return ""
    # fallback: read as text
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Unable to read resume file {path}: {e}")
        return ""


def read_job_file(job_file: str) -> dict:
    if not job_file:
        return {}
    with open(job_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description='Run Smart Gap Analysis against a resume and a job posting')
    parser.add_argument('--resume', type=str, help='Path to resume file (pdf/docx/txt)')
    parser.add_argument('--job', type=str, help='Path to job JSON file (optional)')
    parser.add_argument('--resume-text', type=str, help='Direct resume text (alternative to --resume)')
    parser.add_argument('--out', type=str, default=None, help='Optional output file to save analysis')
    parser.add_argument('--no-web-search', action='store_true', help='Disable web search for course recommendations')
    args = parser.parse_args()

    resume_text = ''
    if args.resume_text:
        resume_text = args.resume_text
    elif args.resume:
        resume_text = extract_resume_text(args.resume)

    if not resume_text:
        print('No resume text provided (use --resume or --resume-text)')
        return

    job = read_job_file(args.job) if args.job else {}
    if not job:
        # Minimal sample job if none provided
        job = {
            'Title': 'Planner (Junior Level, Aviation, Shift Work, Changi)',
            'Company': 'Changi Airline Services',
            'JobDescription': 'We are seeking a junior planner with experience in scheduling, basic aircraft maintenance awareness, and working in shift patterns at Changi Airport. Knowledge of inventory and spare parts is helpful. Comfortable in a high-paced environment.'
        }

    job_text = ''
    # Pick job description or fallback to title
    if isinstance(job.get('JobDescription'), str):
        job_text = job.get('JobDescription')
    elif isinstance(job.get('JobDescription'), dict):
        job_text = job['JobDescription'].get('caption') or job['JobDescription'].get('value') or ''
    if not job_text:
        job_text = job.get('Title', '')

    job_kw, overlap, gaps = build_job_resume_overlap(job_text, resume_text)

    print('\n==== MATCH OVERVIEW ====')
    print(f'- Job keywords (unique): {len(job_kw)}')
    print(f'- Keyword overlap: {len(overlap)}')
    print(f'- Coverage: {int(round(100 * len(overlap)/len(job_kw))) if job_kw else 0}%')

    use_web_search = not args.no_web_search

    # Attempt to run AI analysis; function also checks for missing dependencies and GEMINI_API_KEY
    try:
        analysis, courses = get_smart_gap_analysis(
            job=job,
            resume_text=resume_text,
            keyword_overlap=overlap,
            keyword_gaps=gaps,
            gemini_api_key=os.getenv('GEMINI_API_KEY'),
            tavily_api_key=os.getenv('TAVILY_API_KEY'),
            use_web_search=use_web_search,
        )
    except Exception as e:
        analysis = f'AI function raised an exception: {e}'
        courses = 'No recommendations'

    def dedupe_paragraphs(s: str) -> str:
        import re
        if not s:
            return s
        parts = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
        seen = set(); uniq = []
        for p in parts:
            if p not in seen:
                seen.add(p); uniq.append(p)
        return "\n\n".join(uniq)

    analysis = dedupe_paragraphs(analysis)
    courses = dedupe_paragraphs(courses)

    print('\n==== ANALYSIS ====', '\n')
    print(analysis)

    print('\n==== COURSE RECOMMENDATIONS ====', '\n')
    print(courses)

    if args.out:
        out_data = {
            'analysis': analysis,
            'courses': courses,
            'job_kw_count': len(job_kw),
            'overlap_count': len(overlap),
            'coverage_pct': int(round(100 * len(overlap) / len(job_kw))) if job_kw else 0
        }
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f'Output saved to {args.out}')


if __name__ == '__main__':
    main()
