"""Smart Gap Analysis using LangChain + Gemini API with Web Search."""
import os
from typing import Dict, List, Tuple, Optional
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_smart_gap_analysis(
    job: Dict,
    resume_text: str,
    keyword_overlap: List[str],
    keyword_gaps: List[str],
    gemini_api_key: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    use_web_search: bool = True,
) -> Tuple[str, str]:
    """
    Perform smart gap analysis using LangChain + Gemini API with web search.
    
    Args:
        job: Job dictionary containing job details
        resume_text: Extracted text from user's resume
        keyword_overlap: List of keywords that overlap between job and resume
        keyword_gaps: List of keywords missing from resume
        gemini_api_key: Google Gemini API key (optional, will use env var if not provided)
        tavily_api_key: Tavily API key for web search (optional, will use env var if not provided)
        use_web_search: Whether to use web search for finding courses
        
    Returns:
        Tuple of (analysis_text, course_recommendations)
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_tavily import TavilySearch
    except ImportError as e:
        return (
            f"❌ Missing required packages. Please install: {str(e)}\n\n"
            "Run: pip install langchain-google-genai langchain-community tavily-python python-dotenv",
            "Package installation required"
        )
    
    # Get API keys
    gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
    
    if not gemini_key:
        return (
            "❌ Gemini API key not found. Please provide it in the sidebar or set GEMINI_API_KEY in .env file.",
            "API key required"
        )
    
    # Initialize Gemini LLM
    # Use a larger token limit to avoid truncation of longer analyses
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_key,
        temperature=0.3,
        max_tokens=4096,
    )
    
    # Extract job details
    job_title = job.get("Title", "this position")
    company = job.get("Company", "the company")
    job_description = job.get("JobDescription", "")
    if isinstance(job_description, dict):
        job_description = job_description.get("caption", "") or job_description.get("value", "")
    
    # Build comprehensive job context
    job_context = f"""
    Job Title: {job_title}
    Company: {company}
    Job Description: {job_description}
    """
    
    # === PART 1: GAP ANALYSIS ===
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert career counselor and recruitment specialist with deep knowledge of the Singapore job market.
        
        Your task is to provide a comprehensive, actionable skill gap analysis comparing a candidate's resume against a specific job posting.
        
        Guidelines:
        - Be honest but encouraging
        - Focus on practical, actionable insights
        - Highlight transferable skills
        - Suggest specific ways to bridge gaps
        - Consider Singapore's employment context
        - Be concise but thorough
        """),
        ("human", """Analyze the following job and resume:

JOB DETAILS:
{job_context}

RESUME TEXT:
{resume_text}

KEYWORD OVERLAP ({overlap_count} keywords):
{keyword_overlap}

MISSING KEYWORDS ({gaps_count} keywords):
{keyword_gaps}

Provide a structured analysis with these sections:

1. **MATCH STRENGTH** (overall assessment in 2-3 sentences)

2. **KEY STRENGTHS** (3-5 specific points where the candidate excels or matches well)

3. **SKILL GAPS** (3-5 areas that need development or are missing)

            max_tokens=8192,

Be specific, professional, and Singapore-focused. Use markdown formatting.
""")
    ])
    
    analysis_chain = analysis_prompt | llm | StrOutputParser()
    
    try:
        gap_analysis = analysis_chain.invoke({
            "job_context": job_context,
            "resume_text": resume_text[:6000],  # Limit to avoid token overflow
            "keyword_overlap": ", ".join(keyword_overlap[:20]),
            "overlap_count": len(keyword_overlap),
            "keyword_gaps": ", ".join(keyword_gaps[:20]),
            "gaps_count": len(keyword_gaps),
        })
    except Exception as e:
        gap_analysis = f"Error generating analysis: {str(e)}"
    
    # === PART 2: COURSE RECOMMENDATIONS WITH WEB SEARCH ===
    course_recommendations = ""
    
    if use_web_search and tavily_key:
        try:
            # Initialize web search tool
            search = TavilySearch(
                api_key=tavily_key,
                max_results=3,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
            )
            
            # Create search query
            top_gaps = keyword_gaps[:5]
            search_query = f"Singapore professional courses training for {job_title} {' '.join(top_gaps)} SkillsFuture"
            
            # Perform search
            search_results = search.invoke(search_query)
            
            # Process search results with LLM
            course_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert in Singapore's professional development and training landscape.
                
                Based on web search results, recommend 3-4 relevant courses that:
                - Are available in Singapore
                - Address the identified skill gaps
                - Are ideally SkillsFuture claimable
                - Come from reputable institutions
                - Include both online and classroom options
                """),
                ("human", """Based on these skill gaps for a {job_title} position:
{skill_gaps}

And these web search results about Singapore courses:
{search_results}

Provide 3-4 course recommendations. For each course include:

Format as a numbered list with clear structure.
""")
            ])
            course_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=gemini_key,
                temperature=0.3,
                max_tokens=1500,
            )
            course_chain = course_prompt | course_llm | StrOutputParser()
            
            course_recommendations = course_chain.invoke({
                "job_title": job_title,
                "skill_gaps": ", ".join(keyword_gaps[:10]),
                "search_results": str(search_results),
            })
            # If AI returned less-than-expected output, fall back to LLM-only recommendations
            if not (course_recommendations and course_recommendations.strip() and len(course_recommendations.strip()) > 30):
                course_recommendations = get_llm_course_recommendations(llm, job_title, keyword_gaps)
            
        except Exception as e:
            # Fallback to LLM-only recommendations
            course_recommendations = f"Web search unavailable. Using AI recommendations:\n\n"
            course_recommendations += get_llm_course_recommendations(
                llm, job_title, keyword_gaps
            )
    else:
        # Use LLM-only recommendations without web search
        course_recommendations = get_llm_course_recommendations(
            llm, job_title, keyword_gaps
        )
        # If the LLM didn't provide enough detail, use a heuristic fallback
        if not (course_recommendations and course_recommendations.strip() and len(course_recommendations.strip()) > 60):
            course_recommendations = (
                "No web-based course recommendations found. Suggested fallback courses:\n\n"
                "- 'Digital Office Skills with Microsoft 365' — Singapore Polytechnic PACE\n"
                "- 'Excel Skills for Business' — Coursera (SkillsFuture claimable)\n"
                "- 'Career Resilience & Future Skills' — SkillsFuture Singapore"
            )
    
    # Deduplicate repeated paragraphs in both analysis and course recommendations
    def _dedupe_paragraphs(s: str) -> str:
        if not s:
            return s
        parts = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
        seen = set()
        uniq = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return "\n\n".join(uniq)

    gap_analysis = _dedupe_paragraphs(gap_analysis)
    course_recommendations = _dedupe_paragraphs(course_recommendations)

    return gap_analysis, course_recommendations


def get_llm_course_recommendations(llm, job_title: str, skill_gaps: List[str]) -> str:
    """Generate course recommendations using LLM knowledge only (no web search)."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert in Singapore's professional development landscape.
        Recommend relevant courses from well-known Singapore institutions like:
        - SkillsFuture Singapore
        - NTUC LearningHub
        - Singapore Polytechnic
        - Coursera (SkillsFuture eligible)
        - Udemy
        - LinkedIn Learning
        - SMU Academy
        - NUS/NTU continuing education
        """),
        ("human", """For a {job_title} position with these skill gaps:
{skill_gaps}

Recommend 3-4 relevant courses available in Singapore. For each include:

Format as a numbered list.
""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        return chain.invoke({
            "job_title": job_title,
            "skill_gaps": ", ".join(skill_gaps[:10]),
        })
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"


def get_quick_insights(
    resume_text: str,
    job_description: str,
    gemini_api_key: Optional[str] = None,
) -> str:
    """
    Get quick insights about resume-job fit using Gemini.
    
    Args:
        resume_text: User's resume text
        job_description: Job description text
        gemini_api_key: Google Gemini API key
        
    Returns:
        Quick insights text
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except ImportError:
        return "❌ Missing required packages."
    
    gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return "❌ Gemini API key required."
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_key,
        temperature=0.3,
        max_tokens=500,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful career advisor. Provide quick, actionable insights."),
        ("human", """Compare this resume with the job description and give 3 quick insights (each 1 sentence):

Resume: {resume}
Job: {job}

Format:
✓ [Strength]
⚠ [Gap]
💡 [Quick tip]
""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        return chain.invoke({
            "resume": resume_text[:1500],
            "job": job_description[:1500],
        })
    except Exception as e:
        return f"Error: {str(e)}"
