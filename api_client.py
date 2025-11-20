# api_client.py
import requests
from typing import Iterable, Optional

BASE_URL = "https://www.findsgjobs.com/apis/job/searchable"


def _join(values: Optional[Iterable[int]]) -> Optional[str]:
    if not values:
        return None
    return ",".join(str(v) for v in values)


def fetch_jobs_from_endpoint(
    page: int = 1,
    per_page: int = 20,
    keywords: str = "",
    employment_types: Optional[Iterable[int]] = None,
    job_categories: Optional[Iterable[int]] = None,
    min_education_levels: Optional[Iterable[int]] = None,
    min_years_of_experience: Optional[Iterable[int]] = None,
    mrt_stations: Optional[Iterable[int]] = None,
    position: Optional[str] = None,
    currency: int = 1275916990,
    min_salary: Optional[int] = None,
    max_salary: Optional[int] = None,
    interval: int = 1898,
    sort_field: str = "activation_date",
    sort_direction: str = "desc",
):
    params = {
        "page": page,
        "per_page_count": per_page,     # ✅ Correct key confirmed
        "keywords": keywords or None,
        "EmploymentType": _join(employment_types),
        "JobCategory": _join(job_categories),
        "MinimumEducationLevel": _join(min_education_levels),
        "MinimumYearsofExperience": _join(min_years_of_experience),
        "id_Job_NearestMRTStation": _join(mrt_stations),
        "Position": position,
        "id_Job_Currency": currency,
        "id_Job_Salary": min_salary,
        "id_Job_MaxSalary": max_salary,
        "id_Job_Interval": interval,
        "sort_field": sort_field,
        "sort_direction": sort_direction,
    }

    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}

    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()
