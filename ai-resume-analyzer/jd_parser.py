"""
jd_parser.py — Universal AI-powered job description parser
Works for ANY field: tech, nursing, pharmacy, business, law, education, etc.
Uses Gemini API if key is set, falls back to keyword matching automatically.
"""

import os
import re
import json

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSAL FALLBACK DICTIONARIES
# ─────────────────────────────────────────────────────────────────────────────

_HARD_SKILLS = {
    # Technology
    "python","javascript","typescript","java","csharp","golang","cpp","c",
    "ruby","swift","kotlin","rust","php","scala","react","vue","angular",
    "html","css","nextjs","svelte","node","flask","django","fastapi","spring",
    "aws","azure","gcp","docker","kubernetes","terraform","linux",
    "microservices","serverless","containerization","sql","mysql","postgresql",
    "mongodb","redis","elasticsearch","tensorflow","pytorch","nlp","ai","llm",
    "rest","graphql","grpc","git","ci","cd","jenkins","kafka","distributed",
    # Healthcare / Nursing
    "patient care","clinical assessment","medication administration","wound care",
    "icu","aged care","triage","ahpra","iv therapy","infection control",
    "emergency care","palliative care","mental health","paediatrics",
    "obstetrics","community nursing","clinical documentation","manual handling",
    # Pharmacy
    "dispensing","compounding","pharmacovigilance","drug interactions",
    "schedule 8","sterile manufacturing","clinical trials","regulatory affairs",
    "quality assurance","gmp","tga","medication management","dose administration",
    # Medicine
    "diagnosis","clinical reasoning","surgery","general practice",
    "medical imaging","pathology","pharmacology","prescribing","consultation",
    # Business / Finance
    "financial analysis","financial modelling","excel","power bi","tableau",
    "stakeholder management","project management","crm","salesforce","marketing",
    "kpi","budgeting","forecasting","accounting","auditing","risk management",
    "business development","supply chain","operations management","xero","myob",
    # Law
    "legal research","contract drafting","litigation","compliance",
    "corporate law","conveyancing","family law","criminal law","mediation",
    "legal writing","due diligence","intellectual property",
    # Data / Analytics
    "r","pandas","numpy","jupyter","statistics","data visualisation",
    "machine learning","deep learning","etl","spark","data engineering",
    # Engineering (non-software)
    "autocad","solidworks","matlab","circuit design","embedded systems",
    "mechanical design","structural analysis","cad","revit","catia",
    # HR / People
    "talent acquisition","recruitment","onboarding","performance management",
    "hris","payroll","employee relations","workforce planning","learning development",
    # Marketing / Design
    "seo","sem","google analytics","figma","photoshop","illustrator",
    "content marketing","social media","email marketing","brand management",
    "adobe creative suite","wordpress","hubspot","copywriting",
    # Hospitality / Retail / Customer Service
    "customer service","point of sale","inventory management","food safety",
    "barista","food handling","cash handling","merchandising","retail",
    # Trades / Construction
    "white card","forklift","welding","plumbing","electrical","carpentry",
    "tiling","concreting","heavy machinery","blueprint reading",
}

_SOFT_SKILLS = {
    "communication","teamwork","collaboration","mentorship","leadership",
    "creativity","curiosity","problem solving","adaptable","analytical",
    "time management","critical thinking","empathy","attention to detail",
    "organisation","initiative","resilience","multitasking","interpersonal",
    "self motivated","work ethic","integrity","accountability",
    "conflict resolution","negotiation","presentation","decision making",
    "cultural awareness","emotional intelligence",
}

_EDU_KEYWORDS = {
    "bachelor","master","phd","degree","diploma","certificate","honours",
    "graduate","postgraduate","undergraduate","university","tafe","college",
    "registered nurse","rn","md","mbbs","llb","mba","cpa","ca",
    "engineering","science","nursing","medicine","law","education","business",
    "computer science","software engineering","information technology",
    "pharmacy","psychology","social work","accounting","finance","marketing",
}

_ROLE_KEYWORDS = {
    "graduate","junior","senior","lead","principal","intern","manager",
    "director","coordinator","specialist","analyst","consultant","engineer",
    "developer","nurse","pharmacist","doctor","physician","associate",
    "officer","administrator","advisor","executive","assistant","trainee",
    "apprentice","supervisor","team leader","head of","vice president",
}

_CERT_KEYWORDS = {
    "ahpra","cpa","ca","aws certified","azure certified","google certified",
    "pmp","prince2","cissp","comptia","ccna","first aid","cpr",
    "working with children","police check","white card","forklift licence",
    "food handling certificate","tga","iso","rsa","responsible service",
    "cert iii","cert iv","certificate iii","certificate iv",
}

_FIELD_HINTS = {
    "software engineering": {"python","javascript","react","docker","aws","git","node","java","cpp","typescript"},
    "nursing":              {"patient care","icu","triage","ahpra","wound care","aged care","medication administration"},
    "pharmacy":             {"dispensing","compounding","gmp","tga","pharmacovigilance","schedule 8"},
    "medicine":             {"diagnosis","surgery","prescribing","pathology","mbbs","clinical reasoning"},
    "business":             {"financial analysis","excel","stakeholder management","kpi","crm","accounting","xero"},
    "data science":         {"pandas","numpy","r","statistics","machine learning","tableau","power bi","jupyter"},
    "law":                  {"litigation","contract drafting","legal research","compliance","corporate law"},
    "education":            {"curriculum development","lesson planning","classroom management","student assessment"},
    "engineering":          {"autocad","solidworks","matlab","circuit design","mechanical design","cad","revit"},
    "hr":                   {"talent acquisition","recruitment","onboarding","hris","payroll","employee relations"},
    "marketing":            {"seo","sem","google analytics","content marketing","social media","brand management"},
    "hospitality":          {"customer service","food safety","barista","food handling","point of sale","rsa"},
}

_SYNONYMS = {
    "problem solving":"problemsolving","self-motivated":"selfmotivated",
    "team player":"teamwork","c++":"cpp","c#":"csharp","node.js":"node",
    "vue.js":"vue","react.js":"react","ci/cd":"ci","k8s":"kubernetes",
    "rest api":"rest","rest apis":"rest","agile/scrum":"agile",
    "ms excel":"excel","microsoft excel":"excel","ms word":"word",
    "javascript":"javascript","node js":"node","version control":"git",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = text.lower()
    for src, tgt in sorted(_SYNONYMS.items(), key=lambda x: -len(x[0])):
        text = text.replace(src, tgt)
    text = re.sub(r'ci\s*/\s*cd', 'ci', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _tokenize(text: str) -> set:
    """Single + bi + trigram tokenization for multi-word skill matching."""
    tokens = set()
    words = text.split()
    for w in words:
        if len(w) > 1:
            tokens.add(w)
    for i in range(len(words) - 1):
        tokens.add(f"{words[i]} {words[i+1]}")
    for i in range(len(words) - 2):
        tokens.add(f"{words[i]} {words[i+1]} {words[i+2]}")
    return tokens


def _extract_experience(text: str):
    """Only extract experience years from requirement-related lines, not company history."""
    lines = text.lower().split("\n")
    REQUIREMENT_SIGNALS = {
        "require", "minimum", "at least", "must have", "you have",
        "years experience", "years of experience", "proven experience",
        "ideally", "preferred", "looking for", "we need", "seeking",
    }
    results = []
    for line in lines:
        if any(skip in line for skip in [
            "founded", "over 30 years", "30+ years", "25 years", "20 years",
            "company", "our history", "we have been", "in business",
            "clients", "careers launched", "global", "ftse", "centres",
        ]):
            continue
        has_signal = any(signal in line for signal in REQUIREMENT_SIGNALS)
        matches = re.findall(r"(\d+)\s*\+?\s*years?", line)
        if matches and has_signal:
            results.extend(int(m) for m in matches)
    return max(results, default=None) if results else None


def _detect_field(skills: list) -> str:
    skill_set = set(skills)
    scores = {f: len(skill_set & hints) for f, hints in _FIELD_HINTS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def _build_cleaned_text(field, hard_skills, soft_skills, edu, roles, experience, certs):
    parts = []
    if field and field != "general":
        parts.append(f"Field: {field}")
    if roles:
        parts.append("Role: " + ", ".join(roles))
    if hard_skills:
        parts.append("Required skills: " + ", ".join(hard_skills))
    if soft_skills:
        parts.append("Soft skills: " + ", ".join(soft_skills))
    if edu:
        parts.append("Education: " + ", ".join(edu))
    if certs:
        parts.append("Certifications: " + ", ".join(certs))
    if experience is not None:
        parts.append(f"Experience: {experience}+ years")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI AI EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _parse_with_gemini(raw_text: str) -> dict:
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    prompt = f"""You are an expert job description parser for ALL industries worldwide.

Analyse the job description and return ONLY a valid JSON object.
No markdown, no explanation, no code fences — pure JSON only.

{{
  "field": "one of: software engineering, nursing, pharmacy, business, medicine, data science, law, education, engineering, hr, marketing, hospitality, trades, general",
  "hard_skills": ["every technical/domain skill, tool, software, procedure, technology mentioned — be thorough"],
  "soft_skills": ["every interpersonal, behavioural, transferable skill mentioned"],
  "education": ["every degree, qualification, license requirement — e.g. bachelor of nursing, mbbs, cpa, cert iv"],
  "roles": ["seniority and role type — e.g. graduate, senior engineer, registered nurse, financial analyst"],
  "experience_years": null or integer (highest number of years explicitly stated),
  "certifications": ["every license, registration, certification, check required — e.g. AHPRA, working with children, white card, first aid"]
}}

Important rules:
- Extract EVERY skill mentioned, even indirectly or casually
- Normalise all text to lowercase
- Include implied skills if strongly suggested by the role
- Use [] for empty categories, null for missing experience_years
- Do NOT make up skills not in the text

Job description to parse:
{raw_text}"""

    response  = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    raw_json  = response.text.strip()
    raw_json  = re.sub(r"^```(?:json)?\s*", "", raw_json, flags=re.MULTILINE)
    raw_json  = re.sub(r"\s*```\s*$",       "", raw_json, flags=re.MULTILINE)
    return json.loads(raw_json.strip())


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _parse_with_keywords(raw_text: str) -> dict:
    cleaned = _clean(raw_text)
    tokens  = _tokenize(cleaned)
    hard_skills = sorted(tokens & _HARD_SKILLS)
    soft_skills = sorted(tokens & _SOFT_SKILLS)
    edu         = sorted(tokens & _EDU_KEYWORDS)
    roles       = sorted(tokens & _ROLE_KEYWORDS)
    certs       = sorted(tokens & _CERT_KEYWORDS)
    experience  = _extract_experience(raw_text)
    field       = _detect_field(hard_skills)
    return {
        "field": field, "hard_skills": hard_skills, "soft_skills": soft_skills,
        "education": edu, "roles": roles, "experience_years": experience,
        "certifications": certs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def parse_job_description(raw_text: str) -> dict:
    """Parse any job description for any field. Returns structured dict."""
    if not raw_text or not raw_text.strip():
        return {
            "field": "unknown", "tech_skills": [], "soft_skills": [],
            "edu_keywords": [], "role_keywords": [], "certifications": [],
            "experience": None, "cleaned_text": "", "source": "none",
            "summary": "No input provided.",
        }

    parsed = None
    source = "keyword_fallback"

    try:
        parsed = _parse_with_gemini(raw_text)
        source = "gemini_api"
        print(f"[jd_parser] ✓ Gemini — field: {parsed.get('field','?')}")
    except EnvironmentError:
        print("[jd_parser] No GEMINI_API_KEY — keyword fallback")
    except Exception as e:
        print(f"[jd_parser] Gemini error ({e}) — keyword fallback")

    if parsed is None:
        parsed = _parse_with_keywords(raw_text)

    field       = (parsed.get("field") or "general").lower().strip()
    hard_skills = sorted(set(s.lower().strip() for s in parsed.get("hard_skills",   []) if s))
    soft_skills = sorted(set(s.lower().strip() for s in parsed.get("soft_skills",   []) if s))
    edu         = sorted(set(s.lower().strip() for s in parsed.get("education",     []) if s))
    roles       = sorted(set(s.lower().strip() for s in parsed.get("roles",         []) if s))
    certs       = sorted(set(s.lower().strip() for s in parsed.get("certifications",[]) if s))
    experience  = parsed.get("experience_years")

    if field == "general" and hard_skills:
        field = _detect_field(hard_skills)

    cleaned_text = _build_cleaned_text(field, hard_skills, soft_skills, edu, roles, experience, certs)
    summary = (
        f"[{source}] {field} | "
        f"{len(hard_skills)} skills, {len(soft_skills)} soft skills"
        + (f", {experience}+ yrs" if experience else "")
        + (f", certs: {', '.join(certs[:2])}" if certs else "")
    )

    return {
        "field":          field,
        "tech_skills":    hard_skills,
        "soft_skills":    soft_skills,
        "edu_keywords":   edu,
        "role_keywords":  roles,
        "certifications": certs,
        "experience":     experience,
        "cleaned_text":   cleaned_text,
        "summary":        summary,
        "source":         source,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    tests = {
        "Tech":     "Graduate Software Engineer. C++ required. Docker, AWS. Bachelor Computer Science.",
        "Nursing":  "Registered Nurse ICU. AHPRA required. Patient care, triage, medication. Good communication.",
        "Business": "Financial Analyst. Excel, Power BI, CPA preferred. 3+ years experience.",
        "Pharmacy": "Pharmacist. Dispensing, compounding, TGA. AHPRA. Attention to detail required.",
    }
    for label, jd in tests.items():
        print(f"\n{'='*40}  {label}")
        r = parse_job_description(jd)
        for k, v in r.items():
            print(f"  {k:>16}: {v}")