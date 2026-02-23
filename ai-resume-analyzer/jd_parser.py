"""
jd_parser.py
────────────
Detects and extracts structured requirements from a raw job description.
Exposes a single public function: parse_job_description(text) -> dict

Used by app.py via:
    from jd_parser import parse_job_description
"""

import re

# ── Dictionaries ─────────────────────────────────────────────────────────────

TECH_SKILLS = {
    # Languages
    "python", "javascript", "typescript", "java", "csharp", "golang",
    "c", "cpp", "ruby", "swift", "kotlin", "rust", "php", "scala",
    # Frontend
    "react", "vue", "angular", "html", "css", "nextjs", "svelte",
    # Backend / frameworks
    "node", "flask", "django", "fastapi", "spring", "express", "rails",
    # Cloud & infra
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "linux",
    "microservices", "serverless", "containerization",
    # Data & AI
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "tensorflow", "pytorch", "nlp", "ai", "llm", "ml",
    # APIs & tools
    "rest", "graphql", "grpc", "git", "ci", "cd", "jenkins",
    # Distributed systems
    "kafka", "rabbitmq", "distributed", "concurrency",
}

SOFT_SKILLS = {
    "communication", "teamwork", "collaboration", "mentorship",
    "problemsolving", "leadership", "creativity", "curiosity",
    "selfmotivated", "adaptable", "analytical",
}

EDU_KEYWORDS = {
    "bachelor", "master", "phd", "degree", "computerscience",
    "softwareengineering", "informationtechnology", "engineering",
    "graduate", "honours", "university",
}

ROLE_KEYWORDS = {
    "graduate", "junior", "senior", "lead", "principal", "intern",
    "engineer", "developer", "architect", "devops", "fullstack",
    "backend", "frontend", "mobile", "embedded", "cloud",
}

SYNONYMS = {
    # Keep multi-word first so they're replaced before single-word passes
    "problem solving":      "problemsolving",
    "self-motivated":       "selfmotivated",
    "team player":          "teamwork",
    "computer science":     "computerscience",
    "software engineering": "softwareengineering",
    "information technology": "informationtechnology",
    # Single-token
    "dev":      "developer",
    "js":       "javascript",
    "nodejs":   "node",
    "node.js":  "node",
    "postgres": "postgresql",
    "restful":  "rest",
    "c#":       "csharp",
    "c++":      "cpp",
    "cicd":     "ci",
    "ts":       "typescript",
    "ml":       "ai",
    "k8s":      "kubernetes",
    "go":       "golang",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Lowercase, apply synonyms, strip punctuation."""
    text = text.lower()
    # Multi-word synonyms first (longest → shortest to avoid partial matches)
    for src, tgt in sorted(SYNONYMS.items(), key=lambda x: -len(x[0])):
        text = text.replace(src, tgt)
    # Normalise ci/cd written with a slash
    text = re.sub(r'ci\s*/\s*cd', 'ci', text)
    # Strip everything except letters, digits, spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _tokenize(text: str) -> set:
    """Return a deduplicated set of tokens longer than 1 char."""
    return {w for w in text.split() if len(w) > 1}


def _extract_experience(raw_text: str) -> int | None:
    """Return the highest explicit years-of-experience figure, or None."""
    matches = re.findall(r'(\d+)\s*\+?\s*years?', raw_text, re.IGNORECASE)
    return max((int(m) for m in matches), default=None) if matches else None


def _match_tokens(tokens: set, keyword_set: set) -> list:
    """Return sorted list of tokens that belong to keyword_set."""
    return sorted(tokens & keyword_set)


# ── Public API ────────────────────────────────────────────────────────────────

def parse_job_description(raw_text: str) -> dict:
    """
    Extract structured requirements from a raw job posting string.

    Parameters
    ----------
    raw_text : str
        The full, unprocessed job description (marketing copy and all).

    Returns
    -------
    dict with keys:
        tech_skills  : list[str]  – detected technical skills / technologies
        soft_skills  : list[str]  – detected soft / interpersonal skills
        role_keywords: list[str]  – seniority / role type indicators
        edu_keywords : list[str]  – education-related terms
        experience   : int|None   – highest explicit years-of-exp figure
        cleaned_text : str        – normalised text (ready for TF-IDF / matching)
        summary      : str        – one-line human-readable extraction summary
    """
    if not raw_text or not raw_text.strip():
        return {
            "tech_skills":   [],
            "soft_skills":   [],
            "role_keywords": [],
            "edu_keywords":  [],
            "experience":    None,
            "cleaned_text":  "",
            "summary":       "No input provided.",
        }

    cleaned  = _clean(raw_text)
    tokens   = _tokenize(cleaned)

    tech    = _match_tokens(tokens, TECH_SKILLS)
    soft    = _match_tokens(tokens, SOFT_SKILLS)
    roles   = _match_tokens(tokens, ROLE_KEYWORDS)
    edu     = _match_tokens(tokens, EDU_KEYWORDS)
    exp     = _extract_experience(raw_text)

    # Build a compact cleaned description suitable for the job box
    parts = []
    if roles:
        parts.append("Role: " + ", ".join(roles))
    if tech:
        parts.append("Required skills: " + ", ".join(tech))
    if soft:
        parts.append("Soft skills: " + ", ".join(soft))
    if edu:
        parts.append("Education: " + ", ".join(edu))
    if exp is not None:
        parts.append(f"Experience: {exp}+ years")

    summary = (
        f"Found {len(tech)} tech skill(s), {len(soft)} soft skill(s), "
        f"{len(roles)} role keyword(s), {len(edu)} education keyword(s)"
        + (f", {exp}+ years experience required." if exp else ".")
    )

    return {
        "tech_skills":    tech,
        "soft_skills":    soft,
        "role_keywords":  roles,
        "edu_keywords":   edu,
        "experience":     exp,
        "cleaned_text":   "\n".join(parts),
        "summary":        summary,
    }


# ── Optional Flask route (register in app.py) ────────────────────────────────
# To wire this up, add to app.py:
#
#   from jd_parser import parse_job_description
#
#   @app.route('/parse_jd', methods=['POST'])
#   def parse_jd():
#       data = request.get_json(silent=True) or {}
#       raw  = data.get('raw_jd', '').strip()
#       if not raw:
#           return jsonify({"error": "raw_jd field is required"}), 400
#       return jsonify(parse_job_description(raw))
#
# Frontend call:
#   const res  = await fetch('/parse_jd', {
#       method: 'POST',
#       headers: {'Content-Type': 'application/json'},
#       body: JSON.stringify({ raw_jd: rawText })
#   });
#   const data = await res.json();
#   // data.tech_skills, data.cleaned_text, etc.


# ── CLI quick-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    Launch Your Future as a Graduate Software Engineer at Blackmagic Design.
    We are looking for a graduate or junior engineer with strong C and C++ skills.
    Additional knowledge of Golang highly regarded. Experience with Docker,
    Kubernetes, microservices, and cloud technologies (AWS/GCP) is a plus.
    Requires a Bachelor or Master degree in Computer Science or Software Engineering.
    Strong communication and teamwork skills essential. 2+ years experience preferred.
    """
    result = parse_job_description(sample)
    print("=== JD Parser Output ===")
    for k, v in result.items():
        print(f"{k:>15}: {v}")
