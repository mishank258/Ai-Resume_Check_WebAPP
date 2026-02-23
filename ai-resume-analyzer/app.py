import io
import re
import pdfplumber
from pypdf import PdfReader
from docx import Document
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from jd_parser import parse_job_description

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB limit

# ─────────────────────────────────────────────
# SKILL DICTIONARIES
# ─────────────────────────────────────────────

TECH_SKILLS = {
    "python", "javascript", "typescript", "java", "csharp", "golang",
    "c", "cpp", "ruby", "swift", "kotlin", "rust", "php", "scala",
    "react", "vue", "angular", "html", "css", "nextjs", "svelte",
    "node", "flask", "django", "fastapi", "spring", "express", "rails",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "linux",
    "microservices", "serverless", "containerization",
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "tensorflow", "pytorch", "nlp", "ai", "llm", "ml",
    "rest", "graphql", "grpc", "git", "ci", "cd", "jenkins",
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
    "graduate", "honours", "university", "swinburne", "technology",
    "science", "diploma",
}

ROLE_KEYWORDS = {
    "graduate", "junior", "senior", "lead", "principal", "intern",
    "software", "student", "engineer", "developer", "architect",
    "devops", "fullstack", "backend", "frontend", "mobile", "embedded", "cloud",
}

SKILL_ADJACENCY = {
    "containerization": {"docker", "kubernetes", "aws", "gcp", "azure", "terraform", "linux"},
    "distributed":      {"aws", "gcp", "azure", "microservices", "kafka", "rabbitmq", "node", "rest", "grpc"},
    "microservices":    {"docker", "kubernetes", "aws", "rest", "node", "flask", "fastapi", "grpc"},
    "kubernetes":       {"docker", "containerization", "aws", "gcp", "azure"},
    "terraform":        {"aws", "gcp", "azure", "kubernetes"},
    "mlops":            {"python", "docker", "kubernetes", "tensorflow", "pytorch"},
    "golang":           {"cpp", "rust", "java", "python"},
    "grpc":             {"rest", "graphql", "node", "python"},
    "kafka":            {"rabbitmq", "distributed", "node", "python", "java"},
}

SYNONYMS = {
    "problem solving":        "problemsolving",
    "self-motivated":         "selfmotivated",
    "team player":            "teamwork",
    "computer science":       "computerscience",
    "software engineering":   "softwareengineering",
    "information technology": "informationtechnology",
    "dev":       "developer",
    "js":        "javascript",
    "nodejs":    "node",
    "node.js":   "node",
    "vue.js":    "vue",
    "react.js":  "react",
    "postgres":  "postgresql",
    "restful":   "rest",
    "rest api":  "rest",
    "rest apis": "rest",
    "c#":        "csharp",
    "c++":       "cpp",
    "cicd":      "ci",
    "ci/cd":     "ci",
    "ts":        "typescript",
    "k8s":       "kubernetes",
    "go":        "golang",
    "ml":        "ai",
    "agile":     "teamwork",
    "kanban":    "teamwork",
    "scrum":     "teamwork",
}

W_TECH_SKILL = 0.50
W_TFIDF      = 0.20
W_SOFT_SKILL = 0.15
W_ROLE_LEVEL = 0.10
W_EDUCATION  = 0.05


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    for src, tgt in sorted(SYNONYMS.items(), key=lambda x: -len(x[0])):
        text = text.replace(src, tgt)
    text = re.sub(r'ci\s*/\s*cd', 'ci', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> set:
    return {w for w in text.split() if len(w) > 1}


def extract_experience(text: str) -> int:
    matches = re.findall(r'(\d+)\s*\+?\s*years?', text)
    return max((int(m) for m in matches), default=0)


def tfidf_similarity(text_a: str, text_b: str) -> float:
    try:
        vecs = TfidfVectorizer(min_df=1, stop_words='english').fit_transform([text_a, text_b])
        return float(cosine_similarity(vecs[0], vecs[1])[0][0])
    except Exception:
        return 0.0


def smart_skill_overlap(resume_tokens: set, job_tokens: set, skill_set: set):
    required = job_tokens & skill_set
    if not required:
        return 1.0, set(), set(), set()
    exact_match = set()
    adjacent_match = set()
    truly_missing = set()
    for skill in required:
        if skill in resume_tokens:
            exact_match.add(skill)
        elif skill in SKILL_ADJACENCY and (SKILL_ADJACENCY[skill] & resume_tokens):
            adjacent_match.add(skill)
        else:
            truly_missing.add(skill)
    total_points = len(exact_match) * 1.0 + len(adjacent_match) * 0.5
    return total_points / len(required), exact_match, adjacent_match, truly_missing


def soft_skill_overlap(resume_tokens: set, job_tokens: set):
    required = job_tokens & SOFT_SKILLS
    if not required:
        return 1.0, set(), set()
    PROXIES = {
        "communication": {"communicated", "communication", "customer", "service", "support", "resolved"},
        "teamwork":      {"collaborated", "collaboration", "teamwork", "team", "agile", "kanban", "scrum", "sprint"},
        "leadership":    {"led", "lead", "president", "coordinator", "managed", "organised"},
        "creativity":    {"designed", "built", "developed", "created", "innovative"},
        "curiosity":     {"learning", "explored", "researched", "projects", "personal"},
        "mentorship":    {"mentored", "taught", "trained", "supported", "guided"},
        "problemsolving":{"resolved", "fixed", "debugged", "optimised", "improved", "solutions"},
        "adaptable":     {"adapted", "versatile", "multiple", "cross"},
        "analytical":    {"analysed", "analysis", "data", "evaluated", "measured"},
    }
    matched = set()
    missing = set()
    for skill in required:
        proxies = PROXIES.get(skill, {skill})
        if skill in resume_tokens or (proxies & resume_tokens):
            matched.add(skill)
        else:
            missing.add(skill)
    return len(matched) / len(required), matched, missing


# ─────────────────────────────────────────────
# RESUME FILE EXTRACTION
# ─────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text.strip()
    except Exception:
        pass
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        raise ValueError(f"Could not extract text from PDF: {e}")
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        paragraphs.append(cell.text.strip())
        return "\n".join(paragraphs)
    except Exception as e:
        raise ValueError(f"Could not extract text from DOCX: {e}")


# ─────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────

def analyze_resume(resume: str, job_desc: str):
    resume_clean  = clean_text(resume)
    job_clean     = clean_text(job_desc)
    resume_tokens = tokenize(resume_clean)
    job_tokens    = tokenize(job_clean)

    tech_score, exact_tech, adjacent_tech, missing_tech = smart_skill_overlap(
        resume_tokens, job_tokens, TECH_SKILLS
    )
    tfidf_score = tfidf_similarity(resume_clean, job_clean)
    soft_score, matched_soft, missing_soft = soft_skill_overlap(resume_tokens, job_tokens)

    role_required = job_tokens & ROLE_KEYWORDS
    if not role_required:
        role_score = 1.0
    else:
        role_matched = resume_tokens & role_required
        for rk in role_required:
            if rk in resume.lower():
                role_matched.add(rk)
        role_score = min(1.0, len(role_matched) / len(role_required))

    edu_required = job_tokens & EDU_KEYWORDS
    if not edu_required:
        edu_score = 1.0
    else:
        edu_matched = resume_tokens & edu_required
        for ek in edu_required:
            if ek in resume.lower():
                edu_matched.add(ek)
        edu_score = min(1.0, len(edu_matched) / len(edu_required))

    required_exp = extract_experience(job_clean)
    user_exp     = extract_experience(resume_clean)
    exp_penalty  = 0.0
    if required_exp > 0 and user_exp < required_exp:
        exp_penalty = min(0.20, (required_exp - user_exp) * 0.04)

    raw = (
        W_TECH_SKILL * tech_score
        + W_TFIDF    * tfidf_score
        + W_SOFT_SKILL * soft_score
        + W_ROLE_LEVEL * role_score
        + W_EDUCATION  * edu_score
    )
    final_score = round(max(0.0, min(100.0, raw * (1 - exp_penalty) * 100)), 1)

    if final_score >= 75:
        match_label = "Strong Match"
    elif final_score >= 55:
        match_label = "Good Match"
    elif final_score >= 38:
        match_label = "Partial Match"
    else:
        match_label = "Weak Match"

    suggestions = []
    truly_missing_sorted = sorted(missing_tech)

    label_messages = {
        "Strong Match":  "Strong match! Your profile aligns very well with this role.",
        "Good Match":    "Good match — a few additions could make you a top candidate.",
        "Partial Match": "Partial match — you have related experience; bridge the gaps below to stand out.",
        "Weak Match":    "This role may not match your current profile closely. Focus on the gaps below.",
    }
    suggestions.append(label_messages[match_label])

    if adjacent_tech:
        suggestions.append(
            f"You have related skills that partially cover: {', '.join(sorted(adjacent_tech))}. "
            "Adding explicit experience with these will strengthen your application."
        )
    if truly_missing_sorted:
        suggestions.append(
            f"Skills with no coverage in your resume: {', '.join(truly_missing_sorted[:5])}. "
            "Consider adding projects or certifications."
        )
    if required_exp > 0 and user_exp < required_exp:
        gap = required_exp - user_exp
        suggestions.append(
            f"Role prefers {required_exp}+ years; you have ~{user_exp}. "
            f"Highlight projects to bridge the {gap}-year gap."
        )
    if {"aws", "azure", "gcp"} & missing_tech:
        suggestions.append(
            f"Cloud skills ({', '.join(sorted({'aws','azure','gcp'} & missing_tech)).upper()}) are required — "
            "a free-tier deployment project would demonstrate these."
        )
    if missing_soft:
        suggestions.append(
            f"Make soft skills more explicit in your resume: {', '.join(sorted(missing_soft))}."
        )

    breakdown = {
        "tech_skill_score":   round(tech_score  * 100, 1),
        "tfidf_similarity":   round(tfidf_score  * 100, 1),
        "soft_skill_score":   round(soft_score   * 100, 1),
        "role_level_score":   round(role_score   * 100, 1),
        "education_score":    round(edu_score    * 100, 1),
        "experience_penalty": round(exp_penalty  * 100, 1),
    }

    return final_score, match_label, truly_missing_sorted[:10], suggestions, breakdown


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided. Use field name 'file'."}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({"error": "Empty filename."}), 400
        filename   = file.filename.lower()
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": "Uploaded file is empty."}), 400
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_bytes)
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file_bytes)
        elif filename.endswith('.doc'):
            return jsonify({"error": ".doc not supported — please save as .docx and re-upload."}), 400
        else:
            return jsonify({"error": "Unsupported file type. Upload a PDF or DOCX."}), 400
        if not text or len(text.strip()) < 50:
            return jsonify({"error": "Could not extract enough text. Try copy-pasting instead."}), 400
        return jsonify({
            "text":       text.strip(),
            "char_count": len(text.strip()),
            "filename":   file.filename,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@app.route('/parse_jd', methods=['POST'])
def parse_jd():
    try:
        data = request.get_json(silent=True) or {}
        raw  = data.get('raw_jd', '').strip()
        if not raw:
            return jsonify({"error": "raw_jd field is required"}), 400
        if len(raw) < 30:
            return jsonify({"error": "Job posting text is too short to parse"}), 400
        return jsonify(parse_job_description(raw))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400
        resume   = data.get('resume',   '').strip()
        job_desc = data.get('job_desc', '').strip()
        if not resume or not job_desc:
            return jsonify({"error": "Both resume and job description are required"}), 400
        if len(resume) < 50:
            return jsonify({"error": "Resume text is too short to analyse meaningfully"}), 400
        if len(job_desc) < 30:
            return jsonify({"error": "Job description is too short to analyse meaningfully"}), 400
        score, match_label, missing_keywords, suggestions, breakdown = analyze_resume(resume, job_desc)
        return jsonify({
            "score":            score,
            "match_label":      match_label,
            "missing_keywords": missing_keywords,
            "suggestions":      suggestions,
            "breakdown":        breakdown,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# RUN  ← always last
# ─────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True)