# 🎯 ResumeIQ — AI Resume Analyzer

A smart web app that matches your resume against any job description and gives you an accurate skill-gap report — powered by Google Gemini AI.

Works for **any field**: software engineering, nursing, pharmacy, business, law, data science, education, HR, marketing, hospitality, trades, and more.

---

## ✨ Features

- **Upload or paste your resume** — PDF, DOCX, or plain text
- **AI-powered JD parser** — paste any raw job posting, Gemini extracts requirements automatically for any industry
- **Smart scoring** — skill adjacency matching, soft-skill proxy detection, semantic similarity, and TF-IDF
- **Score breakdown** — Tech Skills, Soft Skills, Role Fit, Education, Context
- **Actionable suggestions** — tells you what to bridge, not just what's missing

---

## 📁 Project Structure

```
ai-resume-analyzer/
│
├── app.py                  # Flask backend — all routes and scoring logic
├── jd_parser.py            # AI-powered job description parser (Gemini + keyword fallback)
├── templates/
│   └── index.html          # Frontend UI
├── requirements.txt        # Python dependencies
├── .env                    # Your API key (never commit this)
├── .gitignore              # Keeps secrets and venv off GitHub
└── README.md
```

---

## 🚀 Getting Started (Cloning from GitHub)

Follow these steps exactly after cloning the repo.

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/ai-resume-analyzer.git
cd ai-resume-analyzer
```

---

### Step 2 — Create a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear in your terminal prompt. This confirms you are in the right environment.

> ⚠️ Every time you open a new terminal to work on this project, you must run the activate command again before running the app.

---

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, scikit-learn, pdfplumber, python-docx, google-generativeai, and all other required packages.

---

### Step 4 — Get your own free Gemini API key

> **Why do you need your own key?**
> The original developer's API key is never pushed to GitHub for security reasons. Each person who runs the app locally needs their own key. It is completely free.

**How to get one:**

1. Go to **https://aistudio.google.com**
2. Sign in with your Google account
3. Click **Get API Key** in the left sidebar
4. Click **Create API key**
5. Copy the key — it looks like `AIzaSy...`

---

### Step 5 — Create your `.env` file

In the `ai-resume-analyzer` folder, create a new file called exactly `.env` (no other name, no extension).

Open it and paste this — replacing the placeholder with your actual key:

```
GEMINI_API_KEY=AIzaSy-your-actual-key-here
```

**Example of what it should look like:**
```
GEMINI_API_KEY=AIzaSyD3xAmPlEkEy123456789abcdef
```

> ⚠️ Do not add quotes around the key. Do not add spaces around the `=` sign.
> This file is already in `.gitignore` so it will never accidentally be pushed to GitHub.

---

### Step 6 — Run the app

```bash
python app.py
```

You should see this in the terminal:

```
* Running on http://127.0.0.1:5000
* Debug mode: on
```

---

### Step 7 — Open the app

Open your browser and go to:

```
http://127.0.0.1:5000
```

When you paste a job posting and click **Extract Requirements**, you should see this in the terminal:

```
[jd_parser] ✓ Gemini — field: software engineering
```

This confirms Gemini AI is working with your key. ✅

---

### What if I skip the API key?

**The app still works.** If no `.env` file is found or the key is missing, the app automatically falls back to keyword-based matching. You will see this in the terminal:

```
[jd_parser] No GEMINI_API_KEY — keyword fallback
```

Keyword fallback works well for most common fields but is less accurate than Gemini for niche roles and unusual job descriptions. We recommend setting up the key for the best experience.

---

## 🧭 How to Use the App

### Step 1 — Paste the raw job posting
Paste the full job ad into the **Raw Job Posting** column — marketing copy and all. Click **Extract Requirements**. Gemini AI will detect the field and extract skills, education, certifications, and role info automatically.

### Step 2 — Check extracted requirements
Colour-coded chips appear below the raw JD box:
- 🟣 Purple = technical / hard skills
- 🔵 Blue = role keywords
- 🟢 Green = education
- 🟡 Yellow = soft skills
- 🏅 Amber = certifications / licenses

Click any chip to toggle it on or off before auto-filling the Job Description box.

### Step 3 — Upload or paste your resume
In the **Your Resume** column:
- **Drag and drop** a PDF or DOCX file onto the upload zone, or click to browse
- Or click **"Paste instead"** to paste your resume as plain text

### Step 4 — Run Analysis
Click **Run Analysis** to get:
- A score out of 100 with a match label
- A full breakdown of each scoring dimension
- Specific suggestions on what to add or improve
- Missing skills highlighted clearly

---

## 📊 How Scoring Works

| Component | Weight | What it measures |
|---|---|---|
| Tech / Hard Skills | 50% | Exact match + partial credit for related skills |
| Context Similarity | 20% | Semantic + TF-IDF prose similarity |
| Soft Skills | 15% | Proxy-aware — detects "collaborated" as teamwork |
| Role Fit | 10% | Seniority and role type alignment |
| Education | 5% | Degree and qualification match |

**Score labels:**
- 🟢 75–100% — Strong Match
- 🔵 55–74% — Good Match
- 🟡 38–54% — Partial Match
- 🔴 0–37% — Weak Match

---

## 🌍 Sharing the App With Others (No Setup Required for Users)

If you want others to use the app **without any setup**, deploy it online with your own key:

### Option A — Render (recommended, free tier)

1. Push your code to GitHub (`.env` is excluded by `.gitignore`)
2. Go to **https://render.com** → Sign up → New Web Service
3. Connect your GitHub repository
4. Set these values:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python app.py`
5. Go to **Environment** tab → Add environment variable:
   - Key: `GEMINI_API_KEY`
   - Value: your Gemini key
6. Click **Deploy** — Render gives you a public URL
7. Share the URL — anyone can use the app with full AI, no setup needed

### Option B — Railway (free tier)

1. Go to **https://railway.app** → New Project → Deploy from GitHub
2. Connect your repository
3. Go to **Variables** tab → add `GEMINI_API_KEY=your-key`
4. Railway auto-detects Flask and deploys automatically
5. Share the generated URL

> In both cases your API key stays private on the server. Users just open the URL — they never see or need a key.

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Make sure venv is active — you should see `(venv)` in your prompt. Then run `pip install -r requirements.txt` |
| `No module named 'dotenv'` | Run `pip install python-dotenv` |
| Gemini not working | Check your `.env` file exists in the project root, the key has no quotes or extra spaces, and you ran `pip install google-generativeai` |
| App uses keyword fallback | Your `GEMINI_API_KEY` is not being read. Run `python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.environ.get('GEMINI_API_KEY', 'NOT FOUND'))"` — it should print your key |
| 404 on `/parse_jd` | Make sure you replaced `app.py` with the latest version where all routes are above the `if __name__ == '__main__':` line |
| Upload shows "Network error" | The Flask server is not running — open a terminal, activate venv, run `python app.py` |
| `.doc` files not supported | Open the file in Word, go to File → Save As → choose `.docx`, then re-upload |
| Wrong venv active | Run `where pip` (Windows) or `which pip` (Mac/Linux) — the path must contain your project folder |
| Score seems too low | Use the JD parser first to extract requirements. Pasting raw marketing copy into the Job Description box reduces accuracy |

---

## 🛠 Full Requirements

```
flask>=3.0.0
scikit-learn>=1.3.0
pypdf>=4.0.0
pdfplumber>=0.10.0
python-docx>=1.1.0
python-dotenv>=1.0.0
google-generativeai>=0.7.0
sentence-transformers>=2.7.0
```

> `sentence-transformers` is optional — the app works without it and falls back to TF-IDF automatically. Installing it improves context matching accuracy.

---
## Screenshots
![App](<Screenshot 2026-02-23 222800.png>)
![Add Job Des](<Screenshot 2026-02-23 222850.png>)
![Add Resume](<Screenshot 2026-02-23 222908.png>)
![Run Analysis](<Screenshot 2026-02-23 222924.png>)
---

## 📄 License

MIT — free to use, modify, and share.

---

## 🙏 Built With

- [Flask](https://flask.palletsprojects.com/) — Python web framework
- [Google Gemini](https://aistudio.google.com/) — AI job description parsing
- [sentence-transformers](https://www.sbert.net/) — Semantic resume matching
- [pdfplumber](https://github.com/jsvine/pdfplumber) — PDF text extraction
- [scikit-learn](https://scikit-learn.org/) — TF-IDF similarity scoring