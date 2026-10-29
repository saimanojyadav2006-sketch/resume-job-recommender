import re
import io
from pathlib import Path

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

st.set_page_config(
    page_title="CareerMatch | Resume Job Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --navy:#10243e; --teal:#16a6a1; --muted:#64748b; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
.stApp { background: #f5f7fb; color: var(--navy); }
.block-container { padding-top: 1.5rem; max-width: 1400px; }
.hero {
  background: linear-gradient(120deg,#10243e 0%,#173f61 65%,#167f85 100%);
  color: white; padding: 2.2rem 2.4rem; border-radius: 22px; margin-bottom: 1.3rem;
  box-shadow: 0 14px 35px rgba(16,36,62,.12);
}
.eyebrow { text-transform:uppercase; letter-spacing:2px; color:#8ce4df; font-size:.75rem; font-weight:700; }
.hero h1 { font-family:Manrope,sans-serif; font-size:2.35rem; line-height:1.15; margin:.6rem 0; color:white; }
.hero p { color:#dbe8f2; max-width:720px; font-size:1rem; margin-bottom:0; }
.pill { display:inline-block; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.2); padding:6px 11px; border-radius:999px; margin:1rem 5px 0 0; font-size:.78rem; }
.metric { background:white; border:1px solid #e4eaf1; padding:1rem 1.1rem; border-radius:16px; min-height:104px; }
.metric-label { color:#64748b; font-size:.82rem; }
.metric-value { color:#10243e; font-family:Manrope,sans-serif; font-size:1.75rem; font-weight:800; margin-top:.25rem; }
.panel { background:white; border:1px solid #e4eaf1; padding:1.25rem; border-radius:18px; }
.small-muted { color:#64748b; font-size:.88rem; }
.step { background:white; border:1px solid #e4eaf1; border-radius:14px; padding:1rem; height:100%; }
.stepnum { display:inline-flex; align-items:center; justify-content:center; background:#dff7f5; color:#087e7b; width:30px; height:30px; border-radius:50%; font-weight:800; margin-right:8px; }
div.stButton > button[kind="primary"] { background:#0f8f8b; border:0; border-radius:10px; font-weight:700; padding:.65rem 1rem; }
div.stButton > button[kind="primary"]:hover { background:#087a77; }
[data-testid="stSidebar"] { background:#10243e; }
[data-testid="stSidebar"] * { color:#eef5fb; }
[data-testid="stSidebar"] .stRadio label { color:#eef5fb; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

SKILLS_DB = {
    "Programming": ["python", "java", "c++", "c#", "javascript", "typescript", "r", "sql"],
    "Data & AI": ["data analysis", "data science", "machine learning", "deep learning", "nlp",
                  "natural language processing", "pandas", "numpy", "scikit-learn", "tensorflow",
                  "pytorch", "power bi", "tableau", "excel", "statistics", "data visualization"],
    "Web Development": ["html", "css", "react", "node.js", "flask", "django", "streamlit", "rest api"],
    "Databases & Cloud": ["mysql", "postgresql", "mongodb", "sqlite", "aws", "azure", "google cloud",
                          "docker", "git", "github", "linux"],
    "Professional": ["communication", "teamwork", "problem solving", "leadership", "project management",
                     "critical thinking", "presentation"],
}

SAMPLE_JOBS = {
    "Data Analyst": {
        "description": "Analyze datasets, clean data with Python and Pandas, write SQL queries, create dashboards in Power BI or Tableau, perform exploratory data analysis, and communicate insights using statistics and data visualization.",
        "skills": ["python", "sql", "pandas", "excel", "power bi", "tableau", "statistics", "data analysis", "data visualization"],
        "level": "Entry level"
    },
    "Python Developer": {
        "description": "Develop and maintain Python applications, write clean code, build REST APIs using Flask or Django, work with SQL databases, use Git and GitHub, debug software, and collaborate with a development team.",
        "skills": ["python", "flask", "django", "rest api", "sql", "git", "github", "problem solving"],
        "level": "Entry level"
    },
    "Machine Learning Intern": {
        "description": "Prepare datasets, perform feature engineering, train and evaluate machine learning models, use Python, NumPy, Pandas and scikit-learn, understand statistics, and explain model results.",
        "skills": ["python", "machine learning", "scikit-learn", "pandas", "numpy", "statistics", "data science"],
        "level": "Internship"
    },
    "NLP Intern": {
        "description": "Preprocess text, apply natural language processing techniques, build text classification and information extraction workflows, use Python, NLTK and scikit-learn, and evaluate NLP models.",
        "skills": ["python", "nlp", "natural language processing", "scikit-learn", "machine learning"],
        "level": "Internship"
    },
    "Frontend Developer": {
        "description": "Build responsive user interfaces using HTML, CSS and JavaScript, develop reusable React components, work with APIs, use Git and GitHub, and collaborate on accessible web experiences.",
        "skills": ["html", "css", "javascript", "react", "rest api", "git", "github", "communication"],
        "level": "Entry level"
    },
    "Database Assistant": {
        "description": "Design and query relational databases, write SQL, maintain MySQL or PostgreSQL databases, validate records, document data workflows, and troubleshoot database issues.",
        "skills": ["sql", "mysql", "postgresql", "sqlite", "excel", "problem solving"],
        "level": "Entry level"
    },
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9+#.\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_pdf_text(uploaded_file):
    if PdfReader is None:
        raise RuntimeError("PDF support needs pypdf. Install dependencies from requirements.txt.")
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_resume_text(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(uploaded_file)
    if suffix == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    raise ValueError("Please upload a PDF or TXT file.")

def extract_skills(text):
    cleaned = clean_text(text)
    found = []
    for category, skills in SKILLS_DB.items():
        for skill in skills:
            if re.search(r'(?<![a-z0-9+#])' + re.escape(skill.lower()) + r'(?![a-z0-9+#])', cleaned):
                found.append((skill, category))
    return found

def score_resume(resume_text, job_text):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([clean_text(resume_text), clean_text(job_text)])
    score = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100)
    return round(score, 1)

def job_skill_coverage(resume_skills, job_skills):
    resume_set = {s for s, _ in resume_skills}
    required = set(job_skills)
    if not required:
        return 0.0, []
    matched = sorted(resume_set.intersection(required))
    return round(len(matched) / len(required) * 100, 1), matched

def analyze_uploaded_resumes(files, job_text, job_title):
    rows = []
    for file in files:
        try:
            raw_text = extract_resume_text(file)
            if not raw_text.strip():
                rows.append({"Candidate": file.name, "Match Score": 0.0, "Skill Coverage": 0.0,
                             "Skills Found": "No extractable text", "Matched Skills": "", "Status": "Check file"})
                continue
            skills = extract_skills(raw_text)
            score = score_resume(raw_text, job_text)
            rows.append({
                "Candidate": file.name,
                "Match Score": score,
                "Skill Coverage": None,
                "Skills Found": ", ".join(s for s, _ in skills) or "No listed skills detected",
                "Matched Skills": ", ".join(s for s, _ in skills),
                "Status": "Analyzed",
                "_raw_text": raw_text,
                "_skills": skills,
            })
        except Exception as exc:
            rows.append({"Candidate": file.name, "Match Score": 0.0, "Skill Coverage": 0.0,
                         "Skills Found": str(exc), "Matched Skills": "", "Status": "Error"})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Match Score", ascending=False, kind="stable").reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
    return df

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🎯 CareerMatch")
    st.caption("Resume-based job recommendation")
    page = st.radio("NAVIGATION", ["Job Recommender", "How It Works", "Project Info"], label_visibility="visible")
    st.markdown("---")
    st.markdown("**Pipeline**")
    for label in ["1  Resume upload", "2  Text extraction", "3  NLP cleaning", "4  Skill extraction", "5  TF-IDF matching", "6  Score & ranking"]:
        st.caption(label)
    st.markdown("---")
    st.caption("Prototype • Local processing")

if page == "Job Recommender":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">AI-powered career discovery</div>
      <h1>Find the roles that fit<br>your resume.</h1>
      <p>Upload a resume, choose a target role or paste a job description, and get transparent match scores with skills detected from your resume.</p>
      <span class="pill">Python</span><span class="pill">NLP preprocessing</span><span class="pill">TF-IDF + cosine similarity</span>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown("### 01 / Your resume")
        uploaded_files = st.file_uploader("Upload one or more resumes (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True,
                                          help="For best results, use text-based PDFs. Scanned image-only PDFs may not extract correctly.")
        st.markdown('<div class="small-muted">You can upload multiple resumes to compare candidates, or upload one resume to find suitable roles.</div>', unsafe_allow_html=True)
        st.markdown("### 02 / Target job")
        job_mode = st.radio("Choose input method", ["Recommend jobs for my resume", "Match against a job description"], horizontal=True)
        if job_mode == "Recommend jobs for my resume":
            chosen_job = st.selectbox("Optional role to prioritize", ["No preference"] + list(SAMPLE_JOBS.keys()))
            custom_job_text = ""
            job_title = chosen_job if chosen_job != "No preference" else "Job recommendations"
            job_text = ""
        else:
            chosen_job = "Custom job description"
            job_title = st.text_input("Job title", placeholder="e.g., Junior Data Analyst")
            custom_job_text = st.text_area("Paste job description", height=170, placeholder="Paste the responsibilities, requirements, and skills from the job posting...")
            job_text = custom_job_text
        analyze = st.button("✨ Analyze resume & recommend jobs", type="primary", use_container_width=True)

    with right:
        st.markdown("### Your results")
        st.markdown("""
        <div class="panel">
          <h4 style="margin-top:0;color:#10243e">What you'll receive</h4>
          <p>✓ Ranked job recommendations</p>
          <p>✓ Resume-to-job match percentage</p>
          <p>✓ Skills detected in your resume</p>
          <p>✓ Skills that match the selected role</p>
          <p style="color:#64748b;font-size:.85rem;margin-bottom:0">Scores indicate text similarity, not a guarantee of job eligibility or hiring success.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<div class="step"><span class="stepnum">1</span><b>Upload</b><p class="small-muted">Add a PDF or TXT resume.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="step"><span class="stepnum">2</span><b>Compare</b><p class="small-muted">The system extracts text and compares it with job requirements.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="step"><span class="stepnum">3</span><b>Discover</b><p class="small-muted">Review recommendations and skill gaps.</p></div>', unsafe_allow_html=True)

    if analyze:
        if not uploaded_files:
            st.warning("Please upload at least one PDF or TXT resume.")
        elif job_mode == "Match against a job description" and not custom_job_text.strip():
            st.warning("Please paste a job description before analyzing.")
        else:
            with st.spinner("Reading resumes and calculating match scores..."):
                if job_mode == "Recommend jobs for my resume":
                    rec_rows = []
                    # Each uploaded resume receives recommendations against each sample role.
                    for file in uploaded_files:
                        try:
                            resume_text = extract_resume_text(file)
                            if not resume_text.strip():
                                st.error(f"{file.name}: no readable text was found.")
                                continue
                            found = extract_skills(resume_text)
                            all_skills = ", ".join(s for s, _ in found) or "No listed skills detected"
                            for title, job in SAMPLE_JOBS.items():
                                if chosen_job != "No preference" and title != chosen_job:
                                    continue
                                score = score_resume(resume_text, job["description"] + " " + " ".join(job["skills"]))
                                coverage, matched = job_skill_coverage(found, job["skills"])
                                missing = sorted(set(job["skills"]) - {s for s, _ in found})
                                rec_rows.append({
                                    "Resume": file.name, "Recommended Role": title, "Level": job["level"],
                                    "Match Score": score, "Skill Coverage": coverage,
                                    "Matched Skills": ", ".join(matched) or "None detected",
                                    "Skills to Develop": ", ".join(missing[:8]) or "No listed gaps",
                                    "_raw_text": resume_text, "_skills": found
                                })
                        except Exception as exc:
                            st.error(f"Could not process {file.name}: {exc}")
                    if rec_rows:
                        rec_df = pd.DataFrame(rec_rows).sort_values(["Resume", "Match Score"], ascending=[True, False]).reset_index(drop=True)
                        st.session_state["recommendations"] = rec_df
                        st.session_state["last_mode"] = "recommend"
                else:
                    results = analyze_uploaded_resumes(uploaded_files, job_text, job_title or "Custom job")
                    if not results.empty:
                        st.session_state["match_results"] = results
                        st.session_state["last_mode"] = "match"

    if st.session_state.get("last_mode") == "recommend" and "recommendations" in st.session_state:
        rec_df = st.session_state["recommendations"]
        st.markdown("---")
        st.markdown("## Recommended jobs")
        resume_names = rec_df["Resume"].drop_duplicates().tolist()
        selected_resume = st.selectbox("View recommendations for", resume_names, key="resume_filter")
        view_df = rec_df[rec_df["Resume"] == selected_resume].copy()
        if not view_df.empty:
            best = view_df.iloc[0]
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric"><div class="metric-label">Top role</div><div class="metric-value" style="font-size:1.2rem">{best["Recommended Role"]}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric"><div class="metric-label">Text match</div><div class="metric-value">{best["Match Score"]:.1f}%</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric"><div class="metric-label">Required skills found</div><div class="metric-value">{best["Skill Coverage"]:.1f}%</div></div>', unsafe_allow_html=True)
        display = view_df[["Recommended Role", "Level", "Match Score", "Skill Coverage", "Matched Skills", "Skills to Develop"]].copy()
        display = display.rename(columns={"Match Score": "Text Match %", "Skill Coverage": "Skill Coverage %"})
        st.dataframe(display, use_container_width=True, hide_index=True,
                     column_config={"Text Match %": st.column_config.ProgressColumn("Text Match %", min_value=0, max_value=100, format="%.1f%%"),
                                    "Skill Coverage %": st.column_config.ProgressColumn("Skill Coverage %", min_value=0, max_value=100, format="%.1f%%")})
        st.download_button("Download recommendations as CSV", view_df.drop(columns=["_raw_text", "_skills"], errors="ignore").to_csv(index=False).encode("utf-8"),
                           file_name="job_recommendations.csv", mime="text/csv")
        st.caption("The text-match score is based on TF-IDF/cosine similarity. Skill coverage is the share of the sample role's listed skills detected in the resume. These are prototype estimates.")
        with st.expander("See skills detected in the selected resume"):
            first_skills = view_df.iloc[0]["_skills"]
            if first_skills:
                for category in sorted(set(cat for _, cat in first_skills)):
                    skills = [skill for skill, cat in first_skills if cat == category]
                    st.markdown(f"**{category}:** " + ", ".join(skills))
            else:
                st.info("No predefined skills were detected.")
    elif st.session_state.get("last_mode") == "match" and "match_results" in st.session_state:
        results = st.session_state["match_results"]
        st.markdown("---")
        st.markdown("## Candidate match results")
        valid = results[results["Status"] == "Analyzed"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric"><div class="metric-label">Resumes analyzed</div><div class="metric-value">{len(valid)}</div></div>', unsafe_allow_html=True)
        with c2:
            top_score = float(valid.iloc[0]["Match Score"]) if len(valid) else 0
            st.markdown(f'<div class="metric"><div class="metric-label">Highest text match</div><div class="metric-value">{top_score:.1f}%</div></div>', unsafe_allow_html=True)
        with c3:
            top_name = str(valid.iloc[0]["Candidate"]) if len(valid) else "—"
            st.markdown(f'<div class="metric"><div class="metric-label">Best match</div><div class="metric-value" style="font-size:1.0rem">{top_name}</div></div>', unsafe_allow_html=True)
        display = results[["Rank", "Candidate", "Match Score", "Skills Found", "Status"]].copy()
        display = display.rename(columns={"Match Score": "Match %"})
        st.dataframe(display, use_container_width=True, hide_index=True,
                     column_config={"Match %": st.column_config.ProgressColumn("Match %", min_value=0, max_value=100, format="%.1f%%")})
        st.download_button("Download results as CSV", results.drop(columns=["_raw_text", "_skills"], errors="ignore").to_csv(index=False).encode("utf-8"),
                           file_name="resume_match_results.csv", mime="text/csv")
        with st.expander("View detected skills per resume"):
            for _, row in results.iterrows():
                st.markdown(f"**{row['Candidate']}** — {row['Skills Found']}")
        st.caption("Match percentages represent TF-IDF text similarity to the job description. They are not probabilities of getting hired.")

elif page == "How It Works":
    st.markdown("""
    <div class="hero"><div class="eyebrow">System workflow</div><h1>From resume to recommendation.</h1>
    <p>A transparent six-step pipeline based on the project architecture in your second-review report.</p></div>
    """, unsafe_allow_html=True)
    steps = [
        ("01", "Resume upload", "Recruiter or job seeker uploads a PDF/TXT resume."),
        ("02", "Text extraction", "PyPDF reads text from PDF files; TXT files are decoded directly."),
        ("03", "NLP preprocessing", "Text is lowercased and normalised before comparison."),
        ("04", "Skill extraction", "Resume text is checked against a predefined, category-wise skill keyword list."),
        ("05", "TF-IDF + cosine similarity", "Resume text and job description are converted to TF-IDF vectors and compared."),
        ("06", "Rank & recommend", "Results are sorted by score and shown with skills found and skill gaps."),
    ]
    cols = st.columns(3)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f'<div class="step" style="margin-bottom:1rem"><span class="stepnum">{num}</span><b>{title}</b><p class="small-muted" style="margin-top:.75rem">{desc}</p></div>', unsafe_allow_html=True)
    st.markdown("### Matching formula")
    st.latex(r"\\text{Cosine Similarity}(A,B)=\\frac{A\\cdot B}{\\|A\\|\\|B\\|}\\times 100")
    st.info("TF-IDF and cosine similarity compare word patterns. They may not understand synonyms or context, so use the scores as decision support—not as an automatic hiring decision.")
    st.markdown("### Technologies")
    st.dataframe(pd.DataFrame([
        ["Python", "Main programming language"],
        ["PyPDF / TXT", "Resume text extraction"],
        ["NLTK-style preprocessing", "Text cleaning and token handling"],
        ["Scikit-learn", "TF-IDF vectorisation and cosine similarity"],
        ["Pandas", "Sort and display results"],
        ["Streamlit", "Web interface"],
    ], columns=["Technology", "Purpose"]), use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div class="hero"><div class="eyebrow">5BCA mini project</div><h1>AI Powered Resume Analyser<br>and Job Matching System</h1>
    <p>A lightweight, locally run prototype for resume parsing, skill extraction and job matching.</p></div>
    """, unsafe_allow_html=True)
    st.markdown("### Project scope")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Implemented in this prototype")
        st.markdown("- PDF and TXT resume parsing\n- Text preprocessing\n- Predefined keyword-based skill extraction\n- TF-IDF and cosine similarity\n- Match score and ranking\n- Streamlit user interface\n- Job recommendations against a sample role catalogue")
    with col2:
        st.markdown("#### Next steps")
        st.markdown("- SQLite persistence and result history\n- Expand and maintain the skills database\n- Semantic matching with Sentence Transformers\n- Add more real job descriptions\n- Add a skills-gap analytics dashboard\n- Test with varied resume formats")
    st.warning("This website is a demonstration prototype. Job descriptions in the recommender are sample descriptions, not live job vacancies. Review results manually and do not use the score as the sole hiring criterion.")
    st.markdown("### Privacy")
    st.write("This app processes uploaded resume files in the running Streamlit session. It does not intentionally save them to a database. Avoid uploading sensitive documents to any demo you do not control.")

st.markdown("---")
st.markdown('<p style="text-align:center;color:#8290a3;font-size:.8rem">CareerMatch · AI Powered Resume Analyser & Job Matching System · Student project prototype</p>', unsafe_allow_html=True)
