# AI Powered Resume Analyser and Job Matching System

This is a college mini-project website built with Python and Streamlit. It demonstrates how resumes can be checked automatically and compared with job requirements.

## What it does
- Upload resumes in PDF or TXT format
- Extract text from the uploaded resume
- Find skills using a predefined skill list
- Compare the resume with a job description using TF-IDF and cosine similarity
- Show a match score and rank resumes
- Suggest sample job roles based on the resume
- Download the results as a CSV file

## How to run
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the website:
   ```bash
   streamlit run app.py
   ```
5. Open the local URL shown in the terminal, usually http://localhost:8501.

## Notes for the college demonstration
- The listed job roles use sample job descriptions; they are not live job vacancies.
- The match percentage measures text similarity. It is not a hiring probability.
- Scanned PDFs may not work if they do not contain selectable text.
- SQLite storage and saved result history are not included yet, matching the planned next step in the project report.
