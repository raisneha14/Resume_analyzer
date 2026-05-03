from flask import Flask, request, jsonify, render_template
import PyPDF2
import io
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")

skills_db = [
    "python","java","machine learning","sql","html","css","javascript",
    "react","android","pandas","numpy","statistics","excel","power bi",
    "linux","network security","ethical hacking","cryptography","firewall",
    "deep learning","nlp","flask","django","c++","data analysis", "Machine learning",
      "snowflakes", "fluttter", " dart", "kotlin", 
    
]

role_db = {
    "data scientist": ["python","machine learning","statistics","pandas","numpy"],
    "web developer": ["html","css","javascript","react"],
    "android developer": ["java","kotlin","android"],
    "data analyst": ["excel","sql","python","power bi"],
    "cyber security": ["network security","ethical hacking","linux","cryptography","firewall"],
    "software engineering": ["java", "dart", "kotlin"]

}


def extract_text(file_bytes):
    text = ""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.lower()

def extract_skills(text):
    return [skill for skill in skills_db if skill in text]

def get_required_skills(job_title):
    job_title = job_title.lower()
    for role in role_db:
        if role in job_title:
            return role_db[role]
    return []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["file"]
    job_title = request.form["job_title"]

    resume_text = extract_text(file.read())
    resume_skills = extract_skills(resume_text)

    required_skills = get_required_skills(job_title)

    matched = list(set(resume_skills) & set(required_skills))
    missing = list(set(required_skills) - set(resume_skills))

    resume_vec = model.encode(resume_text, convert_to_tensor=True)
    job_vec = model.encode(job_title, convert_to_tensor=True)

    semantic_score = float(util.pytorch_cos_sim(resume_vec, job_vec)[0][0]) * 100
    skill_score = (len(matched)/len(required_skills))*100 if required_skills else 0

    final_score = (semantic_score * 0.7) + (skill_score * 0.3)

    if final_score > 70:
        decision = "Strong Fit"
        category = "Highly Recommended"
    elif final_score > 50:
        decision = "Moderate Fit"
        category = "Can Apply"
    else:
        decision = "Weak Fit"
        category = "Needs Improvement"

    return jsonify({
        "match_score": round(final_score,2),
        "decision": decision,
        "category": category,
        "matched_skills": matched,
        "missing_skills": missing
    })

if __name__ == "__main__":
    app.run(debug=True)
