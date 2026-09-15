import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# STEP 1: Define Job Description and Resumes
print("--- STEP 1: Input Job Description & Candidate Resumes ---")
job_description = """
Looking for a Data Scientist proficient in python, sql, machine learning, 
scikit-learn, pandas, numpy, and matplotlib.
"""

resumes = [
    "Data Analyst with expertise in python, sql, pandas, numpy, and matplotlib.",
    "Senior ML Engineer skilled in python, sql, machine learning, scikit-learn, and pandas.",
    "Frontend Web Developer experienced in html, css, javascript, react, and nodejs."
]

candidate_names = ["Candidate 1 (Data Analyst)", "Candidate 2 (ML Engineer)", "Candidate 3 (Web Developer)"]

# STEP 2: Compute Match Scores using Cosine Similarity
print("\n--- STEP 2: Calculating Match Scores ---")
documents = [job_description] + resumes

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)

# Compare Job Description (index 0) against all resumes (index 1 to end)
similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# STEP 3: Rank Candidates
print("\n--- STEP 3: Candidate Rankings ---")
ranking_df = pd.DataFrame({
    'Candidate': candidate_names,
    'Match_Score_%': [round(score * 100, 2) for score in similarity_scores]
}).sort_values(by='Match_Score_%', ascending=False).reset_index(drop=True)

print(ranking_df)

# STEP 4: Skill Gap Analysis
print("\n--- STEP 4: Skill Gap Analysis ---")
required_skills = {'python', 'sql', 'machine learning', 'scikit-learn', 'pandas', 'numpy', 'matplotlib'}

for name, resume in zip(candidate_names, resumes):
    resume_words = set(resume.lower().replace(',', '').split())
    found_skills = required_skills.intersection(resume_words)
    missing_skills = required_skills - found_skills
    
    print(f"\n{name}:")
    print(f" -> Found Skills ({len(found_skills)}): {list(found_skills)}")
    print(f" -> Missing Skills ({len(missing_skills)}): {list(missing_skills)}")
