# app.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Expanded list including Law, Medical, Commerce, and diverse Engineering fields
jobs = [
    # --- Engineering & Tech ---
    {"title": "Software Developer", "skills": "python django flask sql javascript react git backend frontend fullstack development"},
    {"title": "Data Analyst", "skills": "python sql tableau excel pandas statistics data visualization powerbi analyst"},
    {"title": "Cybersecurity Specialist", "skills": "networks linux cryptography security siem wireshark pentesting firewalls hacking"},
    {"title": "Mechatronics Engineer", "skills": "mechanical engineer arduino ide robotics cad solidworks electronics c++ embedded"},
    {"title": "Mechanical Design Engineer", "skills": "mechanical autocad solidworks ansys cad design manufacturing materials"},
    {"title": "Civil Engineer", "skills": "civil structural design autocad revit construction site management surveying estimation"},
    
    # --- Law & Legal ---
    {"title": "Corporate Lawyer", "skills": "law legal corporate contracts compliance mergers acquisitions research litigate governance"},
    {"title": "Criminal Defense Attorney", "skills": "law legal criminal defense litigation court trials advocacy investigation research evidence"},
    {"title": "Intellectual Property (IP) Lawyer", "skills": "law legal patents trademarks copyright ip litigation research technology licensing"},
    
    # --- Commerce & Finance ---
    {"title": "Chartered Accountant / Financial Analyst", "skills": "commerce accounting finance audit tax excel corporate finance gst bookkeeping tally valuation"},
    {"title": "Investment Banker", "skills": "commerce finance investment banking valuation modeling excel m&a markets portfolio trading"},
    {"title": "Marketing Manager", "skills": "commerce marketing seo sem branding social media strategy analytics campaign market research sales"},
    {"title": "Human Resource (HR) Manager", "skills": "commerce hr management recruitment employee relations onboarding payroll training talent development"},
    
    # --- Medical & Healthcare ---
    {"title": "General Physician / Medical Doctor", "skills": "medical healthcare clinical diagnosis treatment patient care medicine pharmacology anatomy mbbs"},
    {"title": "Registered Nurse", "skills": "medical healthcare nursing patient care triage vital signs hospital emergency medicine icu"},
    {"title": "Pharmacist", "skills": "medical healthcare pharmacy pharmacology drugs prescription dosage compounding chemistry counseling"},
    {"title": "Data Scientist (Bioinformatics)", "skills": "medical genomics bioinformatics python r statistics healthcare biology data analysis research"}
]

def main():
    print("--- Job Recommender System Active ---")
    user_skills = input("Enter your skills, interests, or background: ")
    
    job_descriptions = [job["skills"] for job in jobs]
    all_texts = job_descriptions + [user_skills]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    results = sorted(zip(jobs, scores), key=lambda x: x[1], reverse=True)
    
    print("\nRecommended Jobs:")
    matched = False
    for job, score in results:
        match_percentage = score * 100
        if match_percentage > 0:
            print(f"🎯 {job['title']} (Match: {match_percentage:.1f}%)")
            matched = True
        else:
            print(f"⚪ {job['title']} (No Match)")
            
    if not matched:
        print("\n💡 Tip: Try adding more specific keywords (e.g., 'law', 'accounting', 'mbbs', 'python', 'cad').")

if __name__ == "__main__":
    main()