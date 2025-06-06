import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_job_description(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_resumes(folder_path):
    resumes = []
    file_names = []
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist. Please create it and add resume files.")
        return resumes, file_names

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    resumes.append(content)
                    file_names.append(filename)
                else:
                    print(f"Warning: Resume file '{filename}' is empty and will be skipped.")
    if len(resumes) == 0:
        print(f"No resumes found in '{folder_path}'. Please add .txt files with content.")
    return resumes, file_names


def match_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute cosine similarity between job description (index 0) and all resumes
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarities


def main():
    job_description_path = "job_description.txt"
    resumes_folder = "resumes"

    # Load job description
    if not os.path.exists(job_description_path):
        print(f"Job description file '{job_description_path}' not found. Please add it.")
        return
    job_description = load_job_description(job_description_path)

    # Load resumes
    resumes, file_names = load_resumes(resumes_folder)
    if len(resumes) == 0:
        return  # Exit if no resumes loaded

    # Match resumes
    scores = match_resumes(job_description, resumes)

    # Print resume ranking based on similarity
    ranked_resumes = sorted(zip(file_names, scores), key=lambda x: x[1], reverse=True)

    print("Resume Ranking based on relevance to the Job Description:")
    for i, (file, score) in enumerate(ranked_resumes, start=1):
        print(f"{i}. {file} - Similarity Score: {score:.4f}")


if __name__ == "__main__":
    main()
