# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from gensim.models import Word2Vec
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# nltk.download('punkt')
# nltk.download('stopwords')

 

# # Sample JD and Resume (Replace with your real text)
# job_description = """
# We are looking for an experienced AWS Developer with a strong background in Python. 
# Must have hands-on experience in deploying applications using AWS Lambda and API Gateway. 
# Experience with S3, EC2, and DynamoDB is required. 
# Good understanding of serverless architecture is a plus. 
# Excellent communication and problem-solving skills are necessary.
# """

# resume_text = """
# EC2 DynamoDB AWS Developer communication • Interned as a Full Stack python Developer, skilled in Python, HTML, CSS, JavaScript, MySQL, and GitHub.
# • Collaborated effectively in team projects and quickly adapted to new technologies.
# • Known for fast learning and efficient task completion, ensuring project goals were consistently achieved.
# • Introduced new ideas and approaches to the team, enhancing project workflows and problem-solving
 
# """


# # Preprocessing function
# def preprocess(text):
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)
#     words = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     return [word for word in words if word not in stop_words]

# # Train Word2Vec model
# def train_word2vec(corpus):
#     return Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)

# # Get average vector
# def get_avg_vector(words, model):
#     vectors = [model.wv[word] for word in words if word in model.wv]
#     return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# # Main logic
# def main():
#     jd_words = preprocess(job_description)
#     resume_words = preprocess(resume_text)

#     # Train on both
#     model = train_word2vec([jd_words, resume_words])

#     # Get vectors
#     jd_vector = get_avg_vector(jd_words, model)
#     resume_vector = get_avg_vector(resume_words, model)

#     # Cosine similarity
#     similarity = cosine_similarity([jd_vector], [resume_vector])[0][0]
#     print(f"\n✅ Resume matches JD by: {similarity * 100:.2f}%")

# if __name__ == '__main__':
#     main()




#ingrading the resume data and sort to top must data which is relvent to jd 

# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from gensim.models import Word2Vec
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# nltk.download('punkt')
# nltk.download('stopwords')

# # --- 1. Job Description (4 lines) ---
# job_description = """
# We are looking for an experienced AWS Developer with a strong background in Python.
# Must have hands-on experience in deploying applications using AWS Lambda and API Gateway.
# Experience with S3, EC2, and DynamoDB is required.
# Good understanding of serverless architecture is a plus.
# """

# # --- 2. Resume (30 lines) ---
# resume_text = """
# Name: John Doe
# Email: john.doe@example.com
# Phone: 123-456-7890

# Hobbies: Playing chess, painting, hiking, reading novels

# Languages Known: English, Spanish

# Participated in college fests and managed cultural events

# Interned as a Full Stack Python Developer
# Skilled in Python, HTML, CSS, JavaScript, MySQL, GitHub
# Built CI/CD pipelines using GitHub Actions and AWS CodePipeline
# Developed backend systems and RESTful APIs
# Worked on serverless deployment using AWS Lambda and API Gateway
# Used AWS services: S3, EC2, DynamoDB
# Experience with IAM roles and CloudWatch monitoring
# Built dashboards with Grafana and CloudWatch
# Worked in Agile teams with Scrum practices
# Used Docker for containerization and Kubernetes for orchestration
# Experience with unit testing using PyTest
# Collaborated with DevOps teams for production deployment
# Wrote reusable and optimized Python code
# Strong debugging and optimization skills
# Good communication and problem-solving abilities
# Experience in cross-functional team collaboration
# Basic knowledge of ReactJS and frontend frameworks
# Used Postman for API testing and documentation
# Created SQL queries for database operations
# Deployed apps on AWS EC2 and Elastic Beanstalk
# Integrated third-party APIs into existing systems
# Handled version control using Git and GitHub
# Explored AWS Amplify for frontend deployments
# Knowledge of NoSQL databases like MongoDB
# Participated in code reviews and peer programming
# Adapted quickly to new technologies and tools
# Capable of working independently or in teams
# """

# # --- 3. Preprocessing function ---
# def preprocess(text):
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)
#     words = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     return [word for word in words if word not in stop_words]

# # --- 4. Train Word2Vec ---
# def train_word2vec(corpus):
#     return Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)

# # --- 5. Get average vector ---
# def get_avg_vector(words, model):
#     vectors = [model.wv[word] for word in words if word in model.wv]
#     return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# # --- 6. Main logic ---
# def main():
#     # Preprocess JD (all lines as one block)
#     jd_lines = job_description.strip().split('\n')
#     jd_words = preprocess(' '.join(jd_lines))

#     # Split resume into individual lines
#     resume_lines = resume_text.strip().split('\n')
#     preprocessed_lines = [preprocess(line) for line in resume_lines if line.strip()]

#     # Train Word2Vec on all content
#     model = train_word2vec([jd_words] + preprocessed_lines)

#     # JD Vector
#     jd_vector = get_avg_vector(jd_words, model)

#     # Calculate similarity for each resume line
#     similarities = []
#     for i, line_words in enumerate(preprocessed_lines):
#         line_vector = get_avg_vector(line_words, model)
#         sim = cosine_similarity([jd_vector], [line_vector])[0][0]
#         similarities.append((resume_lines[i].strip(), sim))

#     # Sort and get top 5
#     top_5 = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
#     final_score = np.mean([score for _, score in top_5])

#     # Print result
#     print(f"\n✅ Best 5 lines average match with JD: {final_score * 100:.2f}%\n")
#     print("🔍 Top 5 Resume Lines Matching JD:\n")
#     for i, (line, score) in enumerate(top_5, start=1):
#         print(f"{i}. [{score * 100:.2f}%] {line}")

# if __name__ == '__main__':
#     main()



#comparing the each line of resume with full data with jd data 


# import re
# import numpy as np
# from gensim.models import Word2Vec
# from sklearn.metrics.pairwise import cosine_similarity 
# from nltk.tokenize import word_tokenize,sent_tokenize
# from nltk.util import bigrams, trigrams, ngrams


# # Custom basic stopword list
# stop_words = {
#     'the', 'is', 'in', 'and', 'to', 'with', 'a', 'of', 'for', 'on', 'at',
#     'by', 'an', 'be', 'this', 'that', 'from', 'are', 'as', 'it', 'or',
#     'we', 'our', 'have', 'has'
# }

# # Basic tokenizer
# def tokenize(text):
#     text = text.lower()
#     text = re.sub(r'[^a-z0-9\s]', '', text)
#     return [word for word in text.split() if word not in stop_words]

# # Job Description and Resume
# jd_lines = [
#     "We are looking for an experienced AWS Developer with a strong background in Python.",
#     "Must have hands-on experience in deploying applications using AWS Lambda and API Gateway.",
#     "Experience with S3, EC2, and DynamoDB is required.",
#     "Good understanding of serverless architecture is a plus."
# ]


# """(39.6%) Worked on serverless deployment using AWS Lambda and API Gateway
# - (34.15%) Used AWS services: S3, EC2, DynamoDB
# - (27.5%) Strong debugging and optimization skills
# - (23.54%) Handled version control using Git and GitHub
# - (22.15%) Experience with IAM roles and CloudWatch monitoring"""

# resume_lines = [
#     "Name: John Doe",
#     "Email: john.doe@example.com",
#     "Phone: 123-456-7890",
#     "Hobbies: Playing chess, painting, hiking, reading novels",
#     "Languages Known: English, Spanish",
#     "Participated in college fests and managed cultural events",
#     "Interned as a Full Stack Python Developer",
#     "Skilled in Python, HTML, CSS, JavaScript, MySQL, GitHub",
#     "Built CI/CD pipelines using GitHub Actions and AWS CodePipeline",
#     "Developed backend systems and RESTful APIs",
#     "Worked on serverless deployment using AWS Lambda and API Gateway",
#     "Used AWS services: S3, EC2, DynamoDB",
#     "Experience with IAM roles and CloudWatch monitoring",
#     "Built dashboards with Grafana and CloudWatch",
#     "Worked in Agile teams with Scrum practices",
#     "Used Docker for containerization and Kubernetes for orchestration",
#     "Experience with unit testing using PyTest",
#     "Collaborated with DevOps teams for production deployment",
#     "Wrote reusable and optimized Python code",
#     "Strong debugging and optimization skills",
#     "Good communication and problem-solving abilities",
#     "Experience in cross-functional team collaboration",
#     "Basic knowledge of ReactJS and frontend frameworks",
#     "Used Postman for API testing and documentation",
#     "Created SQL queries for database operations",
#     "Deployed apps on AWS EC2 and Elastic Beanstalk",
#     "Integrated third-party APIs into existing systems",
#     "Handled version control using Git and GitHub",
#     "Explored AWS Amplify for frontend deployments",
#     "Knowledge of NoSQL databases like MongoDB",
#     "Participated in code reviews and peer programming"
# ]

# # Tokenize everything
# tokenized_jd = tokenize(" ".join(jd_lines))
# tokenized_resume_lines = [tokenize(line) for line in resume_lines]
 

 

# # Train Word2Vec
# model = Word2Vec([tokenized_jd] + tokenized_resume_lines, vector_size=100, window=5, min_count=1, workers=1)
 
 

# # Vectorize function
# def avg_vector(tokens):
#     vectors = [model.wv[word] for word in tokens if word in model.wv]
#     return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# # JD vector
# jd_vector = avg_vector(tokenized_jd)



# Score each resume line
# results = []
# for line, tokens in zip(resume_lines, tokenized_resume_lines):
#     res_vector = avg_vector(tokens)
#     score = cosine_similarity([jd_vector], [res_vector])[0][0]
#     results.append((line, round(score * 100, 2)))

# # Sort and show top matching lines
# sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
# overall_match_score = round(np.mean([score for _, score in results]), 2)

# # Output
# print(f"\nOverall Resume-JD Match Score: {overall_match_score}%")
# print("\nTop 5 Most Relevant Resume Lines:")
# for line, score in sorted_results[:5]:
#     print(f"- ({score}%) {line}")



"Now i use that google neews  data "

import re
import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams

# Load pretrained Google Word2Vec model (300-dimensional vectors)
print("Loading model... This might take a minute.")
model = api.load("word2vec-google-news-300")
print("Model loaded!")
