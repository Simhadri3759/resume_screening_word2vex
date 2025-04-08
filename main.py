import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity 
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.util import bigrams, trigrams, ngrams


# Custom basic stopword list
stop_words = {
    'the', 'is', 'in', 'and', 'to', 'with', 'a', 'of', 'for', 'on', 'at',
    'by', 'an', 'be', 'this', 'that', 'from', 'are', 'as', 'it', 'or',
    'we', 'our', 'have', 'has'
}

# Basic tokenizer
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return [word for word in text.split() if word not in stop_words]

# Job Description and Resume
jd_lines = [
    "We are looking for an experienced AWS Developer with a strong background in Python.",
    "Must have hands-on experience in deploying applications using AWS Lambda and API Gateway.",
    "Experience with S3, EC2, and DynamoDB is required.",
    "Good understanding of serverless architecture is a plus."
]


"""(39.6%) Worked on serverless deployment using AWS Lambda and API Gateway
- (34.15%) Used AWS services: S3, EC2, DynamoDB
- (27.5%) Strong debugging and optimization skills
- (23.54%) Handled version control using Git and GitHub
- (22.15%) Experience with IAM roles and CloudWatch monitoring"""

resume_lines = [
    "Name: John Doe",
    "Email: john.doe@example.com",
    "Phone: 123-456-7890",
    "Hobbies: Playing chess, painting, hiking, reading novels",
    "Languages Known: English, Spanish",
    "Participated in college fests and managed cultural events",
    "Interned as a Full Stack Python Developer",
    "Skilled in Python, HTML, CSS, JavaScript, MySQL, GitHub",
    "Built CI/CD pipelines using GitHub Actions and AWS CodePipeline",
    "Developed backend systems and RESTful APIs",
    "Worked on serverless deployment using AWS Lambda and API Gateway",
    "Used AWS services: S3, EC2, DynamoDB",
    "Experience with IAM roles and CloudWatch monitoring",
    "Built dashboards with Grafana and CloudWatch",
    "Worked in Agile teams with Scrum practices",
    "Used Docker for containerization and Kubernetes for orchestration",
    "Experience with unit testing using PyTest",
    "Collaborated with DevOps teams for production deployment",
    "Wrote reusable and optimized Python code",
    "Strong debugging and optimization skills",
    "Good communication and problem-solving abilities",
    "Experience in cross-functional team collaboration",
    "Basic knowledge of ReactJS and frontend frameworks",
    "Used Postman for API testing and documentation",
    "Created SQL queries for database operations",
    "Deployed apps on AWS EC2 and Elastic Beanstalk",
    "Integrated third-party APIs into existing systems",
    "Handled version control using Git and GitHub",
    "Explored AWS Amplify for frontend deployments",
    "Knowledge of NoSQL databases like MongoDB",
    "Participated in code reviews and peer programming"
]

# Tokenize everything
tokenized_jd = tokenize(" ".join(jd_lines))
tokenized_resume_lines = [tokenize(line) for line in resume_lines]
 

 

# Train Word2Vec
model = Word2Vec([tokenized_jd] + tokenized_resume_lines, vector_size=100, window=5, min_count=1, workers=1)
 
 

# Vectorize function
def avg_vector(tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# JD vector
jd_vector = avg_vector(tokenized_jd)



# Score each resume line
results = []
for line, tokens in zip(resume_lines, tokenized_resume_lines):
    res_vector = avg_vector(tokens)
    score = cosine_similarity([jd_vector], [res_vector])[0][0]
    results.append((line, round(score * 100, 2)))

# Sort and show top matching lines
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
overall_match_score = round(np.mean([score for _, score in results]), 2)

# Output
print(f"\nOverall Resume-JD Match Score: {overall_match_score}%")
print("\nTop 5 Most Relevant Resume Lines:")
for line, score in sorted_results[:5]:
    print(f"- ({score}%) {line}")



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
