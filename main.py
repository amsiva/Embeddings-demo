from openrouter import OpenRouter
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()


def generate_embeddings(text :str):
    client = OpenRouter(api_key=os.getenv('OPENROUTER_API_KEY'))
    response = client.embeddings.generate(
        model="openai/text-embedding-3-small",
        input=text,
        dimensions=534
    )

    return response.data[0].embedding


docs = [
    "Sachin is the greates cricketer in India and scored first 200 runs in One day international",
    "Steve Waugh is the greatets captain of course from Australia",
    "Jaq Kallis is the greatest allrounder from South Africa and best among all",
    "Mahela Jayawardne and Sangakkara is the greatest Partnership in cricket",
    "MS Dhoni the greatest captain hold almost all trophies for India",
    "Gayle is super six hitter in West Indies",
    "New Zealand is the lovely team where even loved by all countries in the cricket but no worldcup for them",
    "Shane Warne from Australia, Ashwin from India and Murali from Srilanka are the greatest spinners in the world",
    "Ben Stokes is the great player for England. He got somecups for England",
    "Sachin is the leading run scorer in batting"
]


query = "Who is the best batsman"

docs_embedding = [generate_embeddings(doc) for doc in docs]
query_embedding = [generate_embeddings(query)]


scores = cosine_similarity(query_embedding, docs_embedding)

print(scores)


# Retrieve the document based in the best match
best_index, best_score = max(enumerate(scores), key=lambda x: x[1])

print("Most similar document:")
print(docs[best_index])
print(f"Similarity Score is: {best_score}")
