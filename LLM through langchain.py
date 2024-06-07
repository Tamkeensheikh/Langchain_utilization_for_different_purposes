from langchain_community.document_loaders import ReadTheDocsLoader

loader = ReadTheDocsLoader('rtdocs')

docs = loader.load()
print(len(docs))

print(docs[0].page_content)

print(docs[0].page_content.replace('Â ', ' '))

import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
)
    return len(tokens)

tiktoken.encoding_for_model('gpt-3.5-turbo')

token_count =[tiktoken_len(doc.page_content) for doc in docs]

print(f""""Min ;{min(token_count)}
Avg: {int(sum(token_count)/len(token_count))}
Max: {max(token_count)}""")

from matplotlib import pyplot as plt
import seaborn as sns

# set style and color palette for the plot
sns.set_style("whitegrid")
sns.set_palette("muted")

# create histogram
plt.figure(figsize=(12, 6))
sns.histplot(token_count, kde=False, bins=50)

# customize the plot info
plt.title("Token Counts Histogram")
plt.xlabel("Token Count")
plt.ylabel("Frequency")

plt.show()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    #hunk_overalap = 20,
    length_function =tiktoken_len,
    separators=['\n\n','\n',' ','']
)

chunk = text_splitter.split_text(docs[0].page_content)
print(len(chunk))

import hashlib

m = hashlib.md5()

url = docs[0].metadata['source'].replace('rtdocs/','html://')
print (url)

m.update(url.encode('utf-8'))
uid = m.hexdigest()[:12]
print(uid)

import hashlib
import json
from tqdm.auto import tqdm

# Prepare the documents list
documents = []
m = hashlib.md5()

for doc in tqdm(docs):
    url = doc.metadata['source'].replace('rtdocs/', 'https://')
    m.update(url.encode('utf-8'))
    uid = m.hexdigest()[:12]
    chunks = text_splitter.split_text(doc.page_content)
    for i, chunk in enumerate(chunks):
        documents.append({
            'id': f'{uid}-{i}',
            'text': chunk,
            'source': url
        })

print(f"Number of documents: {len(documents)}")

# Write documents to JSON Lines file
with open('train.jsonl', 'w') as f:
    for doc in documents:
        f.write(json.dumps(doc) + '\n')

# Read the documents back from the JSON Lines file
documents = []

with open('train.jsonl', 'r') as f:
    for line in f:
        documents.append(json.loads(line))

print(f"Number of documents read: {len(documents)}")
print("First document:", documents[0])
