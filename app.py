from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_text_from_pdf(path):
    text=""
    pdf_doc=fitz.open(path)
    for page in pdf_doc:
        text+=page.get_text()
    pdf_doc.close()
    return text

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces/newlines
    text = re.sub(r'[^a-zA-Z0-9\s,.]', '', text)  # Remove special characters
    return text.strip()

def retrieve_similar_chunks(query, k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

path=r'data/fpc_manual.pdf'
pdf_text=extract_text_from_pdf(path)
cleaned_text=clean_text(pdf_text)
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)
chunks=text_splitter.split_text(cleaned_text)

embedding_model=SentenceTransformer('all-MiniLm-L6-v2')
chunk_embeddings=embedding_model.encode(chunks)

dimension=len(chunk_embeddings[0])
index=faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings))

model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
tokenizer.pad_token_id = tokenizer.eos_token_id

def generate_answer(query):
    retrieved_chunks = retrieve_similar_chunks(query)
    context = " ".join(retrieved_chunks)
    prompt = f"question: {query} context: {context}"
    
    inputs = tokenizer(prompt, return_tensors="pt",padding=True, truncation=True).to("cuda")
    outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=200,temperature=0.7,do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

st.title("Food Recipes & Safety Precautions")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("I'm all ears, fire away!")

if query:
    answer = generate_answer(query)  # Generate answer
    st.session_state.history.append({"query": query, "answer": answer})

# Display conversation history
for qa in st.session_state.history:
    st.write(f"**Q:** {qa['query']}")
    st.write(f"**A:** {qa['answer']}")