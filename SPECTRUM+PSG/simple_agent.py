import os
import gradio as gr
import pandas as pd
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from googletrans import Translator

# Load environment variables
load_dotenv()

# Set API token
os.environ["AIPROXY_TOKEN"] = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjEwMDIzMzNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.eawnQNAiOzfVMAqybYGuhmWKp4bY964C2N0mwWo6Lko"

# Initialize language model and embeddings
llm = ChatOpenAI(
    base_url="https://aiproxy.sanand.workers.dev/openai/v1",
    api_key=os.environ["AIPROXY_TOKEN"],
    model="gpt-4o-mini",
    temperature=0.2
)

embeddings = OpenAIEmbeddings(
    base_url="https://aiproxy.sanand.workers.dev/openai/v1",
    api_key=os.environ["AIPROXY_TOKEN"],
    model="text-embedding-3-small"
)

# Initialize translator
translator = Translator()

# Sample medical knowledge base (in a real application, you would load this from files)
medical_data = {
    "diseases": [
        {
            "name": "Diabetes",
            "symptoms": ["Increased thirst", "Frequent urination", "Extreme hunger", "Unexplained weight loss"],
            "precautions": ["Regular blood sugar monitoring", "Balanced diet", "Regular exercise"],
            "medications": ["Metformin", "Insulin", "Glipizide"],
            "explanation": "Diabetes is diagnosed when blood glucose levels are consistently elevated due to insufficient insulin production or insulin resistance."
        },
        {
            "name": "Hypertension",
            "symptoms": ["Headaches", "Shortness of breath", "Nosebleeds", "Flushing"],
            "precautions": ["Low sodium diet", "Regular exercise", "Stress management"],
            "medications": ["Lisinopril", "Amlodipine", "Hydrochlorothiazide"],
            "explanation": "Hypertension is diagnosed when blood pressure readings consistently show systolic pressure above 130 mmHg or diastolic pressure above 80 mmHg."
        }
    ]
}

# Save sample data to CSV for vector store creation
def create_sample_dataset():
    rows = []
    for disease in medical_data["diseases"]:
        rows.append({
            "disease": disease["name"],
            "type": "symptoms",
            "content": ", ".join(disease["symptoms"]),
            "explanation": disease["explanation"]
        })
        rows.append({
            "disease": disease["name"],
            "type": "precautions",
            "content": ", ".join(disease["precautions"]),
            "explanation": disease["explanation"]
        })
        rows.append({
            "disease": disease["name"],
            "type": "medications",
            "content": ", ".join(disease["medications"]),
            "explanation": disease["explanation"]
        })

    df = pd.DataFrame(rows)
    df.to_csv("medical_knowledge.csv", index=False)
    return "medical_knowledge.csv"

# Create vector store from knowledge base
def create_vector_store():
    # Create sample dataset
    csv_path = create_sample_dataset()

    # Load documents
    loader = CSVLoader(file_path=csv_path, csv_args={'delimiter': ','})
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Create vector store
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Create custom prompt template for explainable responses
custom_prompt = PromptTemplate.from_template("""
You are a multilingual Healthcare Diagnostic AI assistant. Your goal is to provide accurate medical information
while explaining the reasoning behind your answers.

For each response:
1. Provide a clear answer to the user's question
2. Explain the medical reasoning behind your answer
3. If relevant, include information about symptoms, precautions, treatments, or medications
4. Always clarify that you are providing general information and not a substitute for professional medical advice
                                             
For ANY non-medical questions (including technology, entertainment, politics, general knowledge, 
mathematics, coding, or any other non-medical topic), you MUST respond ONLY with:
"I'm specialized in medical topics only. I cannot answer this question. How can I assist with a health-related concern instead?"                                             

Context information from the medical knowledge base:
{context}

Chat History:
{chat_history}

User Question: {question}

Please provide a helpful, detailed, and explainable response:
""")

# Initialize conversation memory and retrieval chain
def initialize_chain():
    vector_store = create_vector_store()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

    return chain

# Detect language and translate if needed
def detect_and_translate(text, target_lang='en'):
    try:
        detected = translator.detect(text)
        if detected.lang != target_lang:
            translated = translator.translate(text, src=detected.lang, dest=target_lang)
            return translated.text, detected.lang
        return text, detected.lang
    except:
        return text, 'en'  # Default to English if detection fails

# Translate response back to user's language
def translate_response(text, target_lang):
    if target_lang == 'en':
        return text
    try:
        translated = translator.translate(text, src='en', dest=target_lang)
        return translated.text
    except:
        return text  # Return original text if translation fails

# Function to handle user queries
def process_query(user_input, chain):
    # Detect language and translate to English if needed
    english_query, detected_lang = detect_and_translate(user_input)

    # Get response from the chain
    response = chain({"question": english_query})

    # Extract the AI's response
    ai_response = response['answer']

    # Add explanation section if not already included
    if "Explanation:" not in ai_response:
        ai_response += "\n\nExplanation: This information is based on general medical knowledge and should not replace professional medical advice."

    # Translate response back to user's language if needed
    if detected_lang != 'en':
        ai_response = translate_response(ai_response, detected_lang)

    return ai_response

# Initialize the chain
chain = initialize_chain()

# Create Gradio interface
def respond(message, history):
    response = process_query(message, chain)
    return response

# Create Gradio interface
demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(placeholder="Ask me about health conditions, symptoms, or treatments..."),
    outputs="text",
    title="Multilingual Healthcare Diagnostic Assistant with Explainable AI",
    description="Ask questions about medical conditions, symptoms, treatments, or medications in any language. The assistant will provide explanations for its responses."
)

if __name__ == "__main__":
    demo.launch(share=True)

