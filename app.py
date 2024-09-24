from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import login 
import torch
import time
import streamlit as st

def retrieve_documents(db,query, top_k=1):
    results = db.similarity_search(query, k=top_k)
    # Lấy văn bản từ các kết quả tìm kiếm
    documents = [result.page_content for result in results]
    return documents

def get_response(text_generator,prompt):
    response = text_generator(prompt)
    return response[0]['generated_text']

def generate_answer(text_generator, documents):
    system_message = """
        You are an expert in economic.
        Please anwser the following questions, if you can't answer, please say "I don't know".
    """
    user_message = documents
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    output = get_response(text_generator,messages)
    extracted_text = output[-1]['content']

    return extracted_text

def rag(db,text_generator,query):
    # Truy xuất tài liệu từ Faiss
    documents = retrieve_documents(db,query)

    # Sinh câu trả lời từ truy vấn và tài liệu
    answer = generate_answer(text_generator, documents)
    return answer

def read_vector_db(vector_db_path):
    db = FAISS.load_local(
        vector_db_path,
        HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en-v1.5"),
        allow_dangerous_deserialization= True
    )
    return db

def main():
    st.title("Economic Expert System")
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.int8
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.int8,
        num_return_sequences=1,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    vector_db_path = "vector_store/pdf_paiss/db_paiss"
    db = read_vector_db(vector_db_path)
    start = time.time()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    query = st.chat_input("Enter your query: ")
    if query:
        answer = rag(db,text_generator,query)
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )
        with st.chat_message('user'):
            st.markdown(query)
            
        with st.chat_message('assistant'):
            full_res = ""
            holder = st.empty()
            for word in answer.split():
                full_res += word + " "
                time.sleep(0.1)
                holder.markdown(full_res + "▌")
            holder.markdown(full_res)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_res
            }
        )
    else:
        print("Please enter your query")
    end = time.time()
    print("Time to process query: ", end-start)

if __name__ == "__main__":
    main()