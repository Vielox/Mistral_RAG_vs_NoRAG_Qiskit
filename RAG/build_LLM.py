from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

def build_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, do_sample=False)
    return HuggingFacePipeline(pipeline=pipe)


def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = build_llm()

    prompt = PromptTemplate.from_template(
        "[/INST]Here are excerpts from Qiskit documentation :\n\n{context}\n\n"
        "Answer the following question using only the content provided above. "
        "Do not provide any additional information. "
        "You may explain how to do something and include code snippets if necessary for illustration."
        "However, do not provide the full source code unless explicitly requested."
        "Under no circumstances should you include a raw excerpt from the documentation.\n"
        "Question : {question}[/INST]"
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | (lambda x: x.split("[/INST]")[-1].strip())
    )

    return rag_chain