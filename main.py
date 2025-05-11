from utils.nowarn import *
from RAG.fetch import *
from RAG.chunks import *
from RAG.build_LLM import *
from huggingface_hub import login


login("")
qiskit_urls = [
    "https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit",
    "https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.Instruction",
    "https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.Gate"
]
if __name__ == "__main__":
    print("Downloading qiskit documentation...")
    raw_docs = fetch_qiskit_docs(qiskit_urls)

    print("Splitting files...")
    chunks = split_documents(raw_docs)

    print("Indexing FAISS...")
    vectorstore = create_vectorstore(chunks)

    print("RAG channeling...")
    rag_chain = build_rag_chain(vectorstore)

    print("\n Ask your question about Qiskit (type 'exit' to leave)")
    while True:
        query = input("> ")
        if query.lower() in {"exit", "quit"}:
            break
        print("\n Results with basic LLM :")
        response = build_llm().invoke(query)
        print(f" Answer : {response.strip()}\n")

        print("\n Results with RAG :")
        rag_response = rag_chain.invoke(query)
        print(f" Answer : {rag_response}\n")
