from bs4 import BeautifulSoup
import requests
from langchain.schema import Document

def fetch_qiskit_docs(urls):
    documents = []
    for url in urls:
        print(f"Retrieving : {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            main_content = soup.find("div", class_="bd-content") or soup.find("div", {"role": "main"}) or soup.body
            if main_content:
                text = main_content.get_text(separator="\n").strip()
                documents.append(Document(page_content=text, metadata={"source": url}))
            else:
                print(f"⚠️ Aucun contenu principal trouvé sur {url}")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de {url} : {e}")
    return documents