import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://backend:8000")
HIST_ENDPOINT = f"{API_URL}/history"
UPLOAD_ENDPOINT = f"{API_URL}/upload"
ASK_ENDPOINT = f"{API_URL}/ask"
HEALTH_ENDPOINT = f"{API_URL}/healthcheck"

st.set_page_config(page_title="RAG Gemini", layout="centered")

def check_backend():
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=3)
        if r.status_code == 200 or r.text.strip().lower() == "ok":
            return True
    except:
        pass
    return False

def main():
    st.title("🔎💡 Questionnez vos documents avec Gemini")
    if not check_backend():
        st.error("Le backend FastAPI n’est pas joignable.")
        st.stop()

    with st.expander("📤 Uploader un document", expanded=False):
        uploaded_file = st.file_uploader(
            "Sélectionnez un document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"]
        )
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            resp = requests.post(UPLOAD_ENDPOINT, files=files)
            if resp.status_code == 200:
                st.success(f"Fichier {uploaded_file.name} uploadé et indexé !")
            else:
                st.error(f"Erreur : {resp.json().get('detail','')}")

    st.markdown("---")
    st.subheader("🤖 Posez votre question…")

    question = st.text_input("Votre question", key="question_input")

    if st.button("Envoyer", type="primary") and question.strip():
        with st.spinner("Gemini réfléchit..."):
            r = requests.post(ASK_ENDPOINT, json={"question": question})
            if r.status_code == 200:
                data = r.json()
                st.markdown("### 📝 Réponse")
                st.success(data["answer"])
                st.markdown("### 📚 Sources")
                for src in data.get("sources", []):
                    with st.expander(src["title"]):
                        st.write(src["chunk"])
            else:
                st.error("Erreur lors de la requête.")

    st.markdown("---")
    st.markdown("### 🧾 Historique de vos questions")
    hist = []
    try:
        r = requests.get(HIST_ENDPOINT)
        if r.status_code == 200:
            hist = r.json()
    except:
        pass
    if hist:
        for item in reversed(hist[-10:]):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"*A:* {item['answer']}", unsafe_allow_html=True)
            for src in item.get("sources", []):
                st.markdown(f"- Source : *{src['title']}*")
    else:
        st.info("Aucun historique récent.")

    st.markdown("---")
    st.markdown(
        "<small>Propulsé par FastAPI, ChromaDB, Gemini ✨</small>", unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
