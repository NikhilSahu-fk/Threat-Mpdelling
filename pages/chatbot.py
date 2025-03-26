import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

st.title("💬 Gemini Chatbot")

api_key = st.secrets["GEMINI_API_KEY"]
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)

# Retrieve stored outputs
prd_content = st.session_state.get("prd_content", "No PRD data available.")
dfd_mermaid = st.session_state.get("dfd_mermaid", "No DFD generated yet.")
threat_model = st.session_state.get("threat_model", "No Threat Model generated yet.")
dfd_trust_boundary = st.session_state.get("dfd_trust_boundary", "No DFD with Trust Boundaries generated yet.")

# Display extracted PRD content and generated outputs
with st.expander("📄 PRD Data Context", expanded=False):
    st.write(prd_content)

with st.expander("📊 Generated DFD", expanded=False):
    st.code(dfd_mermaid, language="mermaid")

with st.expander("🛡️ Generated Threat Model", expanded=False):
    st.write(threat_model)

with st.expander("🔒 DFD with Trust Boundaries", expanded=False):
    st.code(dfd_trust_boundary, language="mermaid")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input field
if user_input := st.chat_input("Ask me anything:"):
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # Include PRD and generated outputs in chat context
    history = [
        {"role": "system", "content": f"Product Requirements Document (PRD): {prd_content}"},
        {"role": "system", "content": f"DFD Mermaid Code: {dfd_mermaid}"},
        {"role": "system", "content": f"Threat Model: {threat_model}"},
        {"role": "system", "content": f"DFD with Trust Boundaries: {dfd_trust_boundary}"},
    ] + [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]

    # Generate AI response
    response = model.invoke(history)

    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.messages.append({"role": "assistant", "content": response.content})
