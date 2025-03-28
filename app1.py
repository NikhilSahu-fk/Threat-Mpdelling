import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import re
import os
from langchain_core.messages import AIMessage
import pandas as pd
import tempfile
import subprocess

api_key = os.getenv("GOOGLE_API_KEY")  # Read from environment variable
# api_key = "Your API Key"
import os
# print(os.getenv("GOOGLE_API_KEY"))

def filter_think_tags(content):
    """Extracts and removes <think> tags from content, displaying their content in a dropdown."""
    think_matches = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
    filtered_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    return filtered_content.strip(), think_matches

def extract_mermaid_code(content):
    """Extracts only the Mermaid syntax code block from LLM response."""
    match = re.search(r'```mermaid\n(.*?)\n```', content, re.DOTALL)
    return match.group(1) if match else content

def generate_threat_model_prompt(app_type, auth_method, platform, is_internet_facing, dfd_mermaid):
    """
    Generates a structured and precise threat modeling prompt using STRIDE methodology.
    This version is optimized for high accuracy and strict adherence to security analysis.

    Parameters:
    - app_type (str): Type of application (e.g., Web, Mobile, API).
    - auth_method (str): Authentication mechanism (e.g., OAuth, JWT, SAML).
    - platform (str): Hosting platform (e.g., AWS, Azure, On-Premises).
    - is_internet_facing (bool): Whether the application is exposed to the internet.
    - dfd_mermaid (str): The Data Flow Diagram (DFD) in Mermaid syntax.

    Returns:
    - str: The structured prompt for the LLM.
    """

    prompt = f"""
    You are a Security Architect specializing in STRIDE Threat Modeling. Your task is to analyze 
    security threats based on the **System Context**, **PRD Document**, and **DFD Diagram**. 
    Follow the **structured format** strictly and ensure accuracy.

    ---
    
    ## **System Context:**
    - **Application Type:** {app_type}
    - **Authentication Method:** {auth_method}
    - **Platform:** {platform}
    - **Internet Facing:** {is_internet_facing}

    ## **Product Requirements Document (PRD):**
    {{context}}

    ## **Data Flow Diagram (DFD) Representation:**
    Below is the **DFD diagram in Mermaid syntax**, representing system components, 
    data flows, and trust boundaries.

    ```mermaid
    {dfd_mermaid}
    ```

    ---

    ## 🔹 **Instructions (Follow Strictly)**
    - **DO NOT provide general security advice** or explanations.
    - **ONLY analyze assets/components listed in PRD + DFD**.
    - **Identify ALL STRIDE threats** per asset/component.
    - **STRICTLY map each threat to OWASP Top 10 (2021).**
    - **Ensure Markdown table is properly formatted.**

    ---

    ## 🔹 **Allowed STRIDE Categories:**
    - **Spoofing:** Impersonation or unauthorized access.
    - **Tampering:** Unauthorized modification of data or processes.
    - **Repudiation:** Ability to deny malicious actions.
    - **Information Disclosure:** Exposure of sensitive data.
    - **Denial of Service:** Overloading or disrupting services.
    - **Elevation of Privilege:** Unauthorized access escalation.

    ## 🔹 **Allowed OWASP Top 10 (2021) Mappings:**
    - **A01: Broken Access Control**
    - **A02: Cryptographic Failures**
    - **A03: Injection**
    - **A04: Insecure Design**
    - **A05: Security Misconfiguration**
    - **A06: Vulnerable and Outdated Components**
    - **A07: Identification and Authentication Failures**
    - **A08: Software and Data Integrity Failures**
    - **A09: Security Logging and Monitoring Failures**
    - **A10: Server-Side Request Forgery (SSRF)**

    ---

    ## **Output Format (Strictly Follow)**
    
    ### **Asset Threat Model Table (Markdown)**
    | Asset/Component | STRIDE Category | Threat Description | Impact | Trust Boundary Breach | OWASP Mapping | Remediation |
    |----------------|----------------|---------------------|--------|----------------------|---------------|-------------|
    | Example Asset | Spoofing | Unauthorized access using weak authentication | High | Yes | A01: Broken Access Control | Enforce MFA and strong authentication |

    ---

    ## **Additional Threat Analysis Sections (After Table)**

    ### **Trust Boundaries**
    - List ALL trust boundaries identified in the DFD.
    - Define security risks if breached.

    ### **Data Flows**
    - Identify **critical data flows** between components.
    - Indicate which flows cross **trust boundaries**.

    ### **Deployment Threat Model**
    - Identify threats from **network exposure, cloud security, container risks, CI/CD security**.
    - Assess risks in **multi-cloud, hybrid, or on-prem deployments**.

    ### **Build Threat Model**
    - Analyze **supply chain risks, software dependencies, and CI/CD vulnerabilities**.

    ### **Assumptions & Open Questions**
    - List **assumptions** made in the threat model.
    - Highlight **uncertain areas requiring further security validation**.

    **Ensure that the Markdown output is correctly formatted.**
    """
    return prompt


def generate_attack_tree_prompt(threat_model_output, document_context):
    """Generates a detailed attack tree in Mermaid syntax using structured attack methodology."""
    prompt = f"""
    Act as a cybersecurity expert with more than 20 years of experience in STRIDE threat modeling.
    Using the following threat model analysis and document context, generate an **attack tree** in **valid Mermaid syntax**.  **Return ONLY the Mermaid code block. Do not include any explanatory text or descriptions.**

    ## **Threat Model Analysis**
    {threat_model_output}

    ## **Document Context**
    {document_context}

    ## **Instructions**
    - Clearly define the **Root Goal** of the attack.
    - Identify **High-Level Attack Paths (Sub-Goals)** an attacker might use.
    - Expand **each attack path** with detailed steps, considering:
      - Code Injection
      - Exploiting vulnerabilities
      - Supply chain attacks
      - Misconfigurations
    - Apply **Logical Operators**:
      - `[AND]` = All conditions must be met.
      - `[OR]` = Any one condition is sufficient.  *(Note:  The LLM may struggle with complex AND/OR logic in Mermaid.  Start simple and iterate.)*
    - Assign **Attributes** to each node (as Mermaid node text or tooltips - see example):
      - **Likelihood** (Low/Medium/High)
      - **Impact** (Low/Medium/High)
      - **Effort** (Low/Medium/High)
      - **Skill Level** (Low/Medium/High)
      - **Detection Difficulty** (Easy/Moderate/Hard)
    - **Output Format:** Return *only* a valid Mermaid code block, enclosed in triple backticks.  **Do not include any other text.**

    ## **Attack Tree Format**
    ```mermaid
    graph TD;
        A[Root Goal: Compromise Application] -->|Path 1| B[Exploit Known Vulnerability<br>Likelihood: High<br>Impact: High];
        A -->|Path 2| C[Inject Malicious Code<br>Likelihood: Medium<br>Impact: Critical];

        B -->|Step 1| B1[Find Vulnerable Component<br>Effort: Low<br>Skill: Low];
        B1 -->|Step 2| B2[Exploit via Remote Code Execution<br>Effort: Medium<br>Skill: Medium];
        B2 -->|Step 3| B3[Gain Unauthorized Access<br>Effort: High<br>Skill: High];

        C -->|Step 1| C1[Identify Injection Point<br>Effort: Medium<br>Skill: Medium];
        C1 -->|Step 2| C2[Insert Malicious Payload<br>Effort: High<br>Skill: High];
        C2 -->|Step 3| C3[Execute Arbitrary Code<br>Effort: High<br>Skill: High];

        classDef critical fill:#ff4d4d,stroke:#000;
        classDef high fill:#ff751a,stroke:#000;
        classDef medium fill:#ffcc00,stroke:#000;
        classDef low fill:#66ccff,stroke:#000;

        B3:::critical;
        C3:::high;
    ```
    """
    return prompt

def generate_threat_model(docs, dfd_mermaid, app_type, auth_method, platform, is_internet_facing, api_key):
    """
    Generates a threat model using PRD (docs) and DFD (Mermaid code) as context.
    
    Parameters:
    - docs (str): The uploaded PRD document.
    - dfd_mermaid (str): The DFD diagram in Mermaid syntax.
    - app_type (str): Type of application (e.g., Web, Mobile, API).
    - auth_method (str): Authentication mechanism (e.g., OAuth, JWT, SAML).
    - platform (str): Hosting platform (e.g., AWS, Azure, On-Premises).
    - is_internet_facing (bool): Whether the application is exposed to the internet.
    - api_key (str): API key for LLM processing.

    Returns:
    - str: Generated Threat Model analysis.
    """
    
    # Initialize embeddings and vector store
    embedder = HuggingFaceEmbeddings()
    text_splitter = SemanticChunker(embedder)
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # Generate Threat Model prompt with PRD + DFD
    threat_model_prompt = generate_threat_model_prompt(app_type, auth_method, platform, is_internet_facing, dfd_mermaid)
    
    # Define LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

    # Use LLM chain to process query with PRD + DFD context
    QA_PROMPT = PromptTemplate.from_template(threat_model_prompt)
    llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
    combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
    qa = RetrievalQA(combine_documents_chain=combine_documents_chain, retriever=retriever)
    
    return qa({"query": "Generate a threat model based on the PRD and DFD diagram provided."})["result"]


def generate_attack_tree(threat_model_output, document_context):
    """Generates an attack tree using the threat model output and document context."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

    attack_tree_prompt = generate_attack_tree_prompt(threat_model_output, document_context)
    
    attack_tree_response = llm.invoke(attack_tree_prompt)  # Returns an AIMessage object

    # 🔹 Extract text content from the AIMessage object
    if isinstance(attack_tree_response, AIMessage):
        attack_tree_response = attack_tree_response.content  # Extract the actual text

    # 🔹 Debugging Check
    if not isinstance(attack_tree_response, str):
        raise TypeError(f"Expected string but got {type(attack_tree_response)}: {attack_tree_response}")

    return attack_tree_response

def generate_dfd_mermaid(prd_content, api_key):
    """Generates a well-structured Mermaid syntax Data Flow Diagram (DFD) from the PRD document."""
    dfd_prompt = f"""
    **Task:** Generate a **Data Flow Diagram (DFD)** using **Mermaid syntax** based on the provided Product Requirements Document (PRD).

    ---
    ## **PRD Document Content**
    {prd_content}

    ---
    ## **Instructions for Mermaid DFD**
    - **Use `graph TD;`** as the structure.
    - Identify and define **four key DFD components**:
      1. **External Entities** (users, third-party services) → Use `[Entity]`
      2. **Processes** (business logic, API gateways) → Use `((Process))`
      3. **Data Stores** (databases, logs, caches) → Use `[Data Store]`
      4. **Data Flows** (connections, interactions) → Use `--> |Label|`

    - **Guidelines:**
      - Ensure all major components from the PRD are represented.
      - Use meaningful, short, and clear labels for each element.
      - Clearly indicate data flow directions using `-->` or `<--`.
      - Avoid generic labels like "Process1", "EntityX"—instead, infer names from the PRD.
      - **Return ONLY the Mermaid code block** without explanations.

    ---
    ## **Example Output**
    ```
    ```mermaid
    graph TD;
        A[User] -->|Login Request| B((Auth Service));
        B -->|Validates| C[User Database];
        C -->|Response| A;
        B -->|Session Token| D[Session Store];
    ```
    ```
    """

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    dfd_response = llm.invoke(dfd_prompt)

    if isinstance(dfd_response, AIMessage):
        dfd_response = dfd_response.content

    return extract_mermaid_code(dfd_response)


def render_dfd_diagram(mermaid_code):
    """Renders the DFD diagram in Streamlit using Mermaid.js."""
    mermaid_html = f"""
    <div class="mermaid">
        {mermaid_code}
    </div>
    <script type="module">
        import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
        mermaid.initialize({{ startOnLoad: true }});
    </script>
    """
    st.components.v1.html(mermaid_html, height=800, scrolling=True)


def save_mermaid_to_file(mermaid_code, filename="dfd_diagram.mmd"):
    """Saves Mermaid syntax to a file."""
    folder = "saved_diagrams"
    os.makedirs(folder, exist_ok=True)  # Ensure the folder exists

    file_path = os.path.join(folder, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(mermaid_code)

    return file_path  # Return the file path for reference

def extract_threat_model_sections(threat_model_output):
    """Extracts different sections from the threat model response."""
    sections = {
        "table": None,
        "trust_boundaries": [],
        "data_flows": [],
        "deployment_threat_model": [],
        "build_threat_model": [],
        "assumptions": []
    }

    lines = threat_model_output.strip().split("\n")

    # Extract the Markdown table
    table_start = -1
    for i, line in enumerate(lines):
        if line.startswith("|") and "Asset/Component" in line:
            table_start = i
            break

    if table_start != -1:
        headers = [col.strip() for col in lines[table_start].split("|") if col.strip()]
        table_rows = [
            [col.strip() for col in row.split("|") if col.strip()]
            for row in lines[table_start + 2:]  # Skip separator row
            if row.startswith("|")
        ]
        sections["table"] = pd.DataFrame(table_rows, columns=headers)

    # Extract different sections using regex
    def extract_section(pattern, target_list):
        matches = re.findall(pattern, threat_model_output, re.MULTILINE)
        for match in matches:
            target_list.append(f"- **{match[0].strip()}**: {match[1].strip()}")

    extract_section(r"###\s\*\*Trust Boundaries\*\*\s*\n- (.*?)\n(.*?)\n", sections["trust_boundaries"])
    extract_section(r"###\s\*\*Data Flows\*\*\s*\n- (.*?)\n(.*?)\n", sections["data_flows"])
    extract_section(r"###\s\*\*Deployment Threat Model\*\*\s*\n- (.*?)\n(.*?)\n", sections["deployment_threat_model"])
    extract_section(r"###\s\*\*Build Threat Model\*\*\s*\n- (.*?)\n(.*?)\n", sections["build_threat_model"])
    extract_section(r"###\s\*\*Assumptions & Open Questions\*\*\s*\n- (.*?)\n(.*?)\n", sections["assumptions"])

    return sections


def generate_dfd_with_trust_boundaries(dfd_mermaid, threat_model, api_key):
    """
    Enhances the existing DFD Mermaid syntax by adding trust boundaries based on Threat Modeling insights.

    Parameters:
    - dfd_mermaid (str): The original Mermaid syntax of the DFD.
    - threat_model (str): The threat modeling response containing trust boundary information.
    - api_key (str): API key for LLM processing.

    Returns:
    - str: Updated Mermaid syntax with trust boundaries.
    """

    trust_boundary_prompt = f"""
    Enhance the **Data Flow Diagram (DFD)** in **Mermaid syntax** by incorporating **trust boundaries** based on the Threat Model insights.

    ## **Existing DFD**
    ```mermaid
    {dfd_mermaid}
    ```

    ## **Threat Model Insights**
    {threat_model}

    ## **Instructions**
    - Identify security perimeters (trust boundaries) from the threat model.
    - Use `subgraph "Trust Boundary: <name>"` to enclose related components.
    - Clearly indicate which components fall within each boundary.
    - Preserve the original data flows and elements.
    - Return only **Mermaid syntax** enclosed in triple backticks (` ``` `).

    **Example Output:**
    ```mermaid
    graph TD;
        subgraph "Trust Boundary: User Zone"
            A[User] -->|Request| B[Web Server]
        end
        subgraph "Trust Boundary: Internal Network"
            B -->|Processes| C[Application Layer]
            C -->|Stores Data| D[Database]
        end
    ```
    """

    # Call LLM to generate updated DFD
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    dfd_with_trust_boundaries = llm.invoke(trust_boundary_prompt)

    # Extract Mermaid code from response
    if isinstance(dfd_with_trust_boundaries, AIMessage):
        dfd_with_trust_boundaries = dfd_with_trust_boundaries.content

    return extract_mermaid_code(dfd_with_trust_boundaries)


# Initialize Streamlit app
st.set_page_config(page_title="ArgusGPT", layout="wide")
st.title("ArgusGPT - Threat Modeling & Attack Trees")

# Initialize session state variables
if "app_state" not in st.session_state:
    st.session_state.app_state = {
        "threat_model_output": None,
        "attack_tree_output": None,
        "document_context": None,
        "dfd_mermaid_output": None,
        "dfd_trust_boundary": None,
        "threat_model_generated": False,
        "attack_tree_generated": False,
        "dfd_mermaid_generated": False,
        "dfd_trust_boundary_generated": False,
    }

app_state = st.session_state.app_state

# Sidebar description
st.sidebar.title("ArgusGPT")
st.sidebar.markdown("""
**Usage Instructions:**
1. Select application details.
2. Upload a PRD document (PDF).
3. DFD will be generated automatically.
4. Threat Model will be generated automatically.
5. Click **Create Attack Tree** to visualize attack paths.
""")

st.header("Threat Modeling with Gemini-2.0-Flash and RAG")

# User inputs
app_type = st.selectbox("Application Type", ["Web", "Mobile", "API", "Desktop"], index=0)
auth_method = st.selectbox("Authentication Method", ["OAuth", "JWT", "Session-based", "API Key", "Basic"], index=0)
platform = st.text_input("Describe the Application", "")
is_internet_facing = st.radio("Is the application internet-facing?", ["Yes", "No"], index=0)

# Function to extract text from image using Tesseract OCR
def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    extracted_text = pytesseract.image_to_string(gray)
    return extracted_text.strip()

# File uploader
import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import io

st.title("📄 Upload PRD (PDF) or DFD (Image)")

# Radio button to select upload type
upload_type = st.radio("Select Upload Type", ["PRD (PDF)", "DFD (Image)"])

uploaded_file = st.file_uploader(
    f"Upload your {upload_type}", 
    type=["pdf"] if upload_type == "PRD (PDF)" else ["png", "jpg", "jpeg"]
)

if uploaded_file:
    if upload_type == "PRD (PDF)":
        # Extract text from PRD PDF
        pdf_reader = PdfReader(uploaded_file)
        prd_text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

        # Store PRD content in session_state
        st.session_state["prd_content"] = prd_text
        st.session_state["dfd_image"] = None  # Clear DFD image if PRD is uploaded

        st.success("✅ PRD uploaded and processed successfully!")

        # Show extracted PRD content
        with st.expander("🔍 View Extracted PRD Content"):
            st.write(prd_text)

    elif upload_type == "DFD (Image)":
        # Load and display the DFD image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded DFD", use_column_width=True)

        # Extract text from DFD using OCR
        dfd_text = pytesseract.image_to_string(image)

        # Store the image and extracted text in session_state
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=image.format)
        st.session_state["dfd_image"] = img_bytes.getvalue()
        st.session_state["dfd_text"] = dfd_text
        st.session_state["prd_content"] = None  # Clear PRD content if DFD is uploaded

        st.success("✅ DFD image uploaded and processed successfully!")

        # Show extracted DFD text (for debugging or further processing)
        with st.expander("🔍 View Extracted DFD Content"):
            st.write(dfd_text)


# Main content area
main_container = st.container()

with main_container:
    if uploaded_file:
        # Process document
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        loader = PDFPlumberLoader("temp.pdf")
        docs = loader.load()
        app_state["document_context"] = "\n".join([doc.page_content for doc in docs])
        # Store PRD content in session_state
        st.session_state["prd_content"] = app_state["document_context"]
        
        # Auto-generate DFD
        if not app_state["dfd_mermaid_generated"]:
            with st.spinner("Generating DFD Diagram..."):
                try:
                    dfd_mermaid = generate_dfd_mermaid(app_state["document_context"], api_key)
                    app_state["dfd_mermaid_output"] = dfd_mermaid
                    st.session_state["dfd_mermaid"] = dfd_mermaid
                    app_state["dfd_mermaid_generated"] = True
                except Exception as e:
                    st.error(f"Error generating DFD: {str(e)}")
            st.rerun()
        
        # Display DFD before threat model
        if app_state["dfd_mermaid_output"]:
            st.markdown("### Data Flow Diagram (DFD)")
            render_dfd_diagram(app_state["dfd_mermaid_output"])
            # Save Mermaid code and allow download
            file_path = save_mermaid_to_file(app_state["dfd_mermaid_output"])
            print(file_path)
            
        # Auto-generate threat model
        if not app_state["threat_model_generated"]:
            with st.spinner("Generating Threat Model..."):
                try:
                    threat_model = generate_threat_model(docs,app_state["dfd_mermaid_output"], app_type, auth_method, platform, is_internet_facing,api_key)
                    app_state["threat_model_output"] = threat_model
                    st.session_state["threat_model"] = threat_model
                    app_state["threat_model_generated"] = True
                except Exception as e:
                    st.error(f"Error generating threat model: {str(e)}")
            st.rerun()
        
        # Display threat model output
        if app_state["threat_model_output"]:
            filtered_output, think_contents = filter_think_tags(app_state["threat_model_output"])
            
            st.markdown("### Threat Model Response")
            # st.markdown(filtered_output)
            sections = extract_threat_model_sections(app_state["threat_model_output"])

            # Display the Asset Threat Model Table
            if sections["table"] is not None:
                # print(sections["table"])
                # print("HELLO   ******************************\n")
                st.markdown("### Asset Threat Model Table")
                st.table(sections["table"])

            # Display Trust Boundaries
            if sections["trust_boundaries"]:
                st.markdown("### Trust Boundaries")
                st.markdown("\n".join(sections["trust_boundaries"]))

            # Display Data Flows
            if sections["data_flows"]:
                st.markdown("### Data Flows")
                st.markdown("\n".join(sections["data_flows"]))

            # Display Deployment Threat Model
            if sections["deployment_threat_model"]:
                st.markdown("### Deployment Threat Model")
                st.markdown("\n".join(sections["deployment_threat_model"]))

            # Display Build Threat Model
            if sections["build_threat_model"]:
                st.markdown("### Build Threat Model")
                st.markdown("\n".join(sections["build_threat_model"]))

            # Display Assumptions & Open Questions
            if sections["assumptions"]:
                st.markdown("### Assumptions & Open Questions")
                st.markdown("\n".join(sections["assumptions"]))

            if think_contents:
                with st.expander("Reasoning (LLM Insights)"):
                    for think in think_contents:
                        st.markdown(f"- {think}")
            
            st.download_button(
                label="Download Threat Model Report",
                data=filtered_output,
                file_name="threat_model.md",
                mime="text/markdown"
            )
            
            if app_state["threat_model_output"]:
                dfd_trust_boundary = generate_dfd_with_trust_boundaries(app_state["dfd_mermaid_output"], app_state["threat_model_output"], api_key)
                st.markdown("### Data Flow Diagram with Trust Boundaries")
                app_state["dfd_trust_boundary"] = dfd_trust_boundary
                app_state["dfd_trust_boundary_generated"] = True
                st.session_state["dfd_trust_boundary"] = dfd_trust_boundary
                render_dfd_diagram(dfd_trust_boundary)
                


            # Attack Tree Generation
            if st.button("Create Attack Tree") and not app_state["attack_tree_generated"]:
                with st.spinner("Creating Attack Tree..."):
                    try:
                        attack_tree_response = generate_attack_tree(
                            app_state["threat_model_output"],
                            app_state["document_context"]
                        )
                        filtered_attack_tree, tree_think_contents = filter_think_tags(attack_tree_response)
                        mermaid_code = extract_mermaid_code(filtered_attack_tree)
                        app_state["attack_tree_output"] = mermaid_code
                        app_state["attack_tree_generated"] = True
                    
                    except Exception as e:
                        st.error(f"Error generating attack tree: {str(e)}")
                st.rerun()
            
            # Display attack tree if available
            if app_state["attack_tree_output"]:
                st.markdown("### Attack Tree (Mermaid Syntax)")
                st.code(f"""```mermaid\n{app_state["attack_tree_output"]}\n```""", language="mermaid")

                
            #Chatbot
            st.set_page_config(page_title="Threat Modeling", layout="wide")

            st.title("Threat Modeling Tool")
            st.write("Upload your PRD and generate a Threat Model with DFD.")

            # File upload
            uploaded_file = st.file_uploader("Upload PRD", type=["pdf", "txt"])

            if uploaded_file:
                prd_content = uploaded_file.read().decode("utf-8")
                st.session_state["prd_content"] = prd_content  # Store for later use
                st.success("PRD uploaded successfully!")

            # Threat Model Button
            if st.button("Generate Threat Model"):
                st.write("Generating Threat Model...")  # Call your threat model function here
