import os
from rapidfuzz import process, fuzz
import streamlit as st
from streamlit.logger import get_logger
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
from utils import (
    extract_title_and_question,
    create_vector_index,
)
from chains import (
    load_embedding_model,
    load_llm,
    configure_llm_only_chain,
    configure_qa_rag_chain,
    configure_qa_structure_rag_chain,
)


# >>>> initialise - environemnt <<<< 



# Neo4j

neo4j_method = os.environ.get("NEO4J_METHOD") # get the Neo4j connection method. local or aura
print("Neo4J connect method: ", neo4j_method)
print(neo4j_method == "aura")

if "url" not in st.session_state:
    st.session_state["url"]=""
    if neo4j_method == "local":
        st.session_state["url"] = os.environ.get("NEO4J_LOCAL_URI")
        st.session_state["username"] = os.environ.get("NEO4J_LOCAL_USERNAME")
        st.session_state["password"] = os.environ.get("NEO4J_LOCAL_PASSWORD")
        st.session_state["database"] = os.environ.get("NEO4J_LOCAL_DATABASE")
    else:
        st.session_state["url"] = os.environ.get("NEO4J_AURA_URI")
        st.session_state["username"] = os.environ.get("NEO4J_AURA_USERNAME")
        st.session_state["password"] = os.environ.get("NEO4J_AURA_PASSWORD")
        st.session_state["database"] = os.environ.get("NEO4J_AURA_DATABASE")
print("url: ", st.session_state["url"])
url = st.session_state["url"]
username = st.session_state["username"]
password = st.session_state["password"]
database = st.session_state["database"]

embedding_model_name = os.environ.get("EMBEDDING_MODEL")
print("Here embedding model: ", embedding_model_name)
llm_name = os.environ.get("LLM")
# Remapping for Langchain Neo4j integration
# os.environ["NEO4J_URL"] = url


# >>>> initialise - services <<<< 

logger = get_logger(__name__)

neo4j_graph = Neo4jGraph(url=url, username=username, password=password, database=database)

embeddings, dimension = load_embedding_model(
    embedding_model_name, logger=logger
)

llm = load_llm(llm_name, logger=logger)
# >>>>>>initialise address and by law name <<<<
if "address" not in st.session_state:
        st.session_state[f"address"] = []
if "by-law-name" not in st.session_state:
        st.session_state[f"by-law-name"] = []
# llm_chain: LLM only response
llm_chain = configure_llm_only_chain(llm)
#document_url = './documents/sp_94898_strata_by-laws.pdf'
# rag_chain: KG augmented response
#document_url = './documents/sp_94898_strata_by-laws.pdf'
if st.session_state['by-law-name']:
    document_url = st.session_state['by-law-name'][0]
else:
    document_url =''
rag_chain = configure_qa_structure_rag_chain(
    llm, embeddings, embeddings_store_url=url, username=username, password=password, document_url=document_url
)

# SKIPPED: create_vector_index(neo4j_graph, dimension)

# >>>> Class definition - StreamHander <<<< 

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# >>>> Streamlit UI <<<<

styl = f"""
<style>
    /* not great support for :has yet (hello FireFox), but using it for now */
    .element-container:has([aria-label="Choose RAG mode"]) {{
      position: fixed;
      bottom: 33px;
      background: white;
      z-index: 101;
    }}
    .stChatFloatingInputContainer {{
        bottom: 20px;
    }}

    /* Generate question text area */
    textarea[aria-label="Description"] {{
        height: 200px;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)
col1, col2 = st.columns([2,10], vertical_alignment="bottom")
with col1:
    st.image("images/qna-logo.png", width=100) 
with col2:
    st.title("Strata Chatbot")
# >>>> UI interations <<<<
# Function to retrieve the bylaw file based on the address input
def get_property_bylaw(address, address_map):
    # Extract the list of addresses from the map
    address_keys = list(address_map.keys())
    
    # Use fuzzy matching to find the best match for the user-provided address
    match = process.extractOne(address, address_keys, scorer=fuzz.partial_ratio)
    
    # Unpack the result into best match and score
    best_match, score = match[0], match[1]

    # If a good match is found (score > 80), retrieve the bylaw file
    if score > 80:
        matched_file = address_map[best_match]
        return True, best_match, matched_file
    else:
        return False, None, "Sorry, we couldn't find a close match for that address."

# the address file name map
address_map = {
    "2-10 River Road West, Parramatta 2150": "./documents/sp_94898_strata_by-laws.pdf",
    "12-16 East St, Granville NSW": "./documents/sp_99516_strata_by-laws.pdf",
    "456 Oak Avenue, Mascot": "./documents/sp_100991_strata_by-laws.pdf",
    # Add more mappings here as needed
}
    
def input_address():
    address = st.chat_input("Property address?")
    # Retrieve the bylaw information for the given address
    if address:
        with st.chat_message("user"):
            st.write(address)
        with st.chat_message("assistant"):
            find_result, matched_address, return_text = get_property_bylaw(address, address_map)
            print(f"find result: {find_result}, matched address: {matched_address}, return text {return_text}")
            if find_result:
                display_info = f'We found the by-law for: {matched_address}, the name is {return_text}'
                st.session_state[f'by-law-name'].append(return_text)
                st.session_state[f'address'] = matched_address
                st.rerun()
            else:
                display_info = f"We can't find your address {address}, Please enter the property address again"
                st.session_state[f'failed_address'].append(display_info)
            st.write(display_info)
        st.session_state[f"user_input_address" ].append(address)
        

def display_address():
     # Session state
    if "failed_address" not in st.session_state:
        st.session_state[f"failed_address" ] = []
    if "user_input_address" not in st.session_state:
        st.session_state[f"user_input_address" ] = []
    print('failed_address', st.session_state['failed_address'])
    print('user_input_address', st.session_state['user_input_address'])
    if st.session_state[f"failed_address"]:
        size = len(st.session_state[f"failed_address"])
        # Display only the last three exchanges
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input_address"][i])

            with st.chat_message("assistant"):
                st.write(st.session_state[f"failed_address"][i])
        with st.container():
            st.write("&nbsp;")

def chat_input():
    user_input = st.chat_input("What service questions can I help you resolve today?")
    if user_input:      
        with st.chat_message("user"):
            address  = user_input
            st.write(address)
        with st.chat_message("assistant"):
            st.caption(f"RAG: {name}")
            
            stream_handler = StreamHandler(st.empty())

            # Call chain to generate answers
            original_result = output_function(
                {"question": user_input, "chat_history": []}, callbacks=[stream_handler]
            )
            result = original_result["answer"]
            print("DEBUG---output_function")
            print("original result: ", original_result)
            print("result: ", result)

            output = result

            st.session_state[f"user_input"].append(user_input)
            st.session_state[f"generated"].append(output)
            st.session_state[f"rag_mode"].append(name)


def display_chat():
    # Session state
    if "generated" not in st.session_state:
        st.session_state[f"generated"] = []

    if "user_input" not in st.session_state:
        st.session_state[f"user_input"] = []

    if "rag_mode" not in st.session_state:
        st.session_state[f"rag_mode"] = []
    print('user_input', st.session_state['user_input'])
    print('generated', st.session_state['generated'])
    if st.session_state[f"generated"]:
        size = len(st.session_state[f"generated"])
        # Display only the last three exchanges
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input"][i])

            with st.chat_message("assistant"):
                st.caption(f"RAG: {st.session_state[f'rag_mode'][i]}")
                st.write(st.session_state[f"generated"][i])

        with st.expander("Not finding what you're looking for?"):
            st.write(
                "Automatically generate a draft for an internal ticket to our support team."
            )
            st.button(
                "Generate ticket",
                type="primary",
                key="show_ticket",
                on_click=open_sidebar,
            )
        with st.container():
            st.write("&nbsp;")


def mode_select() -> str:
    options = ["Disabled", "Enabled"]
    return st.radio("Select RAG mode", options, horizontal=True)

# >>>>> switch on/off RAG mode


def generate_ticket():
    # Get high ranked questions
    records = neo4j_graph.query(
        "MATCH (q:Question) RETURN q.title AS title, q.body AS body ORDER BY q.score DESC LIMIT 3"
    )
    questions = []
    for i, question in enumerate(records, start=1):
        questions.append((question["title"], question["body"]))
    # Ask LLM to generate new question in the same style
    questions_prompt = ""
    for i, question in enumerate(questions, start=1):
        questions_prompt += f"{i}. {question[0]}\n"
        questions_prompt += f"{question[1]}\n\n"
        questions_prompt += "----\n\n"

    gen_system_template = f"""
    You're an expert in formulating high quality questions. 
    Can you formulate a question in the same style, detail and tone as the following example questions?
    {questions_prompt}
    ---

    Don't make anything up, only use information in the following question.
    Return a title for the question, and the question post itself.

    Return example:
    ---
    Title: How do I use the Neo4j Python driver?
    Question: I'm trying to connect to Neo4j using the Python driver, but I'm getting an error.
    ---
    """
    # we need jinja2 since the questions themselves contain curly braces
    system_prompt = SystemMessagePromptTemplate.from_template(
        gen_system_template, template_format="jinja2"
    )
    q_prompt = st.session_state[f"user_input"][-1]
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            SystemMessagePromptTemplate.from_template(
                """
                Respond in the following format or you will be unplugged.
                ---
                Title: New title
                Question: New question
                ---
                """
            ),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )
    llm_response = llm_chain(
        f"Here's the question to rewrite in the expected format: ```{q_prompt}```",
        [],
        chat_prompt,
    )
    new_title, new_question = extract_title_and_question(llm_response["answer"])
    return (new_title, new_question)


def open_sidebar():
    st.session_state.open_sidebar = True


def close_sidebar():
    st.session_state.open_sidebar = False


if not "open_sidebar" in st.session_state:
    st.session_state.open_sidebar = False
if st.session_state.open_sidebar:
    new_title, new_question = generate_ticket()
    with st.sidebar:
        st.title("Ticket draft")
        st.write("Auto generated draft ticket")
        st.text_input("Title", new_title)
        st.text_area("Description", new_question)
        st.button(
            "Submit to support team",
            type="primary",
            key="submit_ticket",
            on_click=close_sidebar,
        )

# >>>> UI: show chat <<<<
#st.chat_input("What service questions can I help you resolve today?")
if not st.session_state[f'address']:
    st.subheader("Please input your property address.")
    display_address()
    input_address()
else:
    st.subheader(f"We got your property address as {st.session_state['address']}")
    name = mode_select()
    st.write("Please input your questions regarding your property")
    if name == "LLM only" or name == "Disabled":
        output_function = llm_chain
    elif name == "Vector + Graph" or name == "Enabled":
        output_function = rag_chain
    display_chat()
    chat_input()


