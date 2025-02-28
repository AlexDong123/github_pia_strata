from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings import OllamaEmbeddings, SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI, ChatOllama

from langchain_aws import ChatBedrockConverse, ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import List, Any
from utils import BaseLogger
from langchain.chains import GraphCypherQAChain 
import os
from dotenv import load_dotenv
import boto3

def load_embedding_model(embedding_model_name: str, logger=BaseLogger()):
    if embedding_model_name == "ollama":
        embeddings = OllamaEmbeddings(
            base_url= os.environ.get("OLLAMA_BASE_URL"), model="nomic-embed-text:latest"
        )
        dimension = 4096
    elif embedding_model_name == "bedrock":
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        embeddings = BedrockEmbeddings(
            client= bedrock_client,
            model_id = 'amazon.titan-embed-text-v2:0'
        )
        dimension = 1024
    elif embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
    else:
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2", cache_folder="/embedding_model"
        )
        dimension = 384
    print("Embedding use: ", embedding_model_name)
    return embeddings, dimension


def load_llm(llm_name: str, logger=BaseLogger()):
    print("LLM: Using", llm_name)
    if llm_name == "gpt-4":
        return ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
    elif llm_name == "gpt-3.5":
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
    elif llm_name == "bedrock":
        return ChatBedrock(
            model_id = os.environ.get("BEDROCK_MODEL_ID"),
            temperature=os.environ.get("LLM_TEMPERATURE"),
            provider=os.environ.get("BEDROCK_MODEL_PROVIDER"),
            region_name=os.environ.get("BEDROCK_REGION_NAME"),
            max_tokens=os.environ.get("BEDROCK_MAX_TOEKNS"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            streaming=True,
            # other params...
            )
    elif llm_name == "ollama":
        return ChatOllama(
            temperature=0,
            base_url=os.environ.get("OLLAMA_BASE_URL"),
            model=os.environ.get("OLLAMA_MODEL_NAME"),
            streaming=True,
            top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
            num_ctx=3072,  # Sets the size of the context window used to generate the next token.
        )
    print("Finally call gpt-3.5")
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)


def configure_llm_only_chain(llm):
    # LLM only response
    template = """
    You are a helpful assistant that helps with answering general questions.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    def generate_llm_output(
        user_input: str, callbacks: List[Any], prompt=chat_prompt
    ) -> str:
        answer = llm(
            prompt.format_prompt(
                text=user_input,
            ).to_messages(),
            callbacks=callbacks,
        ).content
        return {"answer": answer}

    return generate_llm_output


def configure_qa_rag_chain(llm, embeddings, embeddings_store_url, username, password):
    # RAG response
    general_system_template = """ 
    Use the following pieces of context to answer the question at the end.
    The context contains question-answer pairs and their links from Stackoverflow.
    You should prefer information from accepted or more upvoted answers.
    Make sure to rely on information from the answers and not on questions to provide accuate responses.
    When you find particular answer in the context useful, make sure to cite it in the answer using the link.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----
    {summaries}
    ----
    Each answer you generate should contain a section at the end of links to 
    Stackoverflow questions and answers you found useful, which are described under Source value.
    You can only use links to StackOverflow questions that are present in the context and always
    add links to the end of the answer in the style of citations.
    Generate concise answers with references sources section of links to 
    relevant StackOverflow questions only at the end of the answer.
    """
    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    # Vector + Knowledge Graph response
    kg = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=embeddings_store_url,
        username=username,
        password=password,
        database='neo4j',  # neo4j by default
        index_name="stackoverflow",  # vector by default
        text_node_property="body",  # text by default
        retrieval_query="""
    WITH node AS question, score AS similarity
    CALL  { with question
        MATCH (question)<-[:ANSWERS]-(answer)
        WITH answer
        ORDER BY answer.is_accepted DESC, answer.score DESC
        WITH collect(answer)[..2] as answers
        RETURN reduce(str='', answer IN answers | str + 
                '\n### Answer (Accepted: '+ answer.is_accepted +
                ' Score: ' + answer.score+ '): '+  answer.body + '\n') as answerTexts
    } 
    RETURN '##Question: ' + question.title + '\n' + question.body + '\n' 
        + answerTexts AS text, similarity as score, {source: question.link} AS metadata
    ORDER BY similarity ASC // so that best answers are the last
    """,
    )

    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 2}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=3375,
    )
    return kg_qa

# ADDED
# >>>> Extended to support vector search over strucutured chunking

def configure_qa_structure_rag_chain(llm, embeddings, embeddings_store_url, username, password, document_url):
    # RAG response based on vector search and retrieval of structured chunks

    general_system_template = """ 
    You are a strata manager agent that helps a tenant with answering questions about his or her property.
    Use the following context to answer the question at the end.
    Make sure not to make any changes to the context if possible when prepare answers so as to provide accuate responses.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----
    {summaries}
    ----
    At the end of each answer you should contain metadata for relevant document in the form of (source, page).
    For example, if context has `metadata`:(source:'docu_url', page:1), you should display ('doc_url',  1).
    """
    general_user_template = "Question:```{question}```"
    
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    print("DEBUG--SUMMARY AND QUESTION:")
    print("messages", messages)
    print("qa_prompt", qa_prompt)
    print("+++++++++++")
    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    # Vector + Knowledge Graph response
    kg = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=embeddings_store_url,
        username=username,
        password=password,
        database='neo4j',  # neo4j by default
        index_name="chunkVectorIndex",  # vector by default
        node_label="Embedding",  # embedding node label
        embedding_node_property="value",  # embedding value property
        text_node_property="sentences",  # text by default
        retrieval_query="""
            WITH node AS answerEmb, score 
            ORDER BY score DESC LIMIT 10
            MATCH (answerEmb) <-[:HAS_EMBEDDING]- (answer) -[:HAS_PARENT*]-> (s:Section)
            WITH s, answer, score
            MATCH (d:Document) <-[*]- (s) <-[:HAS_PARENT*]- (chunk:Chunk)
            WHERE d.url = $document_url
            WITH d, s, answer, chunk, score ORDER BY d.url_hash, s.title, chunk.block_idx ASC
            // 3 - prepare results
            WITH d, s, collect(answer) AS answers, collect(chunk) AS chunks, max(score) AS maxScore
            RETURN {source: d.url, page: chunks[0].page_idx+1, matched_chunk_id: id(answers[0])} AS metadata, 
                reduce(text = "", x IN chunks | text + x.sentences + '.') AS text, maxScore AS score LIMIT 3;
    """,
    )
    # Define the parameters to pass into the query
    params = {
        "document_url": document_url  # Example document URL
    }
    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 25, "params": params}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=7000,      # gpt-4
    )
    print("DEBUG:---- kg_qa:")
    print(kg_qa)
    return kg_qa
