from langchain_community.chat_models import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Instantiates the database using local credentials and details.
# The process requires you to install the client, set up a local DBMS connection within the client, and get the credentials afterward.
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, enhanced_schema=True)

# Example prompt of how the exchange of query into cypher would be like.
example_prompt = PromptTemplate.from_template(
    "User input: {question}\nCypher query: {query}"
)

examples = [
    {
        "question": "Find all movies directed by Christopher Nolan.",
        "query": "MATCH (d:Director)-[:DIRECTED]->(m:Movie) WHERE d.name = 'Christopher Nolan' RETURN m.title"
    },
    {
        "question": "List actors who acted in 'Inception'.",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) WHERE m.title = 'Inception' RETURN a.name"
    },
    {
        "question": "What are the genres of the movie 'The Matrix'?",
        "query": "MATCH (m:Movie)-[:IN_GENRE]->(g:Genre) WHERE m.title = 'The Matrix' RETURN g.name"
    }
]

# Few-shot prompting template for examples to enhance RAG responses.
cypher_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Translate the following user inputs into Cypher queries.",
    suffix="User input: {question}\nCypher query:",
    input_variables=["question"]
)

"""
Initializing the OpenAI LLM.

Parameters
----------
:param openai_api_key: Necessary for access the OpenAI API.
:param temperature: Typically set to 0 for most deterministic. The higher it is, the more creative the model gets.
:param model: Mostly variable, but gpt-4o-mini is chosen for balance of cost and performance.
----------
"""
llm = ChatOpenAI(openai_api_key=API_KEY, temperature=0, model_name="gpt-4o-mini")

"""
Initializing the Cypher Question-Answering Chain. This allows direct query of the graph database as it abstracts over the CQL.

Parameters
----------
:param llm: As before, for initializing the LLM model to be used.
:param graph: Specifies the database connection to be used.
:param verbose: Specifies if the query process should be logged in console for view.

Optional:
:param qa_prompt: Prompt template for result generation.
:param cypher_prompt: Prompt template for cypher generation.
:param cypher_llm: LLM for Cypher generation.
:param qa_llm: LLM for result generation.
:param exclude_types: Specify what types of data in the database to be ignored when querying the graphs. Makes queries faster and more relevant.
:param validate_cypher: Detects nodes and relationships, determines the directions of relationship, checks graph schema and updates the direction for them.
----------
"""
chain = GraphCypherQAChain.from_llm(
    llm=llm, graph=graph, verbose=True, validate_cypher=True, cypher_prompt=cypher_prompt
)

# Attach the graph transformer to the LLM for parsing documents into graphs for Neo4J database.
llm_transformer = LLMGraphTransformer(llm=llm)


def add_documents(documents):
    # After setting the LLM and attaching the transformer for parsing documents into graphs, the conversion can begin.
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    """
    Store it as graphs into the Neo4J database.
    
    Parameters
    ----------
    :param include_source: Links nodes to source documents with `MENTIONS` edge.
    :param baseEntityLabel: Adds `__Entity__` label to each node.
    ----------
    """
    graph.add_graph_documents(
        graph_documents,
        include_source=True,
        baseEntityLabel=True
    )

    # Recommended after adding new nodes or data.
    graph.refresh_schema()


def cypher_qa_query_database(query: str):
    result = chain.invoke({f"query": {query}})
    return result


# Not used unless not using Cypher QA Chain.
def query_database(label, property, relationship, node):
    query = f"MATCH ({label} {property})-[:{relationship.upper()}]->({node}) RETURN {node.split(':')[0]}"
    return chain.run(query)
