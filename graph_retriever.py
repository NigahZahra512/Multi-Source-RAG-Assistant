"""
DAY 3 (Part B): GRAPH-BASED RETRIEVER
-----------------------------------------
Idea: Document mein jo important "entities" hain (naam, org, jagah waghera)
unka ek graph banate hain. Agar do entities ek hi sentence mein mojood
hain, hum maan lete hain wo "related" hain aur unke beech ek edge (connection)
bana dete hain.

Jab user kisi entity ke baare mein pooche, hum us entity ko graph mein
dhoondte hain aur uske SEEDHE connections (1-hop neighbours) return karte
hain - ye "relationship" type sawalon mein vector search se behtar hota hai.

Hum yahan spaCy library use kar rahe hain Named Entity Recognition (NER)
ke liye - ye khud pehchan leti hai ke text mein konsa word ek "entity"
hai (jaise "Google", "John Smith", "Pakistan").
"""

import pickle
import networkx as nx
import spacy
from langchain_core.documents import Document

GRAPH_SAVE_PATH = "data/knowledge_graph.pkl"

# spaCy ka chota, fast English model - entities (naam/org/jagah) pehchanne ke liye
nlp = spacy.load("en_core_web_sm")


def extract_entities(sentence: str) -> list[str]:
    """Ek sentence mein se named entities nikalta hai (PERSON, ORG, GPE, etc.)"""
    doc = nlp(sentence)
    return list({ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 2})


def build_knowledge_graph(documents: list[Document], save_path: str = GRAPH_SAVE_PATH) -> nx.Graph:
    """
    Har document ke sentences parhta hai, entities nikalta hai, aur jo
    entities ek hi sentence mein sath aayen unke beech edge bana deta hai.
    Edge ke "context" attribute mein hum wo sentence bhi save karte hain -
    taake baad mein LLM ko actual text mil sake, sirf entity names nahi.
    """
    graph = nx.Graph()

    for doc in documents:
        # Simple sentence split (sentence_window_retriever waali approach)
        sentences = [s.strip() for s in doc.page_content.split(".") if s.strip()]

        for sentence in sentences:
            entities = extract_entities(sentence)

            for entity in entities:
                if not graph.has_node(entity):
                    graph.add_node(entity)

            # Har pair of entities ke beech connection banao
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    e1, e2 = entities[i], entities[j]
                    if graph.has_edge(e1, e2):
                        graph[e1][e2]["contexts"].append(sentence)
                    else:
                        graph.add_edge(e1, e2, contexts=[sentence])

    with open(save_path, "wb") as f:
        pickle.dump(graph, f)

    print(f"[graph] {graph.number_of_nodes()} entities, {graph.number_of_edges()} relationships -> saved to '{save_path}'")
    return graph


def load_knowledge_graph(save_path: str = GRAPH_SAVE_PATH) -> nx.Graph:
    with open(save_path, "rb") as f:
        return pickle.load(f)


def retrieve_from_graph(query: str, max_contexts: int = 4) -> list[str]:
    """
    Query mein se entities nikalta hai, phir graph mein dhoond kar unke
    1-hop neighbours ke contexts (sentences) return karta hai.
    """
    graph = load_knowledge_graph()
    query_entities = extract_entities(query)

    contexts = []
    for entity in query_entities:
        # Exact match na mile to case-insensitive partial match try karo
        matched_node = next(
            (n for n in graph.nodes if entity.lower() in n.lower() or n.lower() in entity.lower()),
            None
        )
        if matched_node is None:
            continue

        for neighbor in graph.neighbors(matched_node):
            edge_data = graph.get_edge_data(matched_node, neighbor)
            contexts.extend(edge_data["contexts"])

    # Duplicates hata kar top results wapis bhejo
    unique_contexts = list(dict.fromkeys(contexts))
    return unique_contexts[:max_contexts]