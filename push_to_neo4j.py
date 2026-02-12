from neo4j import GraphDatabase
import pickle

URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "Pk@223082K"

# =========================
# LOAD GRAPH
# =========================
with open("review_graph.pkl", "rb") as f:
    G = pickle.load(f)

print("Nodes in graph:", G.number_of_nodes())
print("Edges in graph:", G.number_of_edges())

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))


def create_graph(tx):

    # =========================
    # CREATE NODES WITH PROPERTIES
    # =========================
    for node, data in G.nodes(data=True):

        label = data.get("type", "Node")

        tx.run(
            f"""
            MERGE (n:{label} {{id: $id}})
            SET n.rating = $rating,
                n.avg_sentiment = $avg_sentiment,
                n.min_sentiment = $min_sentiment,
                n.max_sentiment = $max_sentiment
            """,
            id=str(node),
            rating=data.get("rating", None),
            avg_sentiment=data.get("avg_sentiment", 0.0),
            min_sentiment=data.get("min_sentiment", 0.0),
            max_sentiment=data.get("max_sentiment", 0.0),
        )

    # =========================
    # CREATE RELATIONSHIPS
    # =========================
    for source, target, data in G.edges(data=True):

        relation = data.get("relation", "RELATES_TO")
        relation = relation.replace(" ", "_").upper()

        relation = ''.join(c for c in relation if c.isalnum() or c == '_')

        if relation == "":
            relation = "RELATES_TO"

        tx.run(
            f"""
            MATCH (a {{id: $source}})
            MATCH (b {{id: $target}})
            MERGE (a)-[r:{relation}]->(b)
            SET r.sentiment = $sentiment
            """,
            source=str(source),
            target=str(target),
            sentiment=data.get("sentiment", None)
        )


try:
    with driver.session() as session:
        session.execute_write(create_graph)

    print("Graph pushed successfully!")

except Exception as e:
    print("Error occurred:", e)

finally:
    driver.close()
