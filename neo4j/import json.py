import json
import pandas as pd

filename = 'review_node2vec_5_extra_sentiment_features.json'

# Read and load json file to extract the data from it
with open(filename, 'r', encoding='utf-8-sig') as file:
    content = file.read()
    print("File content:", content)  # Debug step
    data = json.loads(content)

# If the file contains a list of entries, extract the entries and corresponding properties
rows = []
for entry in data:
    props = entry["n"]["properties"]

    flat_data = {
        "id": props["id"],
    "rating": props.get("rating", 0),
    "avg_sentiment": props.get("avg_sentiment", 0),
    "min_sentiment": props.get("min_sentiment", 0),
    "max_sentiment": props.get("max_sentiment", 0),
    }

    # Extract the node2vec values into separate features
    for idx, val in enumerate(props["node2Vec"]):
        flat_data[f"node2vec_{idx}"] = val

    rows.append(flat_data)

# Create a dataframe from the json data
df = pd.DataFrame(rows)
df.to_csv("review_node2vec_5_extra_sentiment_features.csv", index=False)
df = pd.DataFrame(rows)
df.to_csv("review_node2vec_sentiment.csv", index=False)