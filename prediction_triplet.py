import json

# Load the JSON data from prediction.json
with open("minutes_ground_truth_prediction/response_1_evaluated.json", "r") as file:
    data = json.load(file)["data"]

# Initialize a list to store triplets
triplets = []

# Process each relationship in the JSON
for rel in data["relationships"]:
    triplet = {}
    
    # Mapping relationship details to triplet format
    if rel["type"] == "AFFILIATED_WITH":
        triplet = {
            "person": rel["person"],
            "organization": rel["organization"],
            "relationship": "AFFILIATED_WITH"
        }
    elif rel["type"] == "HAS_ROLE":
        triplet = {
            "person": rel["person"],
            "role": rel["role"],
            "relationship": "HAS_ROLE"
        }
    elif rel["type"] == "INVOLVED_WITH":
        triplet = {
            "person": rel["person"],
            "product_service": rel["product_service"],
            "relationship": "INVOLVED_WITH"
        }
    elif rel["type"] == "LOCATED_AT":
        triplet = {
            "organization": rel["organization"],
            "location": rel["location"],
            "relationship": "LOCATED_AT"
        }

    # Add the triplet to the list
    triplets.append(triplet)

# Output the triplets as a JSON array
with open("minute_1_triplet.json", "w") as output_file:
    json.dump(triplets, output_file, indent=4)

print("Triplet conversion completed. Saved to minute_1_triplet.json.")