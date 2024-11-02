def convert_to_preference_format(dataset):
    json_format = [
        {
            "chosen_conversations": [
                {"content": row["prompt"], "role": "user"},
                {"content": row["chosen"], "role": "assistant"}
            ],
            "rejected_conversations": [
                {"content": row["prompt"], "role": "user"},
                {"content": row["rejected"], "role": "assistant"}
            ]
        }
        for row in dataset
    ]
    return json_format