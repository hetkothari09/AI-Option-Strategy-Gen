Store various strategy's embedded form in the database.
When user enters his prompt indicating a strategy name, then store that in embedded form too.
Check for the user inputted strategy name if present in the database.
Store both of the embedded data in numpy array and convert them into torch format.
Torch format is needed to compare and perform cosine similarity between the embeddings.
Select the output with similarity score more than 0.5.
Return the config stored in the database (if the data stored in database is not in config format then explicitly
format it).
Use AI (RAG) to answer the user's questions on the config file by passing the user's message and config to AI.


Use AI to generate the config if not present.
Also to structure the config file if cannot be done other way.

