# Hierarchical-RAG
Hierarchical RAG is a powerful conversational agent that integrates hierarchical topic modeling and the RAG (Retrieval-Augmented Generation) model for interactive, context-aware question answering. Notably, this project distinguishes itself by avoiding the use of vectorization, providing a unique approach to information retrieval. Additionally, it allows users to query any hierarchical data source, expanding the scope of information exploration.

## Usage
1. Download the [spaCy](spacy.io) NLP model for subject extraction: `python -m spacy download en_core_web_sm`
2. Add your OpenAI API Key at the top of the main file
3. Run the script
4. Try out the example prompts below
5. Experiment with different queries and topics to witness the system's ability to handle complex conversations
6. Go crazy! (not too much)

## Example Prompts
* What is the size of the Milky Way galaxy?
* Compare the religious demographics of Atlanta and Toronto.
* How different are the student clubs of Georgia Tech and IIT Bombay?
* Compare the airports in Chicago, Berlin, and Copenhagen.

## Data Sources
Currently, we fetch all data from Wikipedia.
Adding any custom data source requires two APIs:
* fetch: to fetch data about a topic
* parse: to generate a topic hierarchy

More data sources, such as SEC filings, will be integrated in the future.
