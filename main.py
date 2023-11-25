import json
import openai
import requests
import spacy


openai_api_key = "enter your key here"

local_mode = False
debug_mode = False


def fetch_data(wiki_title):
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": wiki_title,
            "prop": "extracts",
            # 'exintro': True,
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    return wiki_text


# LLM Call abstraction
openai_cache = {}
local_cache = {}


def llm_call(messages, functions=None, local=False):
    local &= local_mode  # Override local parameter if OpenAI mode is enabled
    llm_type = "Local LLM" if local else "OpenAI LLM"
    cache = local_cache if local else openai_cache
    if debug_mode:
        if functions is not None:
            print(f"Calling {llm_type} with messages: {messages}\nand {len(functions)} functions\n")
        else:
            print(f"Calling {llm_type} with messages: {messages}\n")

    message_content = tuple([message.get("content") for message in messages])
    if message_content in cache:
        if debug_mode:
            print("Using cached response")
        return cache.get(tuple([message.get("content") for message in messages]))

    openai.api_base = "http://localhost:8080/v1" if local else "https://api.openai.com/v1"
    openai.api_key = "fake-key" if local else openai_api_key

    params = {
        "model": "wizard" if local else "gpt-3.5-turbo",
        "messages": messages,
        "temperature": 0
    }
    if functions is not None:
        params["functions"] = functions

    result = openai.ChatCompletion.create(**params)

    cache[message_content] = result
    return result


# Create topic heirarchy
class Topic:
    def __init__(self, name, description, text, level, subtopics):
        self.name = name.replace(' ', '_').replace('–', '_').replace(',', '_')
        self.description = description
        self.text = text
        self.level = level
        self.subtopics = subtopics

    def __str__(self):
        return f"Topic(name={self.name}, description={self.description}, level={self.level}, subtopics={self.subtopics})"

    def __repr__(self):
        return str(self)

    def print_tree(self, indent=0):
        print("  " * indent + f"{self.name}")
        for subtopic in self.subtopics:
            subtopic.print_tree(indent + 1)


def to_openai_function(topic):
    return {
        "name": f"tool_{topic.name}",
        "description": f"This content contains information about {topic.description}. "
                       f"Use this tool if you want to answer any questions about {topic.description}.",
        "parameters": {
            "type": "object",
            "required": ["attribute"],
            "properties": {
                "attribute": {
                    "type": "string",
                    "description": f"Attribute to query about {topic.description}",
                }
            },
        },
    }


def create_topic(name, text, level, description_suffix=""):
    heading_prefix = "=" * (level + 2)
    next_heading_prefix = "=" * (level + 3)

    subtopics = []
    if heading_prefix in text:
        subtopic_name = "General Information"
        subtopic_text = ""
        for line in text.splitlines():
            if line.startswith(heading_prefix) and not line.startswith(next_heading_prefix):
                if subtopic_text:
                    subtopics.append(
                        create_topic(subtopic_name, subtopic_text, level + 1, " of " + name + description_suffix))
                subtopic_name = line[level + 3:-level - 3]
                subtopic_text = ""
            elif line:
                subtopic_text += line + "\n"
        if subtopic_text:
            subtopics.append(create_topic(subtopic_name, subtopic_text, level + 1))

    return Topic(
        name=name,
        description=name + description_suffix,
        text=text,
        level=level,
        subtopics=subtopics
    )


def create_topics(wiki_titles):
    topics = []
    for wiki_title in wiki_titles:
        wiki_text = fetch_data(wiki_title)
        topics.append(create_topic(wiki_title, wiki_text, 0))
    return topics


def llm_call_with_function_resolution(function_arguments, topic):
    messages = [
        {"role": "system",
         "content": f"You are an agent designed to answer queries about {topic.description}.\n"
                    "Please always use the tools provided to answer a question. Do not rely on prior knowledge."},
        {"role": "user", "content": f"Describe the {function_arguments} of {topic.description}"}
    ]
    functions = [to_openai_function(topic) for topic in topic.subtopics]
    response = llm_call(messages, functions)
    if not response.choices[0].finish_reason == "function_call":
        return response

    function_name = response.choices[0].message.function_call.name
    function_arguments = json.loads(response.choices[0].message.function_call.arguments)["attribute"].replace("_", " ")
    print(f"Calling level {topic.level + 1} function '{function_name[len('tool_'):]}' with arguments '{function_arguments}'")

    query_topics = topic.subtopics
    if function_name[len("tool_"):] not in [topic.name for topic in query_topics]:
        print("LLM requested a function that does not exist.")
        print("Requested function: ", function_name[len("tool_"):])
        print("Available functions: ", [topic.name for topic in query_topics])
        raise Exception("LLM requested a function that does not exist.")
    current_topic = next(topic for topic in query_topics if topic.name == function_name[len("tool_"):])

    if function_arguments.lower() == "description":
        return current_topic.text

    if current_topic.subtopics:
        return llm_call_with_function_resolution(function_arguments, current_topic)
    else:
        function_response = llm_call([
            {"role": "system",
             "content": f"You are an expert Q&A system that is trusted around the world.\n"
                        f"Always answer the query using the provided context information, and not prior knowledge.\n"
                        f"Some rules to follow:\n"
                        f"1. Never directly reference the given context in your answer.\n"
                        f"2. Avoid statements like 'Based on the context, ...' or 'The context information ...' "
                        f"or anything along those lines."},
            {"role": "user",
             "content": f"Context information is below.\n"
                        f"---------------------\n"
                        f"{current_topic.text}\n"
                        f"---------------------\n"
                        f"Given the context information and not prior knowledge, answer the query.\n"
                        f"Query: {function_arguments}\n"
                        f"Answer: "}
        ], local=True)
        return function_response.choices[0].message.content


def answer_question(question):
    subjects = extract_subjects(question)
    print(f"Extracted subjects: {subjects}")
    print("Creating topic heirarchy...")
    top_level_topics = create_topics(subjects)
    if debug_mode:
        for topic in top_level_topics:
            topic.print_tree()
    query_topics = top_level_topics
    functions = [to_openai_function(topic) for topic in query_topics]
    message_history = [
        {"role": "system",
         "content": "You are an agent designed to answer queries about a set of given cities.\n"
                    "Please always use the tools provided to answer a question. Do not rely on prior knowledge."},
        {"role": "user", "content": question}
    ]
    response = llm_call(message_history, functions)

    while response.choices[0].finish_reason == "function_call":
        function_name = response.choices[0].message.function_call.name
        function_arguments = json.loads(response.choices[0].message.function_call.arguments)["attribute"].replace("_", " ")
        print(f"\nCalling level 0 function '{function_name[len('tool_'):]}' with arguments '{function_arguments}'")
        query_topics = top_level_topics
        current_topic = next(topic for topic in query_topics if topic.name == function_name[len("tool_"):])

        function_response = llm_call_with_function_resolution(function_arguments, current_topic)

        message_history.append(response.choices[0].message)
        message_history.append({
            "role": "function",
            "name": function_name,
            "content": function_response
        })

        response = llm_call(message_history, functions)

    return response.choices[0].message.content


# Subject Extraction
def extract_subjects(query):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)
    return [entity.text for entity in doc.ents if entity.root.pos_ in ["NOUN", "PROPN"]]


if local_mode:
    print("Running in Local mode.")
    print("Note that all LLM calls with functions will still be redirected to OpenAI. (<$0.01 per query)")
    print("Checking if local LLM works...")
    llm_call([{"role": "user", "content": "What is the answer to life?"}], local=True)
    print("Checking if cache works...")
    llm_call([{"role": "user", "content": "What is the answer to life?"}], local=True)
else:
    print("Running in OpenAI mode.")
    print("Note that all LLM calls will be charged to the OpenAI API key. (<$0.01 per query)")

question = ""
while True:
    previous_question = question
    question = str(input("Question (enter 'exit' to exit): "))
    if question.lower() == "exit":
        break
    if question.lower() == "debug":
        debug_mode = not debug_mode
        print(f"Debug mode set to {debug_mode}")
        if not debug_mode or previous_question == "":
            continue
        question = previous_question
    try:
        response = answer_question(question)
        print("\n\nAnswer: ", response + "\n\n")
    except Exception as e:
        if debug_mode:
            raise e
        print(e)
        print("Error occurred while answering question. Please try again.\n\n")