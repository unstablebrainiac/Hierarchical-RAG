# # %% [markdown]
# # # Multi-Document Agents
# #
# # In this guide, you learn towards setting up an agent that can effectively answer different types of questions over a larger set of documents.
# #
# # These questions include the following
# #
# # - QA over a specific doc
# # - QA comparing different docs
# # - Summaries over a specific odc
# # - Comparing summaries between different docs
# #
# # We do this with the following architecture:
# #
# # - setup a "document agent" over each Document: each doc agent can do QA/summarization within its doc
# # - setup a top-level agent over this set of document agents. Do tool retrieval and then do CoT over the set of tools to answer a question.
#
# # %% [markdown]
# # ## Setup and Download Data
# #
# # In this section, we'll define imports and then download Wikipedia articles about different cities. Each article is stored separately.
# #
# # We load in 18 cities - this is not quite at the level of "hundreds" of documents but its still large enough to warrant some top-level document retrieval!
#
# # %%
# from llama_index import (
#     VectorStoreIndex,
#     SummaryIndex,
#     SimpleKeywordTableIndex,
#     SimpleDirectoryReader,
#     ServiceContext,
# )
# from llama_index.schema import IndexNode
# from llama_index.tools import QueryEngineTool, ToolMetadata
# from llama_index.llms import OpenAI
#
# # %%
# wiki_titles = [
#     "Toronto",
#     "Seattle",
#     "Chicago",
#     "Boston",
#     "Houston",
#     "Tokyo",
#     "Berlin",
#     "Lisbon",
#     "Paris",
#     "London",
#     "Atlanta",
#     "Munich",
#     "Shanghai",
#     "Beijing",
#     "Copenhagen",
#     "Moscow",
#     "Cairo",
#     "Karachi",
# ]
#
# # %%
# from pathlib import Path
#
# import requests
#
# for title in wiki_titles:
#     response = requests.get(
#         "https://en.wikipedia.org/w/api.php",
#         params={
#             "action": "query",
#             "format": "json",
#             "titles": title,
#             "prop": "extracts",
#             # 'exintro': True,
#             "explaintext": True,
#         },
#     ).json()
#     page = next(iter(response["query"]["pages"].values()))
#     wiki_text = page["extract"]
#
#     data_path = Path("data")
#     if not data_path.exists():
#         Path.mkdir(data_path)
#
#     with open(data_path / f"{title}.txt", "w") as fp:
#         fp.write(wiki_text)
#
# # %%
# # Load all wiki documents
# city_docs = {}
# for wiki_title in wiki_titles:
#     city_docs[wiki_title] = SimpleDirectoryReader(
#         input_files=[f"data/{wiki_title}.txt"]
#     ).load_data()
#
# # %% [markdown]
# # Define LLM + Service Context + Callback Manager
#
# # %%
# llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
# service_context = ServiceContext.from_defaults(llm=llm)
#
# # %% [markdown]
# # ## Building Multi-Document Agents
# #
# # In this section we show you how to construct the multi-document agent. We first build a document agent for each document, and then define the top-level parent agent with an object index.
#
# # %% [markdown]
# # ### Build Document Agent for each Document
# #
# # In this section we define "document agents" for each document.
# #
# # We define both a vector index (for semantic search) and summary index (for summarization) for each document. The two query engines are then converted into tools that are passed to an OpenAI function calling agent.
# #
# # This document agent can dynamically choose to perform semantic search or summarization within a given document.
# #
# # We create a separate document agent for each city.
#
# # %%
# from llama_index.agent import OpenAIAgent
# from llama_index import load_index_from_storage, StorageContext
# from llama_index.node_parser import SimpleNodeParser
# import os
#
# node_parser = SimpleNodeParser.from_defaults()
#
# # Build agents dictionary
# agents = {}
# query_engines = {}
#
# # this is for the baseline
# all_nodes = []
#
# for idx, wiki_title in enumerate(wiki_titles):
#     nodes = node_parser.get_nodes_from_documents(city_docs[wiki_title])
#     all_nodes.extend(nodes)
#
#     if not os.path.exists(f"./data/{wiki_title}"):
#         # build vector index
#         vector_index = VectorStoreIndex(nodes, service_context=service_context)
#         vector_index.storage_context.persist(persist_dir=f"./data/{wiki_title}")
#     else:
#         vector_index = load_index_from_storage(
#             StorageContext.from_defaults(persist_dir=f"./data/{wiki_title}"),
#             service_context=service_context,
#         )
#
#     # build summary index
#     summary_index = SummaryIndex(nodes, service_context=service_context)
#     # define query engines
#     vector_query_engine = vector_index.as_query_engine()
#     summary_query_engine = summary_index.as_query_engine()
#
#     # define tools
#     query_engine_tools = [
#         QueryEngineTool(
#             query_engine=vector_query_engine,
#             metadata=ToolMetadata(
#                 name="vector_tool",
#                 description=f"Useful for questions related to specific aspects of {wiki_title} (e.g. the history, arts and culture, sports, demographics, or more).",
#             ),
#         ),
#         QueryEngineTool(
#             query_engine=summary_query_engine,
#             metadata=ToolMetadata(
#                 name="summary_tool",
#                 description=f"Useful for any requests that require a holistic summary of EVERYTHING about {wiki_title}. For questions about more specific sections, please use the vector_tool.",
#             ),
#         ),
#     ]
#
#     # build agent
#     function_llm = OpenAI(model="gpt-4")
#     agent = OpenAIAgent.from_tools(
#         query_engine_tools,
#         llm=function_llm,
#         verbose=True,
#         system_prompt=f"""\
# You are a specialized agent designed to answer queries about {wiki_title}.
# You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
# """,
#     )
#
#     agents[wiki_title] = agent
#     query_engines[wiki_title] = vector_index.as_query_engine(similarity_top_k=2)
#
# # %% [markdown]
# # ### Build Retriever-Enabled OpenAI Agent
# #
# # We build a top-level agent that can orchestrate across the different document agents to answer any user query.
# #
# # This agent takes in all document agents as tools. This specific agent `RetrieverOpenAIAgent` performs tool retrieval before tool use (unlike a default agent that tries to put all tools in the prompt).
# #
# # Here we use a top-k retriever, but we encourage you to customize the tool retriever method!
# #
#
# # %%
# # define tool for each document agent
# all_tools = []
# for wiki_title in wiki_titles:
#     wiki_summary = (
#         f"This content contains Wikipedia articles about {wiki_title}. "
#         f"Use this tool if you want to answer any questions about {wiki_title}.\n"
#     )
#     doc_tool = QueryEngineTool(
#         query_engine=agents[wiki_title],
#         metadata=ToolMetadata(
#             name=f"tool_{wiki_title}",
#             description=wiki_summary,
#         ),
#     )
#     all_tools.append(doc_tool)
#
# # %%
# # define an "object" index and retriever over these tools
# from llama_index import VectorStoreIndex
# from llama_index.objects import ObjectIndex, SimpleToolNodeMapping
#
# tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
# obj_index = ObjectIndex.from_objects(
#     all_tools,
#     tool_mapping,
#     VectorStoreIndex,
# )
#
# # %%
# from llama_index.agent import FnRetrieverOpenAIAgent
#
# top_agent = FnRetrieverOpenAIAgent.from_retriever(
#     obj_index.as_retriever(similarity_top_k=3),
#     system_prompt=""" \
# You are an agent designed to answer queries about a set of given cities.
# Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
#
# """,
#     verbose=True,
# )
#
# # %% [markdown]
# # ### Define Baseline Vector Store Index
# #
# # As a point of comparison, we define a "naive" RAG pipeline which dumps all docs into a single vector index collection.
# #
# # We set the top_k = 4
#
# # %%
# base_index = VectorStoreIndex(all_nodes)
# base_query_engine = base_index.as_query_engine(similarity_top_k=4)
#
# # %% [markdown]
# # ## Running Example Queries
# #
# # Let's run some example queries, ranging from QA / summaries over a single document to QA / summarization over multiple documents.
#
# # %%
# # should use Boston agent -> vector tool
# response = top_agent.query("Tell me about the arts and culture in Boston")
#
# # %%
# print(response)
#
# # %%
# # baseline
# response = base_query_engine.query("Tell me about the arts and culture in Boston")
# print(str(response))
#
# # %%
# # should use Houston agent -> vector tool
# response = top_agent.query("Give me a summary of all the positive aspects of Houston")
#
# # %%
# print(response)
#
# # %%
# # baseline
# response = base_query_engine.query(
#     "Give me a summary of all the positive aspects of Houston"
# )
# print(str(response))
#
# # %%
# # baseline: the response doesn't quite match the sources...
# response.source_nodes[1].get_content()
#
# # %%
# response = top_agent.query(
#     "Tell the demographics of Houston, and then compare that with the demographics of Chicago"
# )
#
# # %%
# print(response)
#
# # %%
# # baseline
# response = base_query_engine.query(
#     "Tell the demographics of Houston, and then compare that with the demographics of Chicago"
# )
# print(str(response))
#
# # %%
# # baseline: the response tells you nothing about Chicago...
# response.source_nodes[3].get_content()
#
# # %%
# response = top_agent.query(
#     "Tell me the differences between Shanghai and Beijing in terms of history and current economy"
# )
#
# # %%
# print(str(response))
#
# # %%
# # baseline
# response = base_query_engine.query(
#     "Tell me the differences between Shanghai and Beijing in terms of history and current economy"
# )
# print(str(response))
#
# # %%
#
#
