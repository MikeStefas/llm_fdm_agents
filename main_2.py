import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from typesafe_sdk import TypeSafeClient

from harness.mas_builder import MultiAgentSystem
from harness.orchestrators import FDMOrchestrator
from harness.react_agents import FDMReactAgent
from harness.tools import (
    CODE_TOOLS,
    DOC_READER_TOOLS,
    MEDIA_READER_TOOLS,
    WEB_SEARCH_TOOLS,
)
from harness.utils import (
    CODE_INTERPRETER_DESCRIPTION,
    DOC_READER_DESCRIPTION,
    MEDIA_READER_DESCRIPTION,
    WEB_SEARCH_DESCRIPTION,
    DeterministicFDMClient,
)

load_dotenv()


llm_client = ChatOpenAI(
    model="qwen/qwen3.5-9b",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.0,
    max_tokens=4096,
    seed=42,
    reasoning_effort="none",
    extra_body={
        "provider": {
            "order": ["DeepInfra"],
            "allow_fallbacks": False,
        }
    },
)

fdm_client = DeterministicFDMClient(
    model="cloudflare/clef-flash",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api",
)

web_search_subagent = FDMReactAgent(
    subagent_name="web_search",
    tools=WEB_SEARCH_TOOLS,
    description=WEB_SEARCH_DESCRIPTION,
    fdm_client=fdm_client,
    llm_client=llm_client,
)

doc_reader_subagent = FDMReactAgent(
    subagent_name="doc_reader",
    tools=DOC_READER_TOOLS,
    description=DOC_READER_DESCRIPTION,
    fdm_client=fdm_client,
    llm_client=llm_client,
)

media_reader_subagent = FDMReactAgent(
    subagent_name="media_reader",
    tools=MEDIA_READER_TOOLS,
    description=MEDIA_READER_DESCRIPTION,
    fdm_client=fdm_client,
    llm_client=llm_client,
)

code_interpreter_subagent = FDMReactAgent(
    subagent_name="code_interpreter",
    tools=CODE_TOOLS,
    description=CODE_INTERPRETER_DESCRIPTION,
    fdm_client=fdm_client,
    llm_client=llm_client,
)

subagents = [
    web_search_subagent,
    doc_reader_subagent,
    media_reader_subagent,
    code_interpreter_subagent,
]

fdm_orchestrator = FDMOrchestrator(
    subagents=subagents,
    fdm_client=fdm_client,
    llm_client=llm_client,
)

mas = MultiAgentSystem(
    orchestrator=fdm_orchestrator,
    log_path="results/fdm/mas_log.csv",
)
graph = mas.build_graph()


tasks = [
    {
        "task_id": "b2c257e0-3ad7-4f05-b8e3-d9da973be36e",
        "file": "data/gaia/b2c257e0-3ad7-4f05-b8e3-d9da973be36e.jpg",
        "question": (
            "If this whole pint is made up of ice cream, how many percent above or below "
            "the US federal standards for butterfat content is it when using the standards as "
            "reported by Wikipedia in 2020? Answer as + or - a number rounded to one decimal place. "
            "The attached image of the pint label is located at: data/gaia/b2c257e0-3ad7-4f05-b8e3-d9da973be36e.jpg"
        ),
        "expected_answer": "+4.6",
    },
    {
        "task_id": "17b5a6a3-bc87-42e8-b0fb-6ab0781ef2cc",
        "file": None,
        "question": (
            "I’m researching species that became invasive after people who kept them as pets released them. "
            "There’s a certain species of fish that was popularized as a pet by being the main character "
            "of the movie Finding Nemo. According to the USGS, where was this fish found as a nonnative species, "
            "before the year 2020? I need the answer formatted as the five-digit zip codes of the places the species was found, "
            "separated by commas if there is more than one place."
        ),
        "expected_answer": "34689",
    },
    {
        "task_id": "2b3ef98c-cc05-450b-a719-711aee40ac65",
        "file": "data/gaia/2b3ef98c-cc05-450b-a719-711aee40ac65.mp3",
        "question": (
            "Could you help me out with this assignment? Our professor sprung it on us at the end of class Friday, "
            "and I'm still trying to figure it out. The question he asked us was about an anagram. I've attached an audio recording "
            "of the question that he asked (located at data/gaia/2b3ef98c-cc05-450b-a719-711aee40ac65.mp3), so if you could please "
            "take a listen and give me the answer, I'd really appreciate the help. Please limit your response to the anagram text that "
            "could be generated from the original line which fulfills the professor's request, without any other commentary. "
            "Also, please don't include any punctuation in your response."
        ),
        "expected_answer": "To be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune",
    },
    {
        "task_id": "076c8171-9b3b-49b9-a477-244d2a532826",
        "file": "data/gaia/076c8171-9b3b-49b9-a477-244d2a532826.xlsx",
        "question": (
            "The attached file (located at data/gaia/076c8171-9b3b-49b9-a477-244d2a532826.xlsx) contains a list of vendors "
            "in the Liminal Springs mall, along with each vendor’s monthly revenue and the rent they pay the mall. "
            "I want you to find the vendor that makes the least money, relative to the rent it pays. "
            "Then, tell me what is listed in the “type” column for that vendor."
        ),
        "expected_answer": "Finance",
    },
    {
        "task_id": "7dd30055-0198-452e-8c25-f73dbe27dcb8",
        "file": "data/gaia/7dd30055-0198-452e-8c25-f73dbe27dcb8.pdb",
        "question": (
            "Using the Biopython library in Python, parse the PDB file of the protein identified by the PDB ID 5wb7 "
            "from the RCSB Protein Data Bank (the file is provided locally at: data/gaia/7dd30055-0198-452e-8c25-f73dbe27dcb8.pdb). "
            "Calculate the distance between the first and second atoms as they are listed in the PDB file. "
            "Report the answer in Angstroms, rounded to the nearest picometer."
        ),
        "expected_answer": "1.456",
    },
    {
        "task_id": "c61d22de-5f6c-4958-a7f6-5e9707bd3466",
        "file": None,
        "question": (
            "A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, "
            "where each axis has a label word at both ends. Which of these words is used to describe a type of society in a "
            "Physics and Society article submitted to arXiv.org on August 11, 2016?"
        ),
        "expected_answer": "egalitarian",
    },
]


if __name__ == "__main__":
    for k in range(3):
        for i in [tasks[1]]:
            task_id = i["task_id"]
            mas.set_log_path(f"results/fdm_1/mas_log_{task_id}.csv")
            question = i["question"]
            expected = i["expected_answer"]

            print(f"Running Task: {task_id}")
            print(f"Expected: {expected}\n")

            initial_state = {
                "messages": [HumanMessage(content=question)],
                "task_id": task_id,
            }

            try:
                res = graph.invoke(initial_state)
                final_msg = res["messages"][-1].content
                print(f"Final Result: {final_msg}")
            except Exception as e:
                print(f"Error task {task_id}: {e}")
