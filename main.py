import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from harness.mas_builder import MultiAgentSystem
from harness.orchestrators import LLMOrchestrator
from harness.react_agents import LLMReactAgent
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
)

from config import llm_client

web_search_subagent = LLMReactAgent(
    subagent_name="web_search",
    tools=WEB_SEARCH_TOOLS,
    description=WEB_SEARCH_DESCRIPTION,
    llm_client=llm_client,
)

doc_reader_subagent = LLMReactAgent(
    subagent_name="doc_reader",
    tools=DOC_READER_TOOLS,
    description=DOC_READER_DESCRIPTION,
    llm_client=llm_client,
)

media_reader_subagent = LLMReactAgent(
    subagent_name="media_reader",
    tools=MEDIA_READER_TOOLS,
    description=MEDIA_READER_DESCRIPTION,
    llm_client=llm_client,
)

code_interpreter_subagent = LLMReactAgent(
    subagent_name="code_interpreter",
    tools=CODE_TOOLS,
    description=CODE_INTERPRETER_DESCRIPTION,
    llm_client=llm_client,
)

subagents = [
    web_search_subagent,
    doc_reader_subagent,
    media_reader_subagent,
    code_interpreter_subagent,
]

llm_orchestrator = LLMOrchestrator(
    subagents=subagents,
    llm_client=llm_client,
)

mas = MultiAgentSystem(
    orchestrator=llm_orchestrator,
    log_path="results/llm_1/mas_log.csv",
)
graph = mas.build_graph()


tasks = [
    {
        "task_id": "676e5e31-a554-4acc-9286-b60d90a92d26",
        "file": None,
        "question": (
            'In July 2, 1959 United States standards for grades of processed fruits, vegetables, '
            'and certain other products listed as dehydrated, consider the items in the "dried '
            'and dehydrated section" specifically marked as dehydrated along with any items in '
            'the Frozen/Chilled section that contain the whole name of the item, but not if '
            "they're marked Chilled. As of August 2023, what is the percentage (to the nearest "
            'percent) of those standards that have been superseded by a new version since the '
            'date given in the 1959 standards?'
        ),
        "expected_answer": "86",
    },
    {
        "task_id": "bec74516-02fc-48dc-b202-55e78d0e17cf",
        "file": "data/gaia/bec74516-02fc-48dc-b202-55e78d0e17cf.jsonld",
        "question": (
            "What is the average number of pre-2020 works on the open researcher and contributor "
            "identification pages of the people whose identification is in this file? (The "
            "attached file is located at: data/gaia/bec74516-02fc-48dc-b202-55e78d0e17cf.jsonld)"
        ),
        "expected_answer": "26.4",
    },
    {
        "task_id": "00d579ea-0889-4fd9-a771-2c8d79835c8d",
        "file": None,
        "question": (
            "Assuming scientists in the famous youtube video The Thinking Machine (Artificial "
            "Intelligence in the 1960s) were interviewed the same year, what is the name of the "
            "scientist predicting the sooner thinking machines or robots? Answer using the "
            "format First name Last name"
        ),
        "expected_answer": "Claude Shannon",
    },
    {
        "task_id": "384d0dd8-e8a4-4cfe-963c-d37f256e7662",
        "file": None,
        "question": (
            "In the NCATS PubChem compound database for Food Additive Status classification, "
            "find the compound that has a molecular weight of 100 g/mol or less, 6 heavy atoms, "
            "1 or fewer hydrogen bond acceptors, and a complexity between 10 and 15. Of the "
            "shared gene-chemical co-occurrences between its two possible enzyme "
            "transformations, what is the PubChem CID of the heaviest by molecular weight?"
        ),
        "expected_answer": "4192",
    },
    {
        "task_id": "de9887f5-ead8-4727-876f-5a4078f8598c",
        "file": None,
        "question": (
            "What integer-rounded percentage of the total length of the harlequin shrimp "
            "recorded in Omar Valencfia-Mendez 2017 paper was the sea star fed to the same type "
            "of shrimp in G. Curt Fiedler's 2002 paper?"
        ),
        "expected_answer": "22",
    },
    {
        "task_id": "983bba7c-c092-455f-b6c9-7857003d48fc",
        "file": None,
        "question": (
            "What animals that were mentioned in both Ilias Lagkouvardos's and Olga Tapia's "
            "papers on the alvei species of the genus named for Copenhagen outside the "
            "bibliographies were also present in the 2021 article cited on the alvei species' "
            "Wikipedia page about a multicenter, randomized, double-blind study?"
        ),
        "expected_answer": "mice",
    },
    {
        "task_id": "9b54f9d9-35ee-4a14-b62f-d130ea00317f",
        "file": "data/gaia/9b54f9d9-35ee-4a14-b62f-d130ea00317f.zip",
        "question": (
            "Which of the text elements under CATEGORIES in the XML would contain the one food "
            "in the spreadsheet that does not appear a second time under a different name? (The "
            "attached file is located at: data/gaia/9b54f9d9-35ee-4a14-b62f-d130ea00317f.zip)"
        ),
        "expected_answer": "Soups and Stews",
    },
    {
        "task_id": "56db2318-640f-477a-a82f-bc93ad13e882",
        "file": None,
        "question": (
            "The following numbers function similarly to ISBN 13 numbers, however, their "
            "validation methods are slightly different. Rather than using alternate weights of 1 "
            "and 3, the checksum digit is calculated with an alternate weight of 1 and some "
            "other positive integer less than 10. Otherwise, the checksum digit is calculated as "
            "expected. Unfortunately, there is an error in the data. Two adjacent columns have "
            "been transposed. These errored columns do not involve the final column or one of "
            "the first three columns. Using this information, please provide all potential "
            "solutions with the unknown weight and the smaller index of the two errored columns "
            "(assume we start our indexing at 0 and ignore hyphens). Give your answer in the "
            "form x, y where x is the weight and y is the smaller index of the two transposed "
            "columns.\n\n"
            "978-354181391-9\n"
            "978-946669746-1\n"
            "978-398036139-6\n"
            "978-447656680-4\n"
            "978-279586664-7\n"
            "978-595073693-3\n"
            "978-976647652-6\n"
            "978-591178125-5\n"
            "978-728465924-5\n"
            "978-414825155-9"
        ),
        "expected_answer": "7, 9",
    },
    {
        "task_id": "8131e2c0-0083-4265-9ce7-78c2d568425d",
        "file": None,
        "question": (
            "I was trying to remember how well the Cheater Beater performed in comparison to the "
            "Cheater when James tested it on his channel. I know that the Cheater still "
            "outperformed the Cheater Beater in terms of CFM. Could you please look that up for "
            "me, and report the CFM of both the Cheater and the Cheater Beater? I'm not sure if "
            "he made any changes to his testing, but this was back in season 4, so just report "
            "the value from that season. Please format your response like this: CFM number for "
            "Cheater, CFM number for Cheater beater"
        ),
        "expected_answer": "101.376, 84.348",
    },
    {
        "task_id": "72c06643-a2fa-4186-aa5c-9ec33ae9b445",
        "file": None,
        "question": (
            "What is the volume in milliliters of a system comprised of 0.312 kg Freon-12 "
            "refrigerant when placed at the bottom of the Marianas Trench and allowed to "
            "stabilize at the Trench's peak temperature, rounded to the nearest mL? Provide your "
            "answer as just an integer value."
        ),
        "expected_answer": "55",
    },
]

if __name__ == "__main__":
    for i in tasks:
        task_id = i["task_id"]
        mas.set_log_path(f"results/llm_1/mas_log_{task_id}.csv")
        question = i["question"]
        expected = i["expected_answer"]

        print(f"Running Task: {task_id}")
        print(f"Expected:     {expected}\n")

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
