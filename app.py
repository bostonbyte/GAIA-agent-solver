# app.py

import os
import gradio as gr
import requests
import pandas as pd
import uuid

# --- Smolagent and Tool Imports ---
from smolagents import CodeAgent
from smolagents.models import InferenceClientModel
from smolagents.tools import Tool

# --- Imports for stateful memory tools ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document


DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space" 

# --- Import your new Tool CLASSES from tools.py ---
from tools import (
    WebSearchTool, ScrapeWebpageTool, DownloadFileTool, ReadPDFTool, ReadTextFileTool,
    YouTubeTranscriptionTool, WikiSearchTool, ArxivSearchTool, SemanticSearchTool, MultiplyTool, AddTool, SubtractTool, DivideTool, ModulusTool
)

# --- Constants and LLM setup ---
# NOTE: To handle potentially longer thought processes with the retriever,
#       it's a good idea to increase the timeout and max_tokens.
llm = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
    max_tokens=8000,
    timeout=300,
)

# --- Format Instructions ---
FORMAT_INSTRUCTIONS = """
Your final answer must follow these strict rules:
1.  First, think step-by-step and use your tools to find the answer.
2.  Once you are sure of the answer, finish your response with the exact template: FINAL ANSWER: [YOUR FINAL ANSWER].
3.  [YOUR FINAL ANSWER] must be a number, a few words, or a comma-separated list.
4.  Do not use units like $, %, or commas in numbers. Do not use articles (a, an, the) or abbreviations in strings.
"""

# In app.py

class GaiaAgent:
    def __init__(self):
        print("Initializing GaiaAgent with class-based tools...")
        
        # --- Initialize Stateless Tools ---
        web_search_tool = WebSearchTool()
        scrape_webpage_tool = ScrapeWebpageTool()
        download_file_tool = DownloadFileTool()
        read_pdf_tool = ReadPDFTool()
        read_text_file_tool = ReadTextFileTool()
        youtube_tool = YouTubeTranscriptionTool()
        wiki_tool = WikiSearchTool()
        arxiv_tool = ArxivSearchTool()
        semantic_search_tool = SemanticSearchTool()
        multiply_tool = MultiplyTool()
        add_tool = AddTool()
        subtract_tool = SubtractTool()
        divide_tool = DivideTool()
        modulus_tool = ModulusTool()

        # --- Initialize Stateful Memory Tools ---
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_texts(["Initial empty document"], self.embedding_model)
        
        class CommitToMemoryTool(Tool):
            name = "commit_to_memory"
            description = "Saves important information to the agent's short-term memory for this task."
            inputs = {"text_to_remember": {"type": "string", "description": "The text to be saved in memory."}}
            output_type = "string"
            def forward(tool_self, text_to_remember: str) -> str:
                print(f"--- Committing to Memory: '{text_to_remember[:50]}...' ---")
                new_doc = Document(page_content=text_to_remember)
                self.vector_store.add_documents([new_doc])
                return f"Successfully committed the following to memory: '{text_to_remember}'"

        class RetrieveFromMemoryTool(Tool):
            name = "retrieve_from_memory"
            description = "Retrieves relevant information from the agent's short-term memory."
            inputs = {"query": {"type": "string", "description": "A question or topic to search for in the memory."}}
            output_type = "string"
            def forward(tool_self, query: str) -> str:
                print(f"--- Retrieving from Memory with query: '{query}' ---")
                results = self.vector_store.similarity_search(query, k=3)
                return "\n\n".join([doc.page_content for doc in results]) if results else "No relevant information found."

        commit_tool = CommitToMemoryTool()
        retrieve_tool = RetrieveFromMemoryTool()

        # --- Initialize Agent ---
        self.agent = CodeAgent(
            model=llm,
            tools=[
                web_search_tool, scrape_webpage_tool, download_file_tool, read_pdf_tool,
                read_text_file_tool, youtube_tool, wiki_tool, arxiv_tool,
                semantic_search_tool, commit_tool, retrieve_tool,
                multiply_tool, add_tool, subtract_tool, divide_tool, modulus_tool
            ],
        )
        print("GaiaAgent initialized successfully.")

    # --- THIS METHOD IS NOW SIMPLIFIED ---
    def __call__(self, question: str) -> str:
        # We combine the original question with our formatting rules.
        # The agent already knows about all its tools from the initialization above.
        prompt_with_instructions = f"{question}\n\n{FORMAT_INSTRUCTIONS}"
        
        print(f"Agent received question (first 80 chars): {question[:80]}...")
        try:
            raw_output = self.agent.run(prompt_with_instructions)
            
            print(f"--- Agent's Raw Output ---\n{raw_output}\n--------------------------")
            
            # The robust parsing logic is still essential
            if "FINAL ANSWER:" in raw_output:
                model_answer = raw_output.split("FINAL ANSWER:")[-1].strip()
                print(f"Parsed Final Answer (for submission): {model_answer}")
                return model_answer
            else:
                print("Warning: Agent did not use 'FINAL ANSWER:' template. Returning full output.")
                return raw_output.strip()
        except Exception as e:
            print(f"An error occurred while the agent was running: {e}")
            return f"Agent failed to process the question due to an error: {e}"



# The run_and_submit_all function and Gradio UI code remain unchanged.
def run_and_submit_all(profile: gr.OAuthProfile | None):
    # ... (code is unchanged)
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        print("\n" + "="*50)
        print(f"Creating new agent instance for task_id: {item.get('task_id')}")
        try:
            agent = GaiaAgent()
        except Exception as e:
            print(f"Fatal error creating agent instance: {e}")
            results_log.append({"Task ID": item.get('task_id'), "Question": item.get('question'), "Submitted Answer": f"AGENT INIT ERROR: {e}"})
            continue

        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT RUN ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


with gr.Blocks() as demo:
    # ... (code is unchanged)
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Ensure you have defined your agent logic in `app.py` and your tools in `tools.py`.
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    # ... (code is unchanged)
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") 

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for GAIA Agent Evaluation...")
    demo.launch(debug=True, share=False)
