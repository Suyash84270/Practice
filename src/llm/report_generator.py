import os
import sys
from typing import Dict, TypedDict

# Ensure project root imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from src.logger.logger import get_logger
from src.exception.custom_exception import CustomException
from src.utils.helper import read_yaml
from src.constants.constants import MAIN_CONFIG_FILE

logger = get_logger(__name__)


# ---------------------------------------------------------
# LangGraph State Definition
# ---------------------------------------------------------
class ReportState(TypedDict):
    """
    Shared state passed between LangGraph nodes.
    """
    detections: Dict[str, int]
    formatted_input: str
    report: str


class ReportGenerator:
    """
    Generates a structured construction site inspection report
    from CV detection outputs using a LangGraph workflow.
    """

    def __init__(self):
        try:
            logger.info("Initializing Report Generator")

            config = read_yaml(MAIN_CONFIG_FILE)

            self.model_name = config["llm"]["model"]
            self.temperature = config["llm"].get("temperature", 0.3)

            # -------------------------------------------------
            # Active LLM: OpenAI GPT-4o-mini
            # -------------------------------------------------
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature
            )

            # -------------------------------------------------
            # Alternative: Open-source Llama (example)
            # Uncomment if using Ollama or HuggingFace locally
            # -------------------------------------------------
            #
            # from langchain_community.chat_models import ChatOllama
            #
            # self.llm = ChatOllama(
            #     model="llama3",
            #     temperature=0.3
            # )
            #
            # This allows running the report generator locally
            # without using OpenAI API.
            # -------------------------------------------------

            self.workflow = self._build_graph()

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------------------------------------------------
    # LangGraph Workflow Definition
    # ---------------------------------------------------------
    def _build_graph(self):
        """
        Builds the LangGraph workflow.

        Nodes:
        1. input_node           -> receives detection results
        2. format_prompt_node   -> converts detections into prompt text
        3. llm_generation_node  -> calls LLM
        4. output_node          -> returns report
        """

        graph = StateGraph(ReportState)

        graph.add_node("input_node", self._input_node)
        graph.add_node("format_prompt_node", self._format_prompt_node)
        graph.add_node("llm_generation_node", self._llm_generation_node)
        graph.add_node("output_node", self._output_node)

        graph.set_entry_point("input_node")

        graph.add_edge("input_node", "format_prompt_node")
        graph.add_edge("format_prompt_node", "llm_generation_node")
        graph.add_edge("llm_generation_node", "output_node")
        graph.add_edge("output_node", END)

        return graph.compile()

    # ---------------------------------------------------------
    # Node 1: Input Node
    # ---------------------------------------------------------
    def _input_node(self, state: ReportState) -> ReportState:
        """
        Receives raw CV detection results.
        """
        logger.info("LangGraph input node received detections")
        return state

    # ---------------------------------------------------------
    # Node 2: Prompt Formatting
    # ---------------------------------------------------------
    def _format_prompt_node(self, state: ReportState) -> ReportState:
        """
        Converts detection dictionary into structured prompt text
        for the LLM.
        """

        detections = state["detections"]

        formatted_lines = [
            f"{obj}: {count}" for obj, count in detections.items()
        ]

        detection_summary = "\n".join(formatted_lines)

        prompt = f"""
You are an AI Construction Safety Inspector.

Analyze the following computer vision detections from a construction site image.

Detected objects:
{detection_summary}

Generate a structured site inspection report containing:
1. Site summary
2. Worker safety observations
3. Equipment observations
4. Potential safety violations
5. Safety recommendations

Write the report professionally as an engineering inspection report.
"""

        state["formatted_input"] = prompt

        logger.info("Prompt formatted for LLM")

        return state

    # ---------------------------------------------------------
    # Node 3: LLM Generation
    # ---------------------------------------------------------
    def _llm_generation_node(self, state: ReportState) -> ReportState:
        """
        Sends formatted prompt to LLM and generates report.
        """

        logger.info("Sending prompt to LLM")

        response = self.llm.invoke(state["formatted_input"])

        state["report"] = response.content

        logger.info("LLM report generated")

        return state

    # ---------------------------------------------------------
    # Node 4: Output Node
    # ---------------------------------------------------------
    def _output_node(self, state: ReportState) -> ReportState:
        """
        Final node returning generated report.
        """
        logger.info("LangGraph workflow completed")
        return state

    # ---------------------------------------------------------
    # Public Method
    # ---------------------------------------------------------
    def generate_report(self, detections: Dict[str, int]) -> str:
        """
        Executes LangGraph workflow.

        Args:
            detections (Dict[str, int]):
                Object detection counts from CV model.

        Returns:
            str: Final construction inspection report.
        """

        try:
            result = self.workflow.invoke({
                "detections": detections
            })

            return result["report"]

        except Exception as e:
            raise CustomException(e, sys)