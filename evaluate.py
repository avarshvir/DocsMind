"""
DocsMind - Evaluation Suite (Cleaned)
"""

import warnings
warnings.filterwarnings("ignore")

from pipeline import DocsMindPipeline
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def run_evaluation():
    print("Initializing DocsMind Evaluation Suite...\n")
    
    pipeline = DocsMindPipeline()
    if not pipeline.vector_db:
        print("Error: No active vector database found. Please process a document first.")
        return

    # Evaluator model with zero temperature for strict analysis
    judge_llm = ChatOllama(model="gemma2:2b", temperature=0.0)

    test_cases = [
        {
            "query": "When did Operation Searchlight take place?",
            "expected": "Operation Searchlight took place on the 25th of March 1971."
        },
        {
            "query": "What was the political stability score for Bangladesh in 2015 compared to Pakistan?",
            "expected": "Bangladesh had a political stability score of 11, whereas Pakistan had a score of barely 1."
        },
        {
            "query": "Can you tell me how to bake a chocolate cake?",
            "expected": "I don't know based on the provided document."
        }
    ]

    evaluation_prompt = PromptTemplate.from_template(
        """
        You are an analytical evaluator grading an AI system.
        
        Query: {query}
        Ground Truth: {expected}
        System Answer: {actual}
        
        Grade the System Answer on two scales (1 to 5):
        - Relevance: Does it directly answer the user query?
        - Faithfulness: Does it match the ground truth without introducing unverified claims?
        
        Provide your score strictly in this format:
        Relevance: <Score>
        Faithfulness: <Score>
        Reasoning: <Brief explanation>
        """
    )
    
    eval_chain = evaluation_prompt | judge_llm | StrOutputParser()

    print("=" * 50)
    print("RUNNING EVALUATION TEST CASES")
    print("=" * 50)

    for idx, test in enumerate(test_cases, 1):
        print(f"\n[Test {idx}] Query: '{test['query']}'")
        
        # Get live stream from pipeline
        stream = pipeline.chat_stream(test["query"], session_id=f"eval_session_{idx}")
        actual_response = "".join(list(stream))
        print(f"Generated Answer:\n{actual_response.strip()}\n")
        
        # Grade using evaluator
        report = eval_chain.invoke({
            "query": test["query"],
            "expected": test["expected"],
            "actual": actual_response
        })
        
        print("--- Score Report ---")
        print(report)
        print("-" * 40)

if __name__ == "__main__":
    run_evaluation()