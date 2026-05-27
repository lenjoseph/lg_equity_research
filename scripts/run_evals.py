from langsmith import Client, evaluate
from graph import research_chain
from agents.evaluation.agent import evaluate_aggregated_sentement

client = Client()
dataset_name = "Equity Research Golden Dataset"

# Example: Create dataset if it doesn't exist
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[
            {"ticker": "AAPL", "trade_duration": "long", "trade_direction": "bullish"},
            {"ticker": "TSLA", "trade_duration": "short", "trade_direction": "bearish"},
        ],
        outputs=[
            # Optional: Add ground truth if you have it, or leave empty for reference-free eval
        ],
        dataset_id=dataset.id,
    )


def relevance_evaluator(run, example):

    agent_output = run.outputs.get("combined_sentiment")

    score = evaluate_aggregated_sentement(agent_output)

    return {
        "key": "compliance_score",
        "score": 1 if score[0]["compliant"] else 0,
        "comment": score[0]["feedback"],
    }


results = evaluate(
    research_chain.invoke,
    data=dataset_name,
    evaluators=[relevance_evaluator],
    experiment_prefix="equity-research-v1",
)
