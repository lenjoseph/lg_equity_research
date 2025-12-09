"""Metrics models for observability tracking."""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AgentMetrics(BaseModel):
    """Metrics for a single agent execution."""

    agent_name: str
    latency_ms: float = Field(description="Latency in milliseconds")
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    model: Optional[str] = None
    cached: bool = False  # Whether result was served from cache


class RequestMetrics(BaseModel):
    """Aggregate metrics for an entire research request."""

    total_latency_ms: float = Field(
        default=0.0, description="Total request latency in milliseconds"
    )
    agent_metrics: Dict[str, AgentMetrics] = Field(
        default_factory=dict, description="Per-agent metrics"
    )
    total_input_tokens: int = Field(
        default=0, description="Total input tokens across all agents"
    )
    total_output_tokens: int = Field(
        default=0, description="Total output tokens across all agents"
    )
    total_tokens: int = Field(default=0, description="Total tokens across all agents")

    def add_agent_metrics(self, metrics: AgentMetrics) -> None:
        """Add agent metrics and update totals."""
        self.agent_metrics[metrics.agent_name] = metrics
        if not metrics.cached:
            self.total_input_tokens += metrics.token_usage.input_tokens
            self.total_output_tokens += metrics.token_usage.output_tokens
            self.total_tokens += metrics.token_usage.total_tokens

    def merge(self, other: "RequestMetrics") -> "RequestMetrics":
        """Merge another RequestMetrics into this one (for parallel agent execution)."""
        merged = RequestMetrics(
            total_latency_ms=max(self.total_latency_ms, other.total_latency_ms),
            agent_metrics={**self.agent_metrics, **other.agent_metrics},
            total_input_tokens=self.total_input_tokens + other.total_input_tokens,
            total_output_tokens=self.total_output_tokens + other.total_output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )
        return merged

    def to_response_dict(self) -> dict:
        """Convert metrics to API response format."""
        return {
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_tokens,
            },
            "agents": {
                name: {
                    "latency_ms": round(m.latency_ms, 2),
                    "tokens": {
                        "input": m.token_usage.input_tokens,
                        "output": m.token_usage.output_tokens,
                        "total": m.token_usage.total_tokens,
                    },
                    "model": m.model,
                    "cached": m.cached,
                }
                for name, m in self.agent_metrics.items()
            },
        }


def merge_metrics(left: RequestMetrics, right: RequestMetrics) -> RequestMetrics:
    """Reducer function for LangGraph to merge metrics from parallel nodes."""
    if left is None:
        return right
    if right is None:
        return left
    return left.merge(right)
