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
    budget_exceeded: bool = False  # Whether token budget was exceeded


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
    token_budget: Optional[int] = Field(
        default=None,
        description="Maximum tokens allowed for entire request (None = unlimited)",
    )
    budget_exceeded: bool = Field(
        default=False, description="Whether the token budget was exceeded"
    )

    def add_agent_metrics(self, metrics: AgentMetrics) -> None:
        """Add agent metrics and update totals."""
        self.agent_metrics[metrics.agent_name] = metrics
        if not metrics.cached:
            self.total_input_tokens += metrics.token_usage.input_tokens
            self.total_output_tokens += metrics.token_usage.output_tokens
            self.total_tokens += metrics.token_usage.total_tokens

        # Track if any agent exceeded budget
        if metrics.budget_exceeded:
            self.budget_exceeded = True

        # Check if total exceeds budget
        if self.token_budget is not None and self.total_tokens > self.token_budget:
            self.budget_exceeded = True

    def is_within_budget(self) -> bool:
        """Check if current usage is within budget."""
        if self.token_budget is None:
            return True
        return self.total_tokens <= self.token_budget

    def remaining_budget(self) -> Optional[int]:
        """Get remaining token budget (None if unlimited)."""
        if self.token_budget is None:
            return None
        return max(0, self.token_budget - self.total_tokens)

    def merge(self, other: "RequestMetrics") -> "RequestMetrics":
        """Merge another RequestMetrics into this one (for parallel agent execution)."""
        # Use the stricter budget (non-None takes precedence, then minimum)
        merged_budget = None
        if self.token_budget is not None and other.token_budget is not None:
            merged_budget = min(self.token_budget, other.token_budget)
        elif self.token_budget is not None:
            merged_budget = self.token_budget
        elif other.token_budget is not None:
            merged_budget = other.token_budget

        merged = RequestMetrics(
            total_latency_ms=max(self.total_latency_ms, other.total_latency_ms),
            agent_metrics={**self.agent_metrics, **other.agent_metrics},
            total_input_tokens=self.total_input_tokens + other.total_input_tokens,
            total_output_tokens=self.total_output_tokens + other.total_output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            token_budget=merged_budget,
            budget_exceeded=self.budget_exceeded or other.budget_exceeded,
        )

        # Re-check if merged total exceeds budget
        if (
            merged.token_budget is not None
            and merged.total_tokens > merged.token_budget
        ):
            merged.budget_exceeded = True

        return merged

    def to_response_dict(self) -> dict:
        """Convert metrics to API response format."""
        response = {
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
                    "budget_exceeded": m.budget_exceeded,
                }
                for name, m in self.agent_metrics.items()
            },
        }

        # Include budget information if set
        if self.token_budget is not None:
            response["token_budget"] = {
                "limit": self.token_budget,
                "used": self.total_tokens,
                "remaining": self.remaining_budget(),
                "exceeded": self.budget_exceeded,
            }

        return response


def merge_metrics(left: RequestMetrics, right: RequestMetrics) -> RequestMetrics:
    """Reducer function for LangGraph to merge metrics from parallel nodes."""
    if left is None:
        return right
    if right is None:
        return left
    return left.merge(right)
