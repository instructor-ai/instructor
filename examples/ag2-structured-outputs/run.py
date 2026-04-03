"""AG2 Multi-Agent Structured Outputs with Instructor.

This example demonstrates how AG2 (formerly AutoGen) multi-agent framework
can leverage Instructor for guaranteed structured, Pydantic-validated outputs.

Two AG2 agents collaborate on analyzing a company:
- **Researcher**: Gathers company data using a tool, returns structured CompanyProfile
- **Analyst**: Takes the profile and produces a structured InvestmentAnalysis

Instructor ensures every LLM response is a valid Pydantic model — no manual
JSON parsing, no schema errors, automatic retries on validation failures.

Requirements:
    pip install instructor ag2[openai]

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/ag2-structured-outputs/run.py

AG2: https://ag2.ai/ | 500K+ monthly PyPI downloads
Instructor: https://python.useinstructor.com/ | 3M+ monthly downloads
"""

from __future__ import annotations

import json
import os
from typing import Literal

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from autogen import AssistantAgent, UserProxyAgent, LLMConfig


# ---------------------------------------------------------------------------
# 1. Define Pydantic models for structured outputs
# ---------------------------------------------------------------------------


class CompanyProfile(BaseModel):
    """Structured company profile extracted by the researcher agent."""

    name: str = Field(description="Official company name")
    ticker: str = Field(description="Stock ticker symbol")
    sector: str = Field(description="Industry sector")
    market_cap_billions: float = Field(description="Market cap in billions USD")
    revenue_billions: float = Field(description="Annual revenue in billions USD")
    employees: int = Field(description="Number of employees")
    description: str = Field(description="Brief company description (1-2 sentences)")
    key_products: list[str] = Field(
        description="Top 3 products or services", min_length=1, max_length=5
    )


class InvestmentAnalysis(BaseModel):
    """Structured investment analysis produced by the analyst agent."""

    company: str = Field(description="Company name being analyzed")
    recommendation: Literal["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"] = Field(
        description="Investment recommendation"
    )
    target_price: float = Field(description="12-month price target in USD", gt=0)
    confidence: float = Field(description="Confidence level 0-1", ge=0.0, le=1.0)
    bull_case: str = Field(description="Bull case thesis (1-2 sentences)")
    bear_case: str = Field(description="Bear case thesis (1-2 sentences)")
    key_risks: list[str] = Field(
        description="Top 3 investment risks", min_length=1, max_length=5
    )


# ---------------------------------------------------------------------------
# 2. Instructor-powered extraction function
# ---------------------------------------------------------------------------

# Patch OpenAI client with Instructor for validated structured outputs
instructor_client = instructor.from_openai(OpenAI())


def extract_structured(prompt: str, response_model: type[BaseModel]) -> BaseModel:
    """Use Instructor to extract a validated Pydantic model from LLM output.

    Instructor handles:
    - Schema enforcement via function calling
    - Automatic retries on validation errors
    - Type coercion and Pydantic validation
    """
    return instructor_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_model=response_model,
    )


# ---------------------------------------------------------------------------
# 3. AG2 tool that uses Instructor for structured output
# ---------------------------------------------------------------------------

llm_config = LLMConfig(
    {
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_type": "openai",
    }
)

researcher = AssistantAgent(
    name="researcher",
    system_message=(
        "You are a financial researcher. When asked about a company, "
        "use the research_company tool to gather structured data. "
        "Return the tool result to the user. After returning the result, "
        "say output exactly 'TERMINATE' to end the conversation."
    ),
    llm_config=llm_config,
)

analyst = AssistantAgent(
    name="analyst",
    system_message=(
        "You are an investment analyst. When given a company profile, "
        "use the analyze_investment tool to produce a structured analysis. "
        "Return the tool result to the user. After returning the result, "
        "say output exactly 'TERMINATE' to end the conversation."
    ),
    llm_config=llm_config,
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config=False,
    is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", ""),
)


@user_proxy.register_for_execution()
@researcher.register_for_llm(
    description="Research a company and return a structured profile with key metrics."
)
def research_company(company_name: str) -> str:
    """Research a company using Instructor for guaranteed structured output."""
    profile: CompanyProfile = extract_structured(
        prompt=f"Provide a detailed company profile for {company_name}. "
        f"Include accurate financial data and key products.",
        response_model=CompanyProfile,
    )
    return profile.model_dump_json(indent=2)


@user_proxy.register_for_execution()
@analyst.register_for_llm(
    description="Analyze a company profile and produce an investment recommendation."
)
def analyze_investment(company_profile_json: str) -> str:
    """Analyze a company using Instructor for guaranteed structured output."""
    profile = CompanyProfile.model_validate_json(company_profile_json)
    analysis: InvestmentAnalysis = extract_structured(
        prompt=f"Based on this company profile, provide an investment analysis:\n"
        f"Company: {profile.name} ({profile.ticker})\n"
        f"Sector: {profile.sector}\n"
        f"Market Cap: ${profile.market_cap_billions}B\n"
        f"Revenue: ${profile.revenue_billions}B\n"
        f"Employees: {profile.employees:,}\n"
        f"Products: {', '.join(profile.key_products)}\n"
        f"Description: {profile.description}",
        response_model=InvestmentAnalysis,
    )
    return analysis.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# 4. Run the multi-agent workflow
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("AG2 + Instructor: Multi-Agent Structured Outputs")
    print("=" * 60)

    # Step 1: Researcher gathers structured company data
    print("\n--- Step 1: Researcher gathering company profile ---\n")
    result = user_proxy.run(
        researcher,
        message="Research Apple Inc and provide a structured company profile.",
    ).process()

    # Display the structured output
    print("\n--- Step 2: Analyst producing investment analysis ---\n")

    # Get the last assistant message containing the company profile
    chat_history = researcher.chat_messages.get(user_proxy, [])
    last_tool_result = None
    for msg in reversed(chat_history):
        if msg.get("role") == "tool":
            last_tool_result = msg.get("content", "")
            break

    if last_tool_result:
        # Feed the profile to the analyst
        result = user_proxy.run(
            analyst,
            message=f"Analyze this company profile and provide an investment recommendation:\n{last_tool_result}",
        ).process()

    # Show final structured results
    print("\n" + "=" * 60)
    print("STRUCTURED RESULTS (Pydantic-validated)")
    print("=" * 60)

    analyst_history = analyst.chat_messages.get(user_proxy, [])
    for msg in analyst_history:
        if msg.get("role") == "tool" and msg.get("content"):
            try:
                parsed = json.loads(msg["content"])
                print(json.dumps(parsed, indent=2))
            except json.JSONDecodeError:
                print(msg["content"])

    # ---------------------------------------------------------------------------
    # Try It Yourself: Change the company name below and re-run!
    # ---------------------------------------------------------------------------
    # result = user_proxy.run(
    #     researcher,
    #     message="Research Tesla Inc and provide a structured company profile."
    # ).process()
