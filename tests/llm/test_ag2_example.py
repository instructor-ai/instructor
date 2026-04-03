import importlib.util
import os
import pytest
import sys

def load_run_module():
    
    spec = importlib.util.spec_from_file_location("ag2_structured_outputs", "examples/ag2-structured-outputs/run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ag2_structured_outputs"] = module
    spec.loader.exec_module(module)
    return module

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY")
def test_ag2_instructor_e2e_research():
    ag2_mod = load_run_module()
    
    result_json = ag2_mod.research_company(company_name="OpenAI")
    profile = ag2_mod.CompanyProfile.model_validate_json(result_json)
    
    assert profile.name is not None
    assert profile.sector is not None
    assert profile.description is not None
    assert len(profile.key_products) > 0

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY")
def test_ag2_instructor_e2e_analyst():
    ag2_mod = load_run_module()
    
    test_profile_json = ag2_mod.CompanyProfile(
        name="Test Corp",
        ticker="TST",
        sector="Technology",
        market_cap_billions=100.5,
        revenue_billions=20.0,
        employees=5000,
        description="A major technology testing firm building test frameworks.",
        key_products=["TestFramework", "TestRunner"]
    ).model_dump_json()

    analysis_json = ag2_mod.analyze_investment(company_profile_json=test_profile_json)
    analysis = ag2_mod.InvestmentAnalysis.model_validate_json(analysis_json)
    
    assert "Test Corp" in analysis.company
    assert analysis.recommendation in ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell")
    assert analysis.target_price > 0
    assert 0.0 <= analysis.confidence <= 1.0
    assert len(analysis.key_risks) >= 1

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY")
def test_ag2_instructor_e2e_full_workflow():
    ag2_mod = load_run_module()
    
    # We can effectively run the same sequence the script does to check end-to-end
    result = ag2_mod.user_proxy.run(
        ag2_mod.researcher, 
        message="Research Microsoft and provide a structured company profile."
    ).process()
    
    chat_history = ag2_mod.researcher.chat_messages.get(ag2_mod.user_proxy, [])
    last_tool_result = None
    for msg in reversed(chat_history):
        if msg.get("role") == "tool":
            last_tool_result = msg.get("content", "")
            break
            
    assert last_tool_result is not None, "Researcher didn't call the tool"
    
    # Verify the tool result actually parses as CompanyProfile
    profile = ag2_mod.CompanyProfile.model_validate_json(last_tool_result)
    assert profile.name is not None
    
    # Run the analyst
    result2 = ag2_mod.user_proxy.run(
        ag2_mod.analyst,
        message=f"Analyze this company profile and provide an investment recommendation:\n{last_tool_result}",
    ).process()
    
    analyst_history = ag2_mod.analyst.chat_messages.get(ag2_mod.user_proxy, [])
    analyst_last_tool_result = None
    for msg in reversed(analyst_history):
        if msg.get("role") == "tool":
            analyst_last_tool_result = msg.get("content", "")
            break
            
    assert analyst_last_tool_result is not None, "Analyst didn't call the tool"
    
    # Verify the analyst result parses
    analysis = ag2_mod.InvestmentAnalysis.model_validate_json(analyst_last_tool_result)
    assert analysis.company == profile.name or analysis.company == profile.ticker or "Microsoft" in analysis.company
    assert analysis.recommendation in ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell")
