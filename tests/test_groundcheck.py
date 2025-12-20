"""Tests for GroundCheck source grounding verification."""

import pytest
from instructor.groundcheck import (
    GroundCheck,
    VerificationResult,
    FieldResult,
    VerificationMethod,
    grounding_validator,
    verify_extraction,
)


class TestGroundCheckBasic:
    """Test basic GroundCheck functionality."""
    
    def test_exact_match(self):
        gc = GroundCheck()
        result = gc.verify(
            source_text="Invoice #12345 from Acme Corp",
            extracted_data={"invoice_number": "12345", "vendor": "Acme Corp"}
        )
        assert result.field_results["invoice_number"].confidence > 0.9
        assert result.field_results["vendor"].confidence > 0.9
        assert len(result.flagged_fields) == 0
    
    def test_hallucination_detection(self):
        gc = GroundCheck()
        result = gc.verify(
            source_text="Invoice #12345 from Acme Corp. Total: $500",
            extracted_data={
                "invoice_number": "12345",
                "vendor": "Acme Corp",
                "total": 500,
                "currency": "USD",
                "payment_terms": "Net 30"
            }
        )
        assert "currency" in result.flagged_fields
        assert "payment_terms" in result.flagged_fields
        assert "invoice_number" not in result.flagged_fields
    
    def test_numeric_matching(self):
        gc = GroundCheck()
        result = gc.verify(
            source_text="Total amount: $1,234.56",
            extracted_data={"total": 1234.56}
        )
        assert result.field_results["total"].confidence > 0.8
    
    def test_case_insensitivity(self):
        gc = GroundCheck()
        result = gc.verify(
            source_text="ACME CORPORATION",
            extracted_data={"company": "Acme Corporation"}
        )
        assert result.field_results["company"].confidence > 0.9
    
    def test_empty_values(self):
        gc = GroundCheck()
        result = gc.verify(
            source_text="Some text here",
            extracted_data={"field1": "", "field2": None}
        )
        assert result.field_results["field1"].flagged
        assert result.field_results["field2"].flagged


class TestVerifyExtraction:
    """Test the convenience function."""
    
    def test_basic_verification(self):
        result = verify_extraction(
            source_text="Product: iPhone, Price: $999",
            extracted_data={"product": "iPhone", "price": 999}
        )
        assert isinstance(result, VerificationResult)
        assert result.overall_confidence > 0.7
    
    def test_raise_on_hallucination(self):
        with pytest.raises(ValueError, match="Hallucination detected"):
            verify_extraction(
                source_text="Product: iPhone",
                extracted_data={"product": "iPhone", "color": "blue"},
                raise_on_hallucination=True
            )


class TestGroundingValidator:
    """Test the Pydantic validator integration."""
    
    def test_validator_passes(self):
        source = "Invoice #12345"
        validator = grounding_validator(source, threshold=0.7)
        result = validator("12345")
        assert result == "12345"
    
    def test_validator_fails(self):
        source = "Invoice #12345"
        validator = grounding_validator(source, threshold=0.7)
        with pytest.raises(ValueError, match="not grounded"):
            validator("99999")


class TestRealWorldScenarios:
    """Test real-world extraction scenarios."""
    
    def test_invoice_extraction(self):
        gc = GroundCheck()
        source = """
        INVOICE
        Invoice No: INV-2024-0892
        Date: December 15, 2024
        Bill To: Acme Corporation
        Subtotal: $549.90
        Tax (8%): $43.99
        Total Due: $593.89
        """
        extracted = {
            "invoice_number": "INV-2024-0892",
            "customer": "Acme Corporation",
            "subtotal": 549.90,
            "total": 593.89,
            "payment_terms": "Net 30"
        }
        result = gc.verify(source, extracted)
        assert result.field_results["invoice_number"].confidence > 0.9
        assert result.field_results["customer"].confidence > 0.9
        assert "payment_terms" in result.flagged_fields
    
    def test_contact_extraction(self):
        gc = GroundCheck()
        business_card = """
        Jane Doe
        Senior Software Engineer
        TechCorp Inc.
        jane.doe@techcorp.com
        +1 (555) 987-6543
        """
        extracted = {
            "name": "Jane Doe",
            "title": "Senior Software Engineer",
            "company": "TechCorp Inc.",
            "email": "jane.doe@techcorp.com",
            "linkedin": "linkedin.com/in/janedoe"
        }
        result = gc.verify(business_card, extracted)
        assert result.field_results["name"].confidence > 0.9
        assert result.field_results["email"].confidence > 0.9
        assert "linkedin" in result.flagged_fields
