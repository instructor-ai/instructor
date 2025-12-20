"""Tests for GroundCheck source grounding verification."""

import pytest
from instructor.groundcheck import (
    GroundCheck,
    VerificationResult,
    FieldResult,
    VerificationMethod,
    grounding_validator,
    verify_extraction,
    with_grounding,
    GroundedExtractor,
)
from instructor.core.exceptions import HallucinationError


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
        with pytest.raises(HallucinationError, match="Hallucination detected"):
            verify_extraction(
                source_text="Product: iPhone",
                extracted_data={"product": "iPhone", "color": "blue"},
                raise_on_hallucination=True
            )
    
    def test_hallucination_error_attributes(self):
        """Test that HallucinationError has correct attributes."""
        try:
            verify_extraction(
                source_text="Product: iPhone",
                extracted_data={"product": "iPhone", "color": "blue"},
                raise_on_hallucination=True
            )
        except HallucinationError as e:
            assert "color" in e.flagged_fields
            assert e.confidence < 1.0


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


class TestWithGrounding:
    """Test the with_grounding decorator."""
    
    def test_decorator_passes_grounded(self):
        @with_grounding(source_text="Name: John, Age: 25", raise_on_failure=False)
        def extract():
            return {"name": "John", "age": 25}
        
        result = extract()
        assert result["name"] == "John"
    
    def test_decorator_raises_on_hallucination(self):
        @with_grounding(source_text="Name: John", raise_on_failure=True)
        def extract():
            return {"name": "John", "email": "john@fake.com"}
        
        with pytest.raises(HallucinationError):
            extract()


class TestGroundedExtractor:
    """Test the GroundedExtractor class."""
    
    def test_extractor_init(self):
        # Mock client
        class MockClient:
            pass
        
        extractor = GroundedExtractor(MockClient(), default_threshold=0.8)
        assert extractor.default_threshold == 0.8
        assert extractor.raise_on_hallucination is False


class TestVerificationResult:
    """Test VerificationResult features."""
    
    def test_to_dict(self):
        gc = GroundCheck()
        result = gc.verify(
            source_text="Test value",
            extracted_data={"field": "Test"}
        )
        d = result.to_dict()
        assert "overall_confidence" in d
        assert "flagged_fields" in d
        assert "field_results" in d
    
    def test_is_reliable(self):
        gc = GroundCheck()
        result = gc.verify(
            source_text="Name: John",
            extracted_data={"name": "John"}
        )
        assert result.is_reliable is True
        
        result2 = gc.verify(
            source_text="Name: John",
            extracted_data={"name": "John", "fake": "data"}
        )
        assert result2.is_reliable is False


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
    
    def test_medical_record(self):
        gc = GroundCheck()
        clinical_note = """
        Patient: John Smith, DOB 03/15/1980
        Chief Complaint: Chest pain for 2 days
        Vital Signs: BP 145/92, HR 88
        Assessment: Suspected angina
        Plan: Start aspirin 81mg daily
        """
        extracted = {
            "patient_name": "John Smith",
            "blood_pressure": "145/92",
            "heart_rate": 88,
            "medication": "aspirin 81mg",
            "surgery_scheduled": "Yes"  # Hallucinated
        }
        result = gc.verify(clinical_note, extracted)
        assert result.field_results["patient_name"].confidence > 0.9
        assert "surgery_scheduled" in result.flagged_fields


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_source(self):
        gc = GroundCheck()
        result = gc.verify(source_text="", extracted_data={"field": "value"})
        assert result.field_results["field"].flagged
    
    def test_empty_extracted(self):
        gc = GroundCheck()
        result = gc.verify(source_text="Some text", extracted_data={})
        assert result.overall_confidence == 0.0
    
    def test_special_characters(self):
        gc = GroundCheck()
        result = gc.verify(
            source_text="Email: test@example.com",
            extracted_data={"email": "test@example.com"}
        )
        assert result.field_results["email"].confidence > 0.9
    
    def test_unicode(self):
        gc = GroundCheck()
        result = gc.verify(
            source_text="Customer: José García",
            extracted_data={"customer": "José García"}
        )
        assert result.field_results["customer"].confidence > 0.9
    
    def test_nested_dict(self):
        gc = GroundCheck()
        result = gc.verify(
            source_text="City: New York, ZIP: 10001",
            extracted_data={"address": {"city": "New York", "zip": "10001"}}
        )
        assert result.field_results["address"].method == VerificationMethod.AGGREGATE
    
    def test_list_field(self):
        gc = GroundCheck()
        result = gc.verify(
            source_text="Items: Apple, Banana, Orange",
            extracted_data={"items": ["Apple", "Banana", "Orange"]}
        )
        assert result.field_results["items"].confidence > 0.5
