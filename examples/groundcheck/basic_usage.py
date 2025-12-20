"""
GroundCheck: Detect Hallucinations in LLM Extractions

This example demonstrates how to use GroundCheck to verify that
LLM-extracted data is actually present in the source text.
"""

from instructor import GroundCheck, verify_extraction, HallucinationError


def example_basic():
    """Basic verification example."""
    print("=" * 60)
    print("Example 1: Basic Verification")
    print("=" * 60)
    
    source_text = """
    INVOICE
    Invoice No: INV-2024-0892
    Date: December 15, 2024
    Customer: Acme Corporation
    Total Due: $593.89
    """
    
    extracted_data = {
        "invoice_number": "INV-2024-0892",
        "customer": "Acme Corporation",
        "total": 593.89,
        "payment_terms": "Net 30",  # HALLUCINATED - not in source!
    }
    
    result = verify_extraction(source_text, extracted_data)
    
    print(f"Overall Confidence: {result.overall_confidence:.1%}")
    print(f"Is Reliable: {result.is_reliable}")
    print(f"Flagged Fields: {result.flagged_fields}")
    print()
    
    for field, res in result.field_results.items():
        status = "❌ HALLUCINATED" if res.flagged else "✅ Grounded"
        print(f"  {field}: {res.confidence:.0%} ({res.method.value}) - {status}")


def example_raise_on_hallucination():
    """Example of raising exception on hallucination."""
    print("\n" + "=" * 60)
    print("Example 2: Raise on Hallucination")
    print("=" * 60)
    
    try:
        verify_extraction(
            source_text="Product: iPhone, Price: $999",
            extracted_data={"product": "iPhone", "color": "blue"},
            raise_on_hallucination=True
        )
    except HallucinationError as e:
        print(f"Caught HallucinationError!")
        print(f"  Flagged fields: {e.flagged_fields}")
        print(f"  Confidence: {e.confidence:.1%}")


def example_custom_thresholds():
    """Example with custom confidence thresholds."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Thresholds")
    print("=" * 60)
    
    gc = GroundCheck(
        confidence_threshold=0.8,  # Stricter threshold
        fuzzy_threshold=0.9,       # Require better fuzzy matches
    )
    
    result = gc.verify(
        source_text="Contact: John Smith, Phone: 555-1234",
        extracted_data={
            "name": "John Smith",
            "phone": "555-1234",
            "email": "john@example.com"  # Hallucinated
        }
    )
    
    print(f"Flagged: {result.flagged_fields}")


def example_medical_record():
    """Real-world example: Medical record extraction."""
    print("\n" + "=" * 60)
    print("Example 4: Medical Record (High Stakes)")
    print("=" * 60)
    
    clinical_note = """
    Patient: John Smith, DOB 03/15/1980
    Chief Complaint: Chest pain for 2 days
    Vital Signs: BP 145/92, HR 88
    Assessment: Suspected angina
    Plan: Start aspirin 81mg daily, follow up in 1 week
    """
    
    extracted = {
        "patient_name": "John Smith",
        "blood_pressure": "145/92",
        "heart_rate": 88,
        "diagnosis": "angina",
        "medication": "aspirin 81mg",
        # These are HALLUCINATED - dangerous in medical context!
        "allergies": "Penicillin",
        "surgery_scheduled": "Yes",
    }
    
    result = verify_extraction(clinical_note, extracted)
    
    print("⚠️  Medical extraction results:")
    for field, res in result.field_results.items():
        if res.flagged:
            print(f"  🚨 {field}: HALLUCINATED - verify manually!")
        else:
            print(f"  ✅ {field}: Verified in source")


if __name__ == "__main__":
    example_basic()
    example_raise_on_hallucination()
    example_custom_thresholds()
    example_medical_record()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
