---
title: Payment Data Extraction - Transaction Parsing with Instructor
description: Extract structured payment data from unstructured text using Instructor and Pydantic. Parse transaction details, multi-currency amounts, and payment methods.
---

# Extracting Payment Data from Unstructured Text

This guide demonstrates how to extract structured payment information from unstructured text such as emails, invoices, and bank statements. Using Instructor with Pydantic validation, we ensure the extracted data is accurate and consistent.

## Defining Payment Models

We start by defining Pydantic models that represent payment data. The `PaymentMethod` enum captures common payment types, while `Transaction` includes validation for currency codes and amounts.

```python
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class PaymentMethod(str, Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    DIGITAL_WALLET = "digital_wallet"
    CASH = "cash"
    OTHER = "other"


class Transaction(BaseModel):
    merchant_name: str = Field(description="Name of the merchant or payee")
    amount: float = Field(description="Transaction amount", gt=0)
    currency: str = Field(description="ISO 4217 currency code (e.g., USD, EUR, BRL)")
    payment_method: PaymentMethod = Field(description="Payment method used")
    reference_id: str | None = Field(
        default=None, description="Transaction reference or confirmation number"
    )
    date: str | None = Field(
        default=None, description="Transaction date in YYYY-MM-DD format"
    )

    @field_validator("currency")
    @classmethod
    def validate_currency_code(cls, v: str) -> str:
        v = v.upper().strip()
        if len(v) != 3 or not v.isalpha():
            raise ValueError(f"Invalid ISO 4217 currency code: {v}")
        return v
```

## Extracting Multiple Transactions

Payment documents often contain multiple transactions. The `PaymentReport` model aggregates them and validates that the total matches the sum of individual amounts (when all transactions share the same currency).

```python
from pydantic import model_validator


class PaymentReport(BaseModel):
    transactions: list[Transaction] = Field(
        description="List of payment transactions found in the text"
    )
    total_amount: float | None = Field(
        default=None, description="Total amount if stated in the document"
    )

    @model_validator(mode="after")
    def validate_total(self):
        if self.total_amount is None:
            return self

        currencies = {t.currency for t in self.transactions}
        if len(currencies) == 1:
            calculated = sum(t.amount for t in self.transactions)
            if abs(calculated - self.total_amount) > 0.01:
                raise ValueError(
                    f"Total {self.total_amount} does not match "
                    f"sum of transactions {calculated}"
                )
        return self
```

## Processing Payment Text

The `extract_payments` function uses Instructor to parse unstructured payment text into our structured models. Instructor handles retries automatically when validation fails, asking the LLM to correct its output.

```python
import instructor

# <%hide%>
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


class PaymentMethod(str, Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    DIGITAL_WALLET = "digital_wallet"
    CASH = "cash"
    OTHER = "other"


class Transaction(BaseModel):
    merchant_name: str = Field(description="Name of the merchant or payee")
    amount: float = Field(description="Transaction amount", gt=0)
    currency: str = Field(description="ISO 4217 currency code (e.g., USD, EUR, BRL)")
    payment_method: PaymentMethod = Field(description="Payment method used")
    reference_id: str | None = Field(
        default=None, description="Transaction reference or confirmation number"
    )
    date: str | None = Field(
        default=None, description="Transaction date in YYYY-MM-DD format"
    )

    @field_validator("currency")
    @classmethod
    def validate_currency_code(cls, v: str) -> str:
        v = v.upper().strip()
        if len(v) != 3 or not v.isalpha():
            raise ValueError(f"Invalid ISO 4217 currency code: {v}")
        return v


class PaymentReport(BaseModel):
    transactions: list[Transaction] = Field(
        description="List of payment transactions found in the text"
    )
    total_amount: float | None = Field(
        default=None, description="Total amount if stated in the document"
    )

    @model_validator(mode="after")
    def validate_total(self):
        if self.total_amount is None:
            return self
        currencies = {t.currency for t in self.transactions}
        if len(currencies) == 1:
            calculated = sum(t.amount for t in self.transactions)
            if abs(calculated - self.total_amount) > 0.01:
                raise ValueError(
                    f"Total {self.total_amount} does not match "
                    f"sum of transactions {calculated}"
                )
        return self


# <%hide%>
from openai import OpenAI

client = instructor.from_openai(OpenAI())


def extract_payments(text: str) -> PaymentReport:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=PaymentReport,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract all payment transactions from the provided text. "
                    "For each transaction, identify the merchant, amount, currency, "
                    "payment method, reference ID if available, and date if available."
                ),
            },
            {"role": "user", "content": text},
        ],
        max_retries=3,
    )
```

## Example Usage

Here's how to extract payment data from a bank statement email:

```python
statement = """
Your monthly statement for January 2026:

01/05 - Amazon.com, Visa ending 4532, $127.99 (Ref: AMZ-20260105-7821)
01/08 - Uber Technologies, Mastercard ending 8901, $23.50 (Ref: UBER-8834)
01/12 - Wire transfer to Acme Corp, $2,500.00 (Ref: WIRE-2026-0112)
01/15 - PayPal payment to Freelancer.com, $450.00 (Ref: PP-FL-9921)
01/20 - Starbucks, Apple Pay, $6.75

Total charges: $3,108.24
"""

report = extract_payments(statement)

for txn in report.transactions:
    print(f"{txn.date} | {txn.merchant_name:20s} | {txn.currency} {txn.amount:>10.2f} | {txn.payment_method.value}")
```

Expected output:

```
2026-01-05 | Amazon.com           | USD     127.99 | credit_card
2026-01-08 | Uber Technologies    | USD      23.50 | credit_card
2026-01-12 | Acme Corp            | USD    2500.00 | bank_transfer
2026-01-15 | Freelancer.com       | USD     450.00 | digital_wallet
2026-01-20 | Starbucks            | USD       6.75 | digital_wallet
```

## Multi-Currency Support

The currency validator ensures ISO 4217 compliance, and the total validator correctly handles multi-currency reports by skipping the total check when multiple currencies are present:

```python
multi_currency_text = """
International expense report:
- Hotel in Tokyo: JPY 45,000 (credit card)
- Flight to London: EUR 320.00 (credit card)
- Taxi in New York: USD 42.50 (debit card)
"""

report = extract_payments(multi_currency_text)
# total_amount will be None since currencies differ
```

## Key Takeaways

1. **Pydantic validators catch errors early** — Currency codes and amounts are validated before the data reaches your application.
2. **Instructor retries on validation failure** — If the LLM returns an invalid currency code, Instructor asks it to fix the output automatically.
3. **Domain enums improve accuracy** — Using `PaymentMethod` as an enum constrains the LLM to valid payment types instead of free-form text.
4. **Cross-field validation** — The `validate_total` model validator ensures consistency between individual transactions and the reported total.
