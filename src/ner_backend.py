"""Shared PII backend adapter built around NVIDIA GLiNER-PII."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import sys
from typing import Any, List, Optional


MODEL_NAME = "nvidia/gliner-pii"


def _canonical_label(label: Any) -> str:
    raw = str(label).strip().lower() if label is not None else ""
    out = []
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "/", ".", ":", "'"}:
            out.append("_")
    canonical = "".join(out)
    while "__" in canonical:
        canonical = canonical.replace("__", "_")
    return canonical.strip("_")


PII_LABEL_ALIASES = {
    "per": "NAME",
    "person": "NAME",
    "name": "NAME",
    "first_name": "NAME",
    "last_name": "NAME",
    "organization": "ORG",
    "organisation": "ORG",
    "company": "ORG",
    "org": "ORG",
    "location": "ADDRESS",
    "address": "ADDRESS",
    "street_address": "ADDRESS",
    "geo": "ADDRESS",
    "gpe": "ADDRESS",
    "loc": "ADDRESS",
    "email": "EMAIL",
    "email_address": "EMAIL",
    "phone": "PHONE",
    "phone_number": "PHONE",
    "mobile_phone_number": "PHONE",
    "landline_phone_number": "PHONE",
    "fax_number": "PHONE",
    "credit_card": "CREDIT_CARD",
    "credit_card_number": "CREDIT_CARD",
    "credit_card_brand": "CREDIT_CARD",
    "bank_account": "BANK_ACCOUNT",
    "bank_account_number": "BANK_ACCOUNT",
    "iban": "BANK_ACCOUNT",
    "swift": "BANK_ACCOUNT",
    "social_security_number": "SSN",
    "ssn": "SSN",
    "passport": "PASSPORT",
    "passport_number": "PASSPORT",
    "driver_license": "DRIVING_LICENSE",
    "drivers_license": "DRIVING_LICENSE",
    "driver_s_license_number": "DRIVING_LICENSE",
    "driving_license": "DRIVING_LICENSE",
    "national_id_number": "VOTER_ID",
    "voter_id": "VOTER_ID",
    "aadhaar": "AADHAR",
    "aadhar": "AADHAR",
    "pan": "PAN",
    "tax_identification_number": "PAN",
    "ip_address": "IP_ADDRESS",
    "username": "USERNAME",
    "user_name": "USERNAME",
    "password": "PASSWORD",
    "api_key": "API_KEY",
    "token": "API_KEY",
    "secret": "API_KEY",
    "device_id": "DEVICE_ID",
    "serial_number": "DEVICE_ID",
    "crypto_wallet": "CRYPTO_WALLET",
    "bitcoin_wallet": "CRYPTO_WALLET",
    "medical_id": "MEDICAL_ID",
    "health_insurance_id_number": "MEDICAL_ID",
    "health_insurance_number": "MEDICAL_ID",
    "date_of_birth": "DOB",
    "dob": "DOB",
}


MASKABLE_LABELS = {
    "NAME",
    "ORG",
    "ADDRESS",
    "EMAIL",
    "PHONE",
    "CREDIT_CARD",
    "BANK_ACCOUNT",
    "SSN",
    "PASSPORT",
    "DRIVING_LICENSE",
    "MEDICAL_ID",
    "VOTER_ID",
    "AADHAR",
    "PAN",
    "IP_ADDRESS",
    "USERNAME",
    "PASSWORD",
    "API_KEY",
    "DEVICE_ID",
    "CRYPTO_WALLET",
    "DOB",
    "MISC",
}


GLINER_QUERY_LABELS = [
    "Person",
    "Organization",
    "Location",
    "Address",
    "Email",
    "Phone Number",
    "Credit Card",
    "Bank Account",
    "Social Security Number",
    "Passport Number",
    "Driver's License Number",
    "Aadhaar",
    "Pan",
    "IP Address",
    "Username",
    "Password",
    "API Key",
    "Device ID",
    "Crypto Wallet",
    "Medical ID",
    "Voter ID",
    "Date of Birth",
]


@dataclass
class PIIEntity:
    text: str
    label_: str
    start_char: int
    end_char: int
    score: float = 0.0


@dataclass
class PIIDoc:
    text: str
    ents: List[PIIEntity]


class GLiNERPIIAdapter:
    """A small spaCy-like wrapper around the GLiNER PII model."""

    def __init__(self, model_name: str = MODEL_NAME):
        from gliner import GLiNER

        self.model_name = model_name
        self.model = GLiNER.from_pretrained(model_name)

    def __call__(self, text: str) -> PIIDoc:
        raw_entities = self.model.predict_entities(
            text,
            GLINER_QUERY_LABELS,
            threshold=0.35,
        )
        ents: List[PIIEntity] = []

        for item in raw_entities:
            label = normalize_pii_label(item.get("label") or item.get("entity_type") or "")
            if label not in MASKABLE_LABELS:
                continue

            start = int(item.get("start", 0))
            end = int(item.get("end", 0))
            score = float(item.get("score", 0.0))
            ents.append(
                PIIEntity(
                    text=text[start:end],
                    label_=label,
                    start_char=start,
                    end_char=end,
                    score=score,
                )
            )

        ents.sort(key=lambda ent: ent.start_char)
        return PIIDoc(text=text, ents=_merge_adjacent_entities(text, ents))


def _merge_adjacent_entities(text: str, entities: List[PIIEntity]) -> List[PIIEntity]:
    """Merge adjacent same-label entities to repair token-split spans."""

    if not entities:
        return []

    merged: List[PIIEntity] = [entities[0]]
    for ent in entities[1:]:
        prev = merged[-1]
        gap = ent.start_char - prev.end_char

        if prev.label_ == ent.label_ and 0 <= gap <= 1:
            merged[-1] = PIIEntity(
                text=text[prev.start_char : max(prev.end_char, ent.end_char)],
                label_=prev.label_,
                start_char=prev.start_char,
                end_char=max(prev.end_char, ent.end_char),
                score=max(prev.score, ent.score),
            )
        else:
            merged.append(ent)

    return merged


@lru_cache(maxsize=1)
def load_pii_backend(model_name: str = MODEL_NAME) -> Optional[GLiNERPIIAdapter]:
    """Load the GLiNER PII backend once and reuse it across calls."""

    try:
        return GLiNERPIIAdapter(model_name=model_name)
    except Exception as exc:
        print(f"[PII backend init failed] {exc}", file=sys.stderr)
        return None


def load_ner_backend(model_name: str = MODEL_NAME) -> Optional[GLiNERPIIAdapter]:
    """Backward-compatible alias for older imports."""

    return load_pii_backend(model_name=model_name)


def normalize_pii_label(label: Any) -> str:
    """Normalize backend labels to the project schema."""

    raw = _canonical_label(label)
    if raw in PII_LABEL_ALIASES:
        return PII_LABEL_ALIASES[raw]

    upper = str(label).strip().upper() if label is not None else ""
    direct_aliases = {
        "PERSON": "NAME",
        "NAME": "NAME",
        "ORG": "ORG",
        "ORGANIZATION": "ORG",
        "ADDRESS": "ADDRESS",
        "LOCATION": "ADDRESS",
        "EMAIL": "EMAIL",
        "PHONE": "PHONE",
        "PHONE_NUMBER": "PHONE",
        "CREDIT_CARD": "CREDIT_CARD",
        "BANK_ACCOUNT": "BANK_ACCOUNT",
        "SSN": "SSN",
        "PASSPORT": "PASSPORT",
        "DRIVING_LICENSE": "DRIVING_LICENSE",
        "AADHAR": "AADHAR",
        "PAN": "PAN",
        "IP_ADDRESS": "IP_ADDRESS",
        "USERNAME": "USERNAME",
        "PASSWORD": "PASSWORD",
        "API_KEY": "API_KEY",
        "DEVICE_ID": "DEVICE_ID",
        "CRYPTO_WALLET": "CRYPTO_WALLET",
        "DOB": "DOB",
        "MISC": "MISC",
    }
    return direct_aliases.get(upper, upper)


normalize_ner_label = normalize_pii_label
