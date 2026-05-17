"""Synthetic dataset generators for privacy-focused multi-agent experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class SampleBuilder:
    """Utility for building text samples while tracking entity offsets."""

    def __init__(self) -> None:
        self._parts: List[str] = []
        self._entities: List[Dict[str, Any]] = []
        self._length = 0

    def add(self, text: str) -> None:
        self._parts.append(text)
        self._length += len(text)

    def entity(self, label: str, value: str) -> None:
        start = self._length
        self._parts.append(value)
        self._length += len(value)
        self._entities.append(
            {
                "label": label,
                "start": start,
                "end": self._length,
                "value": value,
            }
        )

    def build(self) -> Tuple[str, List[Dict[str, Any]]]:
        return "".join(self._parts), list(self._entities)


_FIRST_NAMES = [
    "Ava",
    "Noah",
    "Mia",
    "Ethan",
    "Sophia",
    "Liam",
    "Olivia",
    "Jacob",
    "Emma",
    "Arjun",
    "Sara",
    "Ravi",
]

_LAST_NAMES = [
    "Patel",
    "Sharma",
    "Johnson",
    "Williams",
    "Brown",
    "Mehta",
    "Khan",
    "Clark",
    "Gupta",
    "Lee",
]

_CITIES = [
    "London",
    "Delhi",
    "Bengaluru",
    "Mumbai",
    "Manchester",
    "Chicago",
    "Pune",
    "Boston",
]

_DOMAINS = [
    "example.com",
    "mailbox.org",
    "securemail.net",
    "privacyguard.io",
]

_DEVICE_PREFIXES = ["LAP", "SRV", "TAB", "MOB"]


def _make_phone(index: int, seed: int) -> str:
    base = seed * 1000 + index * 37
    digits = f"{base:010d}"[-10:]
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"


def _make_aadhar(index: int, seed: int) -> str:
    value = seed * 100000 + index * 271
    digits = f"{value:012d}"[-12:]
    return f"{digits[:4]}-{digits[4:8]}-{digits[8:]}"


def _make_pan(index: int, seed: int) -> str:
    letters = "ABCDEPQRSX"
    prefix = "".join(letters[(seed + index + i) % len(letters)] for i in range(5))
    digits = f"{seed * 17 + index * 13:04d}"[-4:]
    suffix = letters[(seed + index * 2) % len(letters)]
    return f"{prefix}{digits}{suffix}"


def _make_bank_account(index: int, seed: int) -> str:
    value = seed * 123456 + index * 9973
    return f"{value:012d}"[-12:]


def _make_passport(index: int, seed: int) -> str:
    value = seed * 73 + index * 19
    return f"P{value:07d}"


def _make_driving_license(index: int, seed: int) -> str:
    value = seed * 91 + index * 23
    return f"DL-{value:010d}"[-13:]


def _make_ip(index: int, seed: int) -> str:
    octets = [
        10 + ((seed + index) % 200),
        (seed * 3 + index * 7) % 255,
        (seed * 5 + index * 11) % 255,
        10 + (index * 17) % 240,
    ]
    return ".".join(str(o) for o in octets)


def _make_api_key(index: int, seed: int) -> str:
    return f"sk_live_{seed:02d}{index:02d}{(seed * 17 + index * 29):08d}"


def _make_wallet(index: int, seed: int) -> str:
    value = f"{seed:04x}{index:04x}{(seed * 31 + index * 11):032x}"
    return "0x" + value[:40]


def _make_username(first: str, index: int) -> str:
    return f"{first.lower()}_{index:02d}"


def _finalize_sample(
    builder: SampleBuilder,
    *,
    dataset_name: str,
    difficulty: str,
    scenario: str,
) -> Dict[str, Any]:
    text, entities = builder.build()
    return {
        "text": text,
        "entities": entities,
        "dataset_name": dataset_name,
        "difficulty": difficulty,
        "scenario": scenario,
    }


def _build_contact_sample(index: int, seed: int) -> Dict[str, Any]:
    first = _FIRST_NAMES[index % len(_FIRST_NAMES)]
    last = _LAST_NAMES[(index + seed) % len(_LAST_NAMES)]
    city = _CITIES[(index + seed) % len(_CITIES)]
    domain = _DOMAINS[(index + seed) % len(_DOMAINS)]
    email = f"{first.lower()}.{last.lower()}.{index:02d}@{domain}"
    phone = _make_phone(index, seed)

    builder = SampleBuilder()
    builder.add("Customer ")
    builder.entity("NAME", f"{first} {last}")
    builder.add(" can be reached at ")
    builder.entity("EMAIL", email)
    builder.add(" or ")
    builder.entity("PHONE", phone)
    builder.add(f" from the {city} office regarding service updates.")
    return _finalize_sample(
        builder,
        dataset_name="contacts_low",
        difficulty="low",
        scenario="contact",
    )


def _build_chat_sample(index: int, seed: int) -> Dict[str, Any]:
    first = _FIRST_NAMES[(index + 1) % len(_FIRST_NAMES)]
    second = _FIRST_NAMES[(index + 4) % len(_FIRST_NAMES)]
    domain = _DOMAINS[(index + seed) % len(_DOMAINS)]
    email = f"{first.lower()}.{second.lower()}.{index:02d}@{domain}"
    phone = _make_phone(index + 3, seed)
    user = _make_username(first, index)

    builder = SampleBuilder()
    builder.add("Chat transcript:\n")
    builder.entity("NAME", first)
    builder.add(": Please message me at ")
    builder.entity("EMAIL", email)
    builder.add(".\n")
    builder.entity("NAME", second)
    builder.add(": Sure, my backup number is ")
    builder.entity("PHONE", phone)
    builder.add(".\n")
    builder.add("System note: handle ")
    builder.entity("USERNAME", user)
    builder.add(" is active for this conversation.")
    return _finalize_sample(
        builder,
        dataset_name="chat_transcripts",
        difficulty="medium",
        scenario="chat",
    )


def _build_medical_sample(index: int, seed: int) -> Dict[str, Any]:
    first = _FIRST_NAMES[(index + 2) % len(_FIRST_NAMES)]
    last = _LAST_NAMES[(index + 3) % len(_LAST_NAMES)]
    ssn = f"{100 + (seed + index) % 900:03d}-{10 + (index * 7) % 90:02d}-{1000 + (seed * 13 + index * 17) % 9000:04d}"
    aadhar = _make_aadhar(index, seed)
    passport = _make_passport(index, seed)

    builder = SampleBuilder()
    builder.add("Clinical note: patient ")
    builder.entity("NAME", f"{first} {last}")
    builder.add(" presented with prior SSN ")
    builder.entity("SSN", ssn)
    builder.add(", Aadhaar ")
    builder.entity("AADHAR", aadhar)
    builder.add(", and passport ")
    builder.entity("PASSPORT", passport)
    builder.add(" for verification.")
    return _finalize_sample(
        builder,
        dataset_name="medical_notes",
        difficulty="high",
        scenario="medical",
    )


def _build_security_sample(index: int, seed: int) -> Dict[str, Any]:
    first = _FIRST_NAMES[(index + 5) % len(_FIRST_NAMES)]
    last = _LAST_NAMES[(index + 1) % len(_LAST_NAMES)]
    ip = _make_ip(index, seed)
    api_key = _make_api_key(index, seed)
    wallet = _make_wallet(index, seed)
    device = f"{_DEVICE_PREFIXES[index % len(_DEVICE_PREFIXES)]}-{seed:02d}{index:06d}"
    username = _make_username(first, index)

    builder = SampleBuilder()
    builder.add("Security log: ")
    builder.entity("IP_ADDRESS", ip)
    builder.add(" accepted login for user ")
    builder.entity("USERNAME", username)
    builder.add(". Device ")
    builder.entity("DEVICE_ID", device)
    builder.add(" reported API key ")
    builder.entity("API_KEY", api_key)
    builder.add(" and wallet ")
    builder.entity("CRYPTO_WALLET", wallet)
    builder.add(f" during review by {first} {last}.")
    return _finalize_sample(
        builder,
        dataset_name="security_logs",
        difficulty="high",
        scenario="security",
    )


def _build_compliance_sample(index: int, seed: int) -> Dict[str, Any]:
    first = _FIRST_NAMES[(index + 6) % len(_FIRST_NAMES)]
    last = _LAST_NAMES[(index + 2) % len(_LAST_NAMES)]
    pan = _make_pan(index, seed)
    bank = _make_bank_account(index, seed)
    dl = _make_driving_license(index, seed)
    phone = _make_phone(index + 7, seed)

    builder = SampleBuilder()
    builder.add("Compliance case for ")
    builder.entity("NAME", f"{first} {last}")
    builder.add(" includes PAN ")
    builder.entity("PAN", pan)
    builder.add(", bank account ")
    builder.entity("BANK_ACCOUNT", bank)
    builder.add(", driving license ")
    builder.entity("DRIVING_LICENSE", dl)
    builder.add(", and contact ")
    builder.entity("PHONE", phone)
    builder.add(" on file.")
    return _finalize_sample(
        builder,
        dataset_name="compliance_records",
        difficulty="medium",
        scenario="compliance",
    )


def generate_research_datasets(sample_count: int = 20, seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """Generate several privacy-oriented datasets with varied difficulty."""

    datasets: Dict[str, List[Dict[str, Any]]] = {
        "contacts_low": [],
        "chat_transcripts": [],
        "medical_notes": [],
        "security_logs": [],
        "compliance_records": [],
    }

    for i in range(sample_count):
        datasets["contacts_low"].append(_build_contact_sample(i, seed))
        datasets["chat_transcripts"].append(_build_chat_sample(i, seed))
        datasets["medical_notes"].append(_build_medical_sample(i, seed))
        datasets["security_logs"].append(_build_security_sample(i, seed))
        datasets["compliance_records"].append(_build_compliance_sample(i, seed))

    return datasets


__all__ = ["generate_research_datasets"]
