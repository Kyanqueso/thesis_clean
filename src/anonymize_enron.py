#!/usr/bin/env python
"""De-identify the AESLC email pool before it reaches any dataset.

The emails are public (FERC released the Enron corpus; AESLC is the curated
subset of it), so nothing here can stop someone finding an original — the
original is public. What this does control is what THIS project stores, embeds,
displays and hands around: no names, contact details or identifying numbers in
data/, in the embeddings, on the dashboard, or in the thesis.

Order of operations, and why:

  1. residue sweep — attachment stubs and trailing legal disclaimers. These are
     boilerplate the sender did not write, and the disclaimer block is dense
     with switchboard numbers and office addresses. Cutting it first means
     Presidio never has to be right about it.
  2. Presidio — PERSON / EMAIL / PHONE / URL (the direct identifiers the ethics
     protocol names) plus SSN, credit card and IBAN, and three recognizers the
     stock engine has no idea about:
       ENRON_INTERNAL_ADDRESS   Name/HOU/ECT@ECT  (name + business unit)
       PHONE_EXTENSION          X35968, ext. 1234 (direct-dial reachability)
       STREET_ADDRESS           number + street type, PO boxes
  3. sign-off pass — a name alone on the line after "Thanks," or behind a dash.
     NER reads those weakly because there is no sentence around them.

Every detected span becomes a typed placeholder (<PERSON>, <EMAIL>, ...) rather
than being deleted: the text keeps its shape, and a reader can see WHAT was
removed, which is what makes the leak audit meaningful.

LOCATION and DATE_TIME are off by default (--locations, --dates). Neither is a
direct identifier, both fire constantly in business mail, and scrubbing every
date would gut the text. Turn them on if the ethics reviewer asks for
HIPAA-Safe-Harbor-style date removal.

Usage:
    python src/anonymize_enron.py audit                 # full 7,000-doc pool
    python src/anonymize_enron.py audit --limit 300     # quick pass
    python src/anonymize_enron.py audit --samples 25 --out scrubbed_samples.txt
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- Patterns -------------------------------------------------------------
# Shared by the scrubber and the audit, so the thing that removes an artefact
# and the thing that reports on it can never drift apart.

P_ENRON_INTERNAL_STRICT = re.compile(
    r"\b[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3}\s*/[A-Z]{2,6}/[A-Z]{2,6}(?:@[A-Z]{2,6})?")
P_ENRON_INTERNAL_LOOSE = re.compile(
    r"\b[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3}\s*/[A-Z]{2,6}@[A-Z]{2,6}")
P_EXTENSION = re.compile(r"\b(?:x|ext\.?)\s?\d{3,6}\b", re.I)
# Presidio's phone recognizer is region-configured and misses +44-style numbers
# written with spaces; one survived the first 300-email pass.
P_INTL_PHONE = re.compile(r"\+\d{1,3}(?:[\s.\-]?\(?\d{1,4}\)?){2,5}")
# Anything phone-SHAPED, whether or not libphonenumber likes it. Presidio missed
# these inside contact lists, where the surrounding text gives it no signal.
P_NANP_PHONE = re.compile(r"\(?\b\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b")
# quoted-printable soft breaks split an address across lines ("user@host.=\ncom")
P_SOFT_BREAK = re.compile(r"=\r?\n")
# AESLC's own cleaning turned some soft breaks into "= " mid-token, leaving
# "user@corp.enron.= com". Rejoined only INSIDE an address, so ordinary prose
# like "total= 5" is untouched.
P_QP_SPLIT_EMAIL = re.compile(r"([\w.+-]+@[\w.-]*?)=\s+(?=\w)")
# Presidio's stock URL recognizer treats "lemons...Try" as a hostname and ate
# prose around ellipses. These require a scheme or a real TLD.
P_URL_SCHEME = re.compile(r"(?:https?://|ftp://|www\.)[\w./?=&%#:+~@-]+", re.I)
P_URL_BARE = re.compile(
    r"\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|edu|gov|mil|int|io|co|uk|de|fr|ca|au"
    r"|biz|info|tv|us)\b(?:/[\w./?=&%#:+~@-]*)?", re.I)
P_HTML_TAG = re.compile(r"<[^<>\n]{1,200}>")
P_HTML_ISH = re.compile(r"(?i)<(?:font|br|html|body|table|div|p|span)\b")
# RFC display name: "Lastname, Firstname" <addr>. NER catches the surname and
# leaves the given name sitting inside the quotes.
P_DISPLAY_NAME = re.compile(r"\"[^\"\n]{1,60}\"(?=\s*<)")
P_ATTACHMENT_STUB = re.compile(r"<<[^<>\n]{1,160}>>")
P_STREET = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][\w.'-]+\s+){0,3}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|"
    r"Circle|Cir|Way|Place|Pl|Parkway|Pkwy|Highway|Hwy|Suite|Ste|Floor)\b\.?",
    re.I)
P_PO_BOX = re.compile(r"\bP\.?\s?O\.?\s*Box\s+\d+\b", re.I)

# A disclaimer is only cut when it opens its own line AND the block that follows
# actually talks like one. An email whose subject is confidentiality keeps its body.
P_DISCLAIMER_OPEN = re.compile(
    r"(?im)^[ \t>*_-]*(?:"
    r"(?:this|the)\s+(?:e-?mail|message|transmission|communication|information)\b"
    r"|confidentiality\s+notice"
    r"|notice\s*:"
    r"|\*+\s*confidential"
    r")")
P_DISCLAIMER_CONFIRM = re.compile(
    r"(?i)(confidential|intended\s+recipient|privileged|received\s+this\s+in\s+error"
    r"|unauthorized\s+(?:use|review|disclosure)|delete\s+(?:this|the)\s+(?:e-?mail|message))")
# Disclaimers also run inline, wrapped into a paragraph with no line of their
# own. Only these give-away phrases qualify — a bare "confidential" is ordinary
# business prose and cutting on it would truncate real emails.
P_DISCLAIMER_INLINE = re.compile(
    r"(?i)(intended\s+recipient"
    r"|received\s+this\s+(?:e-?mail\s+|message\s+)?in\s+error"
    r"|unauthorized\s+(?:use|review|disclosure|copying|dissemination)"
    r"|notify\s+the\s+sender\s+immediately)")

# The audit table's row definitions, verbatim in intent: each entry is
# (label, compiled pattern). Reported as rows-containing and total occurrences.
AUDIT_PATTERNS: dict[str, re.Pattern] = {
    "email address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "enron.com address": re.compile(r"[\w.+-]+@enron\.com", re.I),
    "phone number": re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "URL": re.compile(r"(?:https?://|www\.)[\w./?=&%#-]+", re.I),
    "internal address (strict)": P_ENRON_INTERNAL_STRICT,
    "internal address (loose)": P_ENRON_INTERNAL_LOOSE,
    "phone extension": P_EXTENSION,
    "attachment stub": P_ATTACHMENT_STUB,
    "confidentiality wording": re.compile(r"(?i)confidential"),
    "'intended recipient'": re.compile(r"(?i)intended\s+recipient"),
    "'received in error'": re.compile(r"(?i)received\s+this\s+in\s+error"),
    # a quoted header BLOCK (two or more header lines together) — a lone
    # "Subject: ..." line is usually the author labelling their own message,
    # and any name in it is scrubbed like any other
    "quoted header block": re.compile(
        r"(?im)^\s*(?:From|Sent|To|Cc|Bcc|Subject)\s*:.*\r?\n\s*(?:From|Sent|To|Cc|Bcc|Subject)\s*:"),
    # the separator form, not the words "original message" in a sentence
    "'Original Message' bar": re.compile(r"(?im)^[\s>]*-{2,}\s*original\s+message\s*-{2,}"),
    "'Forwarded by'": re.compile(r"(?i)forwarded\s+by"),
    "street address": P_STREET,
    "PO box": P_PO_BOX,
    "SSN-shaped": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card-shaped (Luhn)": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
}


def _luhn(value: str) -> bool:
    """Checksum every real card number satisfies.

    Without it, "card-shaped" counts any sixteen digits in groups of four —
    deal numbers, column-aligned tables — and the count went UP after scrubbing
    purely because collapsing runs of spaces pushed such columns together. A
    false alarm in a leak report is not a safe error: it trains the reader to
    ignore the row that matters.
    """
    digits = [int(c) for c in value if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# label -> extra test a raw regex hit must also pass to count
VALIDATORS = {"card-shaped (Luhn)": _luhn}

# Two different questions, so two different totals. An IDENTIFIER left in the
# text is a privacy failure. RESIDUE is boilerplate the sender did not write:
# worth removing for data quality, but once its names and addresses are
# placeholders it identifies nobody, and counting it as a leak would bury the
# rows that matter.
IDENTIFIER_LABELS = {
    "email address", "enron.com address", "phone number", "URL",
    "internal address (strict)", "internal address (loose)", "phone extension",
    "street address", "PO box", "SSN-shaped", "card-shaped (Luhn)",
}

# Entity -> placeholder. Everything identifying collapses to a small vocabulary,
# so the scrubbed text reads consistently and the audit can count placeholders.
PLACEHOLDER = {
    "PERSON": "<PERSON>",
    "ENRON_INTERNAL_ADDRESS": "<PERSON>",
    "EMAIL_ADDRESS": "<EMAIL>",
    "PHONE_NUMBER": "<PHONE>",
    "PHONE_EXTENSION": "<PHONE>",
    "URL": "<URL>",
    "URL_STRICT": "<URL>",
    "STREET_ADDRESS": "<ADDRESS>",
    "LOCATION": "<LOCATION>",
    "DATE_TIME": "<DATE>",
    "US_SSN": "<ID_NUMBER>",
    "CREDIT_CARD": "<ID_NUMBER>",
    "IBAN_CODE": "<ID_NUMBER>",
    "US_PASSPORT": "<ID_NUMBER>",
    "US_BANK_NUMBER": "<ID_NUMBER>",
}
BASE_ENTITIES = (
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "URL_STRICT",
    "US_SSN", "CREDIT_CARD", "IBAN_CODE", "US_PASSPORT",
    "ENRON_INTERNAL_ADDRESS", "PHONE_EXTENSION", "STREET_ADDRESS",
)
# US_DRIVER_LICENSE and US_BANK_NUMBER are deliberately absent from the default
# set: both match ordinary alphanumeric tokens at low confidence and would redact
# deal numbers and ordinary words, which is over-scrubbing dressed up as safety.

SCORE_THRESHOLD = 0.35

# name alone on the line after a sign-off, or behind a leading dash
P_SIGNOFF = re.compile(
    r"(?im)^(?:thanks|thank\s+you|regards|best\s+regards|best|sincerely|cheers|"
    r"yours|take\s+care)[,!.]*[ \t]*\r?\n+([^\n]{1,40})$")
P_DASH_NAME = re.compile(r"(?m)^[ \t]*[-–—]{1,2}[ \t]*([A-Za-z][\w.'-]{1,20})[ \t]*$")
# every token capitalised: "Phillip K Allen" is a sign-off, "Let me know" is not
P_NAMEY = re.compile(r"^[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,2}[.!]?$")
# "Dear Richard", "Hi Sophie" — a greeting is the one place a first name appears
# with no sentence around it, and NER reads it weakly. Not line-anchored: these
# turn up mid-line after a date header.
P_GREETING = re.compile(
    r"(?i)\b(?:dear|hi|hello|hey|good\s+(?:morning|afternoon|evening))[ ,]+"
    r"([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,2})")
# "Sophie's admission" — the possessive strips the sentence cues NER relies on.
# Over-scrubs the odd organisation ("Tulane's"), which costs a word and leaks
# nothing.
P_POSSESSIVE = re.compile(r"\b([A-Z][a-z]{2,15})['’]s\b")

# Words that follow a greeting but name nobody, and given names that are
# ordinary English. Kept deliberately short: over-scrubbing a real name is
# cheap, under-scrubbing one is the failure this module exists to prevent.
NAME_STOPLIST = {
    "all", "friends", "family", "team", "everyone", "everybody", "folks", "sir",
    "madam", "colleagues", "customer", "customers", "guys", "there", "gentlemen",
    "ladies", "group", "sirs", "member", "members",
    # calendar and common nouns that are also given names
    "april", "may", "june", "july", "august", "march", "january", "monday",
    "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "bill", "mark", "rich", "grant", "will", "art", "chuck", "frank", "jack",
    "don", "rob", "buck", "sue", "gene", "hope", "faith", "max", "ray", "gay",
    "enron", "houston", "texas",
    # sentence-initial words that take an apostrophe-s and are not people —
    # "Here's", "People's Corp" were both redacted before these were listed
    "here", "there", "this", "that", "what", "who", "where", "when", "it",
    "let", "today", "tomorrow", "yesterday", "tonight", "everyone", "someone",
    "anyone", "nobody", "people", "company", "corp", "inc", "board", "committee",
    "council", "management", "government", "state", "city", "school", "office",
    "university", "department", "market", "bank", "client", "year", "week",
    "month", "day", "night", "morning", "afternoon", "evening", "one", "each",
    "new", "our", "your", "his", "her", "their", "the", "and", "but", "for",
}

_ANALYZER = None


def _analyzer():
    """Presidio, built once. Import is lazy: the audit's pattern counts and the
    dataset builder's --no-deidentify path should not pay for loading spaCy."""
    global _ANALYZER
    if _ANALYZER is not None:
        return _ANALYZER
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

    engine = AnalyzerEngine()
    custom = [
        ("ENRON_INTERNAL_ADDRESS", "enron_internal", P_ENRON_INTERNAL_STRICT.pattern, 0.75),
        ("PHONE_EXTENSION", "phone_extension", P_EXTENSION.pattern, 0.6),
        ("PHONE_NUMBER", "intl_phone", P_INTL_PHONE.pattern, 0.6),
        ("PHONE_NUMBER", "nanp_phone", P_NANP_PHONE.pattern, 0.55),
        ("PERSON", "display_name", P_DISPLAY_NAME.pattern, 0.7),
        ("URL_STRICT", "url_scheme", P_URL_SCHEME.pattern, 0.7),
        ("URL_STRICT", "url_bare", P_URL_BARE.pattern, 0.6),
        ("STREET_ADDRESS", "street_address", P_STREET.pattern, 0.6),
        ("STREET_ADDRESS", "po_box", P_PO_BOX.pattern, 0.7),
    ]
    for entity, name, regex, score in custom:
        engine.registry.add_recognizer(PatternRecognizer(
            supported_entity=entity, name=name,
            patterns=[Pattern(name=name, regex=regex, score=score)]))
    _ANALYZER = engine
    return engine


# --- Step 1: residue sweep ------------------------------------------------

def clean_residue(text: str) -> str:
    """Drop what the sender did not write: attachment stubs and a trailing
    legal disclaimer. AESLC already removed most of this at source (0.4% of
    rows still carry a disclaimer, 1.6% an attachment stub), so this is a
    sweep, not a rewrite."""
    # rejoin quoted-printable soft breaks first: an address split as
    # "name@host.=\ncom" is invisible to every downstream recognizer
    text = P_SOFT_BREAK.sub("", text)
    text = P_QP_SPLIT_EMAIL.sub(r"\1", text)
    if P_HTML_ISH.search(text):      # a handful of bodies are HTML mail
        text = P_HTML_TAG.sub(" ", text)
    text = P_ATTACHMENT_STUB.sub(" ", text)
    for m in P_DISCLAIMER_OPEN.finditer(text):
        tail = text[m.start():]
        if P_DISCLAIMER_CONFIRM.search(tail[:400]):
            text = text[:m.start()]
            break

    # inline variant: no line of its own, so fall back to the sentence carrying
    # the phrase. Restricted to the tail of the message — a disclaimer is always
    # last, while an email that DISCUSSES one mentions it anywhere.
    # 0.35, not "the tail": several disclaimers sit a third of the way in,
    # after a short covering note. A body that is mostly disclaimer ends up
    # under the length floor and is dropped by the caller, which is correct.
    m = P_DISCLAIMER_INLINE.search(text)
    if m and m.start() > 0.35 * len(text):
        cut = max(text.rfind("\n", 0, m.start()), text.rfind(". ", 0, m.start()) + 1)
        if cut > 0:
            text = text[:cut]

    return re.sub(r"[ \t]{2,}", " ", text).strip()


# --- Step 2+3: span detection and replacement -----------------------------

def _extra_spans(text: str) -> list[tuple[int, int, str, float, str]]:
    """Name positions NER handles badly: sign-offs, dashes, greetings,
    possessives. Tagged "rule" so the vocabulary can refuse to learn from them —
    these fire on position, not on evidence that the word IS a name."""
    out = []
    for m in P_SIGNOFF.finditer(text):
        line = m.group(1).strip()
        if line and P_NAMEY.match(line):
            out.append((m.start(1), m.start(1) + len(line), "PERSON", 0.6, "rule"))
    for m in P_DASH_NAME.finditer(text):
        out.append((m.start(1), m.end(1), "PERSON", 0.6, "rule"))
    for m in P_GREETING.finditer(text):
        target = m.group(1)
        if target.split()[0].lower() not in NAME_STOPLIST:
            out.append((m.start(1), m.end(1), "PERSON", 0.65, "rule"))
    for m in P_POSSESSIVE.finditer(text):
        if m.group(1).lower() not in NAME_STOPLIST:
            out.append((m.start(1), m.end(1), "PERSON", 0.55, "rule"))
    return out


def _merge(spans: list[tuple[int, int, str, float, str]]) -> list[tuple[int, int, str, str]]:
    """Overlapping detections become one span labelled by the highest-scoring
    member. Presidio happily returns EMAIL_ADDRESS and URL over the same
    characters; replacing them independently would corrupt the offsets."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: (s[0], -s[1]))
    merged: list[list] = []
    for start, end, kind, score, source in spans:
        if merged and start < merged[-1][1]:
            top = merged[-1]
            top[1] = max(top[1], end)
            if score > top[3]:
                top[2], top[3], top[4] = kind, score, source
        else:
            merged.append([start, end, kind, score, source])
    return [(s, e, k, src) for s, e, k, _, src in merged]


def detect(text: str, entities: tuple[str, ...] = BASE_ENTITIES) -> list[tuple[int, int, str]]:
    """Merged identifier spans. Separate from replacement so a pool can be
    analysed once and rewritten twice — see scrub_pool."""
    if not text or not text.strip():
        return []
    results = _analyzer().analyze(text=text, language="en", entities=list(entities),
                                  score_threshold=SCORE_THRESHOLD)
    # "ner" only where the model itself decided; the custom PatternRecognizers
    # are positional like the rules, so they do not teach the vocabulary either
    spans = [(r.start, r.end, r.entity_type, r.score,
              "ner" if "spacy" in str((r.recognition_metadata or {})
                                      .get("recognizer_name", "")).lower() else "rule")
             for r in results]
    spans += _extra_spans(text)
    return _merge(spans)


def apply_spans(text: str, spans: list[tuple[int, int, str]],
                vocab: re.Pattern | None = None) -> str:
    """Replace detected spans with placeholders, then sweep the pool-wide name
    vocabulary over whatever is left."""
    out = text
    for start, end, kind, _source in reversed(spans):
        out = out[:start] + PLACEHOLDER.get(kind, f"<{kind}>") + out[end:]
    if vocab is not None:
        out = vocab.sub("<PERSON>", out)
    # "Greg/Phillip" -> "<PERSON>/<PERSON>" -> one placeholder; repeated tags
    # add nothing and read as noise
    out = re.sub(r"(<(?:PERSON|EMAIL|PHONE|LOCATION|DATE|ID_NUMBER|ADDRESS|URL)>)"
                 r"(?:[\s/,&]+\1)+", r"\1", out)
    # "<addr@host>" scrubs to "<<EMAIL>>", which then reads as an attachment
    # stub to the audit; the angle brackets were the mail syntax, not content
    out = re.sub(r"<(<[A-Z_]+>)>", r"\1", out)
    return out


def scrub(text: str, entities: tuple[str, ...] = BASE_ENTITIES,
          vocab: re.Pattern | None = None) -> str:
    """Replace every detected identifier with its typed placeholder."""
    if not text or not text.strip():
        return text
    return apply_spans(text, detect(text, entities), vocab)


def deidentify_email(body: str, subject: str,
                     entities: tuple[str, ...] = BASE_ENTITIES,
                     vocab: re.Pattern | None = None) -> tuple[str, str]:
    """One email, cleaned then scrubbed. The subject goes through the same pass:
    it becomes the user_intent column, and subjects name people too."""
    return (scrub(clean_residue(body), entities, vocab),
            scrub(subject, entities, vocab))


# --- Pool-wide name vocabulary -------------------------------------------
# A name NER misses in one email it usually catches in another: "Sophie" is
# invisible in "Sophie's admission" but plain in "Sophie asked me to". So the
# pool is analysed once, every PERSON span contributes its tokens to a
# vocabulary, and the rewrite sweeps that vocabulary over every email. Misses
# are repaired using the pool's own evidence rather than an external name list.

P_CAPITALISED = re.compile(r"\b[A-Z][\w'-]{2,}\b")


def person_vocabulary(samples: list[tuple[str, list[tuple[int, int, str, str]]]],
                      min_count: int = 3, min_ratio: float = 0.5) -> re.Pattern | None:
    """Tokens the MODEL repeatedly called a person, and rarely anything else.

    Two guards, both learned the hard way — a first version swept "Let", "This"
    and "December" out of every email:

      min_count   a token must be tagged PERSON several times, so one bad tag
                  cannot redact a word pool-wide.
      min_ratio   of every capitalised appearance of that token, at least half
                  must be inside a PERSON span. "The" is capitalised thousands
                  of times and tagged almost never, so it cannot qualify however
                  often it is mis-tagged; a real name is tagged nearly every time.

    Only spans NER produced are learned from — the positional rules (sign-off,
    greeting, possessive) fire on where a word sits, not on what it is, so
    letting them teach the vocabulary is how "Let me know" became a name.
    """
    from collections import Counter

    tagged: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    for text, spans in samples:
        for token in P_CAPITALISED.findall(text):
            seen[token] += 1
        for start, end, kind, source in spans:
            if kind != "PERSON" or source != "ner":
                continue
            for token in re.findall(r"[A-Z][\w'-]+", text[start:end]):
                if len(token) >= 3 and token.lower() not in NAME_STOPLIST:
                    tagged[token] += 1
    names = sorted({n for n, c in tagged.items()
                    if c >= min_count and c / max(seen[n], c) >= min_ratio},
                   key=len, reverse=True)
    if not names:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")(?:['’]s)?\b")


def scrub_pool(pairs: list[tuple[str, str]], entities: tuple[str, ...] = BASE_ENTITIES,
               min_count: int = 3, use_vocab: bool = True,
               progress: bool = True) -> tuple[list[tuple[str, str]], int]:
    """De-identify a whole pool. Returns (scrubbed pairs, vocabulary size).

    The analyzer runs once per email; the vocabulary pass is regex-cheap.
    """
    from tqdm import tqdm

    cleaned = [(clean_residue(b), s) for b, s in pairs]
    it = tqdm(cleaned, desc="Presidio") if progress else cleaned
    detected = [((b, detect(b, entities)), (s, detect(s, entities))) for b, s in it]

    vocab = None
    if use_vocab:
        flat = [pair for row in detected for pair in row]
        vocab = person_vocabulary(flat, min_count=min_count)

    out = [(apply_spans(b, bs, vocab), apply_spans(s, ss, vocab))
           for (b, bs), (s, ss) in detected]
    size = 0 if vocab is None else vocab.pattern.count("|") + 1
    return out, size


# --- Audit ----------------------------------------------------------------

def audit_counts(texts: list[str]) -> dict[str, tuple[int, int]]:
    """{pattern: (rows containing it, total occurrences)} — the table the HTML
    audit reports, regenerated from whatever pool is passed in."""
    out = {}
    for label, pattern in AUDIT_PATTERNS.items():
        check = VALIDATORS.get(label)
        rows = occurrences = 0
        for t in texts:
            hits = [m.group(0) for m in pattern.finditer(t)]
            if check:
                hits = [h for h in hits if check(h)]
            if hits:
                rows += 1
                occurrences += len(hits)
        out[label] = (rows, occurrences)
    return out


def placeholder_counts(texts: list[str]) -> dict[str, int]:
    tags = sorted(set(PLACEHOLDER.values()))
    return {tag: sum(t.count(tag) for t in texts) for tag in tags}


def _report(before: list[str], after: list[str], seconds: float) -> int:
    n = len(before)
    b, a = audit_counts(before), audit_counts(after)
    print(f"\n{'artefact':<28}{'kind':<11}{'before':>18}{'after':>18}   status")
    print("-" * 93)
    leaks = residue = 0
    for label in AUDIT_PATTERNS:
        (rb, ob), (ra, oa) = b[label], a[label]
        status = "clear" if ra == 0 else ("unchanged" if ra >= rb else "reduced")
        kind = "identifier" if label in IDENTIFIER_LABELS else "residue"
        if label in IDENTIFIER_LABELS:
            leaks += ra
        else:
            residue += ra
        print(f"{label:<28}{kind:<11}{f'{rb:,} rows ({ob:,})':>18}"
              f"{f'{ra:,} rows ({oa:,})':>18}   {status}")

    print("\nplaceholders written:")
    for tag, count in placeholder_counts(after).items():
        if count:
            print(f"   {tag:<14}{count:>8,}")

    kept = sum(len(t) for t in after) / max(1, sum(len(t) for t in before))
    print(f"\n{n:,} emails · {seconds:.1f}s ({seconds / max(1, n) * 1000:.0f} ms each) "
          f"· {kept:.1%} of the original characters kept")
    print(f"residual IDENTIFIER rows: {leaks:,}"
          + ("  <- inspect before trusting the pool" if leaks else "  (none)"))
    print(f"residual boilerplate rows: {residue:,}  (not identifying on their own; "
          f"any names/addresses inside are already placeholders)")
    print("\nThis table cannot see a missed NAME — no regex can. Read the written "
          "samples by hand before signing the pool off.")
    return leaks


def cmd_audit(args: argparse.Namespace) -> int:
    from organic_dataset import pull_aeslc  # heavy (datasets); only needed here

    pool = pull_aeslc(args.limit, args.seed, deidentify=False)
    bodies = [r["context"] for r in pool]
    subjects = [r["user_intent"] for r in pool]

    print(f"\nde-identifying {len(pool):,} emails "
          f"(entities: {', '.join(_entities(args))}) ...")
    t0 = time.perf_counter()
    entities = _entities(args)
    done, vocab_size = scrub_pool(list(zip(bodies, subjects)), entities,
                                  min_count=args.min_count, use_vocab=not args.no_vocab)
    seconds = time.perf_counter() - t0
    if not args.no_vocab:
        print(f"   name vocabulary: {vocab_size:,} names seen as PERSON "
              f"{args.min_count}+ times, swept over every email")

    scrubbed_bodies = [b for b, _ in done]
    scrubbed_subjects = [s for _, s in done]
    leaks = _report(bodies + subjects, scrubbed_bodies + scrubbed_subjects, seconds)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("Scrubbed AESLC samples for manual review.\n"
                    "Read for: names, initials, nicknames, or details that identify "
                    "someone without naming them.\n" + "=" * 70 + "\n")
            for i in range(min(args.samples, len(scrubbed_bodies))):
                f.write(f"\n--- sample {i + 1} ---\nSUBJECT: {scrubbed_subjects[i]}\n"
                        f"{scrubbed_bodies[i]}\n")
        print(f"\n{min(args.samples, len(scrubbed_bodies))} scrubbed samples -> {out}")
        print("   Review these by hand; the regex table above cannot see a missed name.")
    return 1 if leaks else 0


def _entities(args: argparse.Namespace) -> tuple[str, ...]:
    entities = list(BASE_ENTITIES)
    if args.locations:
        entities.append("LOCATION")
    if args.dates:
        entities.append("DATE_TIME")
    return tuple(entities)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="Scrub the AESLC pool and report what is left.")
    a.add_argument("--limit", type=int, default=7000, help="Emails to pull and scrub.")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--samples", type=int, default=25, help="Scrubbed samples to write.")
    a.add_argument("--out", type=Path, default=None, help="Where to write those samples.")
    a.add_argument("--locations", action="store_true", help="Also scrub LOCATION.")
    a.add_argument("--dates", action="store_true", help="Also scrub DATE_TIME.")
    a.add_argument("--no-vocab", action="store_true",
                   help="Skip the pool-wide name vocabulary pass.")
    a.add_argument("--min-count", type=int, default=3,
                   help="Times a token must be seen as PERSON to enter the vocabulary.")
    a.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
