"""De-identify AESLC emails (Presidio plus custom recognizers); run it directly to audit the result."""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from tqdm import tqdm

# organic_dataset imports this module back for scrub_pool; plain module imports keep that cycle safe
import organic_dataset
from progress import track  # src/progress.py: the dashboard's bar; tqdm stays for the terminal

# patterns shared by the scrubber and the audit below, so the two cannot drift apart
# internal mail-system addresses ("Jane Doe/HOU/ECT@ECT")
P_INTERNAL_STRICT = re.compile(
    r"\b[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3}\s*/[A-Z]{2,6}/[A-Z]{2,6}(?:@[A-Z]{2,6})?")
P_INTERNAL_LOOSE = re.compile(
    r"\b[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3}\s*/[A-Z]{2,6}@[A-Z]{2,6}")
P_EXTENSION = re.compile(r"\b(?:x|ext\.?)\s?\d{3,6}\b", re.I)
# +44-style numbers with spaces, which Presidio's phone recognizer misses
P_INTL_PHONE = re.compile(r"\+\d{1,3}(?:[\s.\-]?\(?\d{1,4}\)?){2,5}")
# anything phone-shaped, for contact lists where Presidio has no context to go on
P_NANP_PHONE = re.compile(r"\(?\b\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b")
# quoted-printable soft breaks split an address across lines ("user@host.=\ncom")
P_SOFT_BREAK = re.compile(r"=\r?\n")
# AESLC left "= " inside some addresses ("user@corp.example.= com"); rejoined only inside an address
P_QP_SPLIT_EMAIL = re.compile(r"([\w.+-]+@[\w.-]*?)=\s+(?=\w)")
# stricter than Presidio's URL recognizer: a URL needs a scheme or a real TLD
P_URL_SCHEME = re.compile(r"(?:https?://|ftp://|www\.)[\w./?=&%#:+~@-]+", re.I)
P_URL_BARE = re.compile(
    r"\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|edu|gov|mil|int|io|co|uk|de|fr|ca|au"
    r"|biz|info|tv|us)\b(?:/[\w./?=&%#:+~@-]*)?", re.I)
P_HTML_TAG = re.compile(r"<[^<>\n]{1,200}>")
P_HTML_ISH = re.compile(r"(?i)<(?:font|br|html|body|table|div|p|span)\b")
# RFC display name ("Lastname, Firstname" <addr>), where NER misses the given name
P_DISPLAY_NAME = re.compile(r"\"[^\"\n]{1,60}\"(?=\s*<)")
P_ATTACHMENT_STUB = re.compile(r"<<[^<>\n]{1,160}>>")
P_STREET = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][\w.'-]+\s+){0,3}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|"
    r"Circle|Cir|Way|Place|Pl|Parkway|Pkwy|Highway|Hwy|Suite|Ste|Floor)\b\.?",
    re.I)
P_PO_BOX = re.compile(r"\bP\.?\s?O\.?\s*Box\s+\d+\b", re.I)

# a disclaimer is cut only when it opens a line and the text after it confirms it
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
# inline disclaimers: only give-away phrases, since a bare "confidential" is ordinary prose
P_DISCLAIMER_INLINE = re.compile(
    r"(?i)(intended\s+recipient"
    r"|received\s+this\s+(?:e-?mail\s+|message\s+)?in\s+error"
    r"|unauthorized\s+(?:use|review|disclosure|copying|dissemination)"
    r"|notify\s+the\s+sender\s+immediately)")

PLACEHOLDER = {
    "PERSON": "<PERSON>",
    "INTERNAL_ADDRESS": "<PERSON>",
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
# US_DRIVER_LICENSE and US_BANK_NUMBER are left out: they over-match ordinary tokens
BASE_ENTITIES = (
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "URL_STRICT",
    "US_SSN", "CREDIT_CARD", "IBAN_CODE", "US_PASSPORT",
    "INTERNAL_ADDRESS", "PHONE_EXTENSION", "STREET_ADDRESS",
)

SCORE_THRESHOLD = 0.35

# name alone on the line after a sign-off, or behind a leading dash
P_SIGNOFF = re.compile(
    r"(?im)^(?:thanks|thank\s+you|regards|best\s+regards|best|sincerely|cheers|"
    r"yours|take\s+care)[,!.]*[ \t]*\r?\n+([^\n]{1,40})$")
P_DASH_NAME = re.compile(r"(?m)^[ \t]*[-–—]{1,2}[ \t]*([A-Za-z][\w.'-]{1,20})[ \t]*$")
# every token capitalised: "Phillip K Allen" is a sign-off, "Let me know" is not
P_NAMEY = re.compile(r"^[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,2}[.!]?$")
# a first name after a greeting ("Dear Richard"), which NER reads weakly
P_GREETING = re.compile(
    r"(?i)\b(?:dear|hi|hello|hey|good\s+(?:morning|afternoon|evening))[ ,]+"
    r"([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,2})")
# possessive names ("Sophie's"), which lack the sentence cues NER relies on
P_POSSESSIVE = re.compile(r"\b([A-Z][a-z]{2,15})['’]s\b")

# words that follow a greeting or take 's but name nobody, and given names that are ordinary words
NAME_STOPLIST = {
    "all", "friends", "family", "team", "everyone", "everybody", "folks", "sir",
    "madam", "colleagues", "customer", "customers", "guys", "there", "gentlemen",
    "ladies", "group", "sirs", "member", "members",
    "april", "may", "june", "july", "august", "march", "january", "monday",
    "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "bill", "mark", "rich", "grant", "will", "art", "chuck", "frank", "jack",
    "don", "rob", "buck", "sue", "gene", "hope", "faith", "max", "ray", "gay",
    "enron", "houston", "texas",   # AESLC's company and city, not people
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
    """Presidio engine with the custom recognizers, built once on first use."""
    global _ANALYZER
    if _ANALYZER is not None:
        return _ANALYZER
    engine = AnalyzerEngine()
    custom = [
        ("INTERNAL_ADDRESS", "internal_address", P_INTERNAL_STRICT.pattern, 0.75),
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


def clean_residue(text: str) -> str:
    """Drop text the sender didn't write: attachment stubs, HTML tags and trailing disclaimers."""
    # rejoin soft breaks first, or a split address is invisible to every recognizer
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

    # inline disclaimer: cut at its sentence, but only past the first 35% of the email
    m = P_DISCLAIMER_INLINE.search(text)
    if m and m.start() > 0.35 * len(text):
        cut = max(text.rfind("\n", 0, m.start()), text.rfind(". ", 0, m.start()) + 1)
        if cut > 0:
            text = text[:cut]

    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _extra_spans(text: str) -> list[tuple[int, int, str, float, str]]:
    """Positional name rules, tagged "rule" so they never teach the name vocabulary."""
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
    """Merge overlapping detections into one span, labelled by the highest-scoring one."""
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
    """Merged identifier spans for one text."""
    if not text or not text.strip():
        return []
    results = _analyzer().analyze(text=text, language="en", entities=list(entities),
                                  score_threshold=SCORE_THRESHOLD)
    # "ner" only for spaCy's own detections; the custom pattern recognizers count as rules
    spans = [(r.start, r.end, r.entity_type, r.score,
              "ner" if "spacy" in str((r.recognition_metadata or {})
                                      .get("recognizer_name", "")).lower() else "rule")
             for r in results]
    spans += _extra_spans(text)
    return _merge(spans)


def apply_spans(text: str, spans: list[tuple[int, int, str]],
                vocab: re.Pattern | None = None) -> str:
    """Replace spans with placeholders, then sweep the pool-wide name vocabulary."""
    out = text
    for start, end, kind, _source in reversed(spans):
        out = out[:start] + PLACEHOLDER.get(kind, f"<{kind}>") + out[end:]
    if vocab is not None:
        out = vocab.sub("<PERSON>", out)
    # collapse repeated tags ("<PERSON>/<PERSON>" -> "<PERSON>")
    out = re.sub(r"(<(?:PERSON|EMAIL|PHONE|LOCATION|DATE|ID_NUMBER|ADDRESS|URL)>)"
                 r"(?:[\s/,&]+\1)+", r"\1", out)
    # "<addr@host>" scrubs to "<<EMAIL>>"; drop the mail-syntax brackets
    out = re.sub(r"<(<[A-Z_]+>)>", r"\1", out)
    return out


P_CAPITALISED = re.compile(r"\b[A-Z][\w'-]{2,}\b")


def person_vocabulary(samples: list[tuple[str, list[tuple[int, int, str, str]]]],
                      min_count: int = 3, min_ratio: float = 0.5) -> re.Pattern | None:
    """Names NER tagged PERSON at least min_count times and in at least min_ratio of their uses."""
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
    # longest first so "Johnson" wins over "John"; ties alphabetical, so the pattern is the same every run
    names = sorted({n for n, c in tagged.items()
                    if c >= min_count and c / max(seen[n], c) >= min_ratio},
                   key=lambda n: (-len(n), n))
    if not names:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")(?:['’]s)?\b")


def scrub_pool(pairs: list[tuple[str, str]], entities: tuple[str, ...] = BASE_ENTITIES,
               min_count: int = 3, use_vocab: bool = True,
               progress: bool = True) -> tuple[list[tuple[str, str]], int]:
    """De-identify a whole pool; a name NER catches in one email is swept from all of them."""
    cleaned = [(clean_residue(b), s) for b, s in pairs]
    it = cleaned
    if progress:
        it = track(tqdm(cleaned, desc="Presidio"), "anonymize", "emails", total=len(cleaned))
    detected = [((b, detect(b, entities)), (s, detect(s, entities))) for b, s in it]

    vocab = None
    if use_vocab:
        flat = [pair for row in detected for pair in row]
        vocab = person_vocabulary(flat, min_count=min_count)

    out = [(apply_spans(b, bs, vocab), apply_spans(s, ss, vocab))
           for (b, bs), (s, ss) in detected]
    size = 0 if vocab is None else vocab.pattern.count("|") + 1
    return out, size


# ---------------- audit: python src/deidentify.py --limit 7000 --out review.txt ----------------

# audit rows: label -> pattern, reported as rows containing it and total occurrences
AUDIT_PATTERNS: dict[str, re.Pattern] = {
    "email address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "company-domain address": re.compile(r"[\w.+-]+@enron\.com", re.I),   # AESLC's own mail domain
    "phone number": re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "URL": re.compile(r"(?:https?://|www\.)[\w./?=&%#-]+", re.I),
    "internal address (strict)": P_INTERNAL_STRICT,
    "internal address (loose)": P_INTERNAL_LOOSE,
    "phone extension": P_EXTENSION,
    "attachment stub": P_ATTACHMENT_STUB,
    "confidentiality wording": re.compile(r"(?i)confidential"),
    "'intended recipient'": re.compile(r"(?i)intended\s+recipient"),
    "'received in error'": re.compile(r"(?i)received\s+this\s+in\s+error"),
    # two or more header lines together, not a lone "Subject:" line
    "quoted header block": re.compile(
        r"(?im)^\s*(?:From|Sent|To|Cc|Bcc|Subject)\s*:.*\r?\n\s*(?:From|Sent|To|Cc|Bcc|Subject)\s*:"),
    "'Original Message' bar": re.compile(r"(?im)^[\s>]*-{2,}\s*original\s+message\s*-{2,}"),
    "'Forwarded by'": re.compile(r"(?i)forwarded\s+by"),
    "street address": P_STREET,
    "PO box": P_PO_BOX,
    "SSN-shaped": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card-shaped (Luhn)": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
}


def _luhn(value: str) -> bool:
    """Card-number checksum, so deal numbers and aligned tables don't count as cards."""
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

# identifiers left in the text are privacy leaks; the other audit rows are only boilerplate
IDENTIFIER_LABELS = {
    "email address", "company-domain address", "phone number", "URL",
    "internal address (strict)", "internal address (loose)", "phone extension",
    "street address", "PO box", "SSN-shaped", "card-shaped (Luhn)",
}


def audit_counts(texts: list[str]) -> dict[str, tuple[int, int]]:
    """{pattern: (rows containing it, total occurrences)}."""
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


AUDIT_SEED = 42       # same pool draw as the dataset build
AUDIT_SAMPLES = 25    # scrubbed emails written out for reading by hand


def audit(limit: int, out: Path | None) -> int:
    """Scrub `limit` AESLC emails exactly as the dataset build does, report what is left, write samples."""
    pool = organic_dataset.pull_aeslc(limit, AUDIT_SEED, scrub=False)
    bodies = [r["context"] for r in pool]
    subjects = [r["user_intent"] for r in pool]

    print(f"\nde-identifying {len(pool):,} emails (entities: {', '.join(BASE_ENTITIES)}) ...")
    t0 = time.perf_counter()
    done, vocab_size = scrub_pool(list(zip(bodies, subjects)))
    seconds = time.perf_counter() - t0
    print(f"   name vocabulary: {vocab_size:,} names seen as PERSON 3+ times, swept over every email")

    scrubbed_bodies = [b for b, _ in done]
    scrubbed_subjects = [s for _, s in done]
    leaks = _report(bodies + subjects, scrubbed_bodies + scrubbed_subjects, seconds)

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("Scrubbed AESLC samples for manual review.\n"
                    "Read for: names, initials, nicknames, or details that identify "
                    "someone without naming them.\n" + "=" * 70 + "\n")
            for i in range(min(AUDIT_SAMPLES, len(scrubbed_bodies))):
                f.write(f"\n--- sample {i + 1} ---\nSUBJECT: {scrubbed_subjects[i]}\n"
                        f"{scrubbed_bodies[i]}\n")
        print(f"\n{min(AUDIT_SAMPLES, len(scrubbed_bodies))} scrubbed samples -> {out}")
        print("   Review these by hand; the regex table above cannot see a missed name.")
    return 1 if leaks else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=7000, help="Emails to pull and scrub.")
    parser.add_argument("--out", type=Path, default=None, help="Where to write scrubbed samples to read by hand.")
    args = parser.parse_args()
    return audit(args.limit, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
