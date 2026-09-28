"""
Gazetteer-based post-processing for fine-grained entity typing AND
span correction.

Two distinct jobs, both driven by the same underlying term lists:

1. classify(entity_text) -- exact-match lookup on an already-extracted
   entity string. This was the original (v1) behavior: takes what the
   model already found and labels it more specifically.

2. find_spans_in_text(text) -- scans the ORIGINAL, FULL input text for
   known gazetteer terms directly, independent of what the model
   extracted. This is what fixes two specific, evidence-backed failure
   modes seen in testing:
     - Span-boundary truncation: model extracts "Finance" instead of
       "Federal Ministry of Finance", or "gote Group" instead of
       "Dangote Group".
     - Subword fragmentation: model splits "Burna Boy" into two separate
       entities ("Bur" + "na Boy") because the name wasn't in training data.
   Since the gazetteer already lists correct full-length authoritative
   strings, scanning the raw text finds them intact regardless of how
   the model's subword tokenizer mangled them -- then api.py reconciles
   these matches against (and can outright replace/merge/inject) the
   model's own detections.

Usage:
    from gazetteer_lookup import Gazetteer

    gaz = Gazetteer("./gazetteers")
    subcategory = gaz.classify("Lagos")             # -> "Nigerian City"
    spans = gaz.find_spans_in_text("... Dangote Group ...")
    # -> [{"start": ..., "end": ..., "text": "Dangote Group",
    #      "category": "Business/Brand", "entity_group": "ORG"}]
"""

import json
import os
import re


# Maps each fine-grained gazetteer category to the coarse entity type your
# base model uses (PER/LOC/ORG/DATE), so text-scanned matches -- which have
# no model-predicted tag of their own -- get a sensible one assigned.
CATEGORY_TO_ENTITY_GROUP = {
    "Nigerian State": "LOC",
    "Nigerian City": "LOC",
    "Political Party": "ORG",
    "Government Agency": "ORG",
    "University/Institution": "ORG",
    "Business/Brand": "ORG",
    "Sports Team": "ORG",
    "Social Movement": "ORG",
    "Entertainment Personality": "PER",
    "Politician/Public Official": "PER",
    "Landmark/Ministry": "ORG",
}


class Gazetteer:
    def __init__(self, gazetteer_dir="./gazetteers"):
        self.lookup = {}       # normalized_text -> category label (for classify())
        self.terms = []        # list of (original_text, category) (for find_spans_in_text())

        self._load(os.path.join(gazetteer_dir, "states_and_cities.json"), {
            "states": "Nigerian State",
            "major_cities": "Nigerian City",
        })
        self._load(os.path.join(gazetteer_dir, "political_parties.json"), {
            "parties": "Political Party",
        })
        self._load(os.path.join(gazetteer_dir, "government_agencies.json"), {
            "agencies": "Government Agency",
        })
        self._load(os.path.join(gazetteer_dir, "universities.json"), {
            "universities": "University/Institution",
        })
        self._load(os.path.join(gazetteer_dir, "businesses_and_brands.json"), {
            "businesses": "Business/Brand",
        })
        self._load(os.path.join(gazetteer_dir, "sports.json"), {
            "national_teams": "Sports Team",
            "domestic_clubs": "Sports Team",
        })
        self._load(os.path.join(gazetteer_dir, "social_movements.json"), {
            "movements": "Social Movement",
        })
        self._load(os.path.join(gazetteer_dir, "entertainment_personalities.json"), {
            "personalities": "Entertainment Personality",
        })
        self._load(os.path.join(gazetteer_dir, "politicians_and_officials.json"), {
            "current_president": "Politician/Public Official",
            "current_vice_president": "Politician/Public Official",
            "current_state_governors": "Politician/Public Official",
            "current_senate_president": "Politician/Public Official",
            "current_speaker_house_of_reps": "Politician/Public Official",
            "notable_past_officeholders_for_historical_mentions": "Politician/Public Official",
        })
        self._load(os.path.join(gazetteer_dir, "landmarks.json"), {
            "landmarks_and_ministries": "Landmark/Ministry",
        })

        print(f"Gazetteer loaded: {len(self.lookup)} known entity strings across all categories.")

        # Build one compiled regex covering every known term, longest-first
        # so e.g. "Dangote Group" matches before a shorter overlapping term
        # would. Uses \w (Unicode-aware in Python's re module) rather than
        # [A-Za-z0-9] for boundaries, so diacritic letters used in Yoruba/
        # Igbo/Hausa orthography (ị, ọ, ṣ, etc.) correctly count as "word
        # characters" -- without this, a boundary was incorrectly detected
        # right after a diacritic letter, causing false matches like a
        # single-letter party abbreviation ("A") matching the trailing "a"
        # in an unrelated word like "ịbịa".
        #
        # Terms under MIN_TERM_LENGTH_FOR_SCAN are excluded from full-text
        # scanning entirely -- very short strings (single-letter party
        # abbreviations, 2-letter acronyms) are too ambiguous to safely
        # search for in free text, even with correct boundaries. They
        # remain usable via classify() for already-extracted entities,
        # where the search space is far smaller and less noisy.
        MIN_TERM_LENGTH_FOR_SCAN = 3
        scannable_terms = [(t, c) for t, c in self.terms if len(t) >= MIN_TERM_LENGTH_FOR_SCAN]
        terms_sorted = sorted(scannable_terms, key=lambda t: len(t[0]), reverse=True)
        if terms_sorted:
            alternation = "|".join(re.escape(term) for term, _ in terms_sorted)
            self._pattern = re.compile(
                r"(?<!\w)(" + alternation + r")(?!\w)",
                re.IGNORECASE | re.UNICODE,
            )
        else:
            self._pattern = None
        # Map normalized term -> category, for resolving which category a
        # regex match belongs to (regex only tells us the matched text).
        self._term_to_category = {self._normalize(t): c for t, c in self.terms}

    def _load(self, filepath, key_to_category):
        if not os.path.exists(filepath):
            print(f"  Warning: gazetteer file not found, skipping: {filepath}")
            return
        # utf-8-sig transparently strips a UTF-8 BOM if present (e.g. from
        # PowerShell's `Out-File -Encoding utf8`) and behaves identically to
        # plain utf-8 for files that don't have one -- safe either way.
        with open(filepath, encoding="utf-8-sig") as f:
            data = json.load(f)
        for key, category in key_to_category.items():
            for entry in data.get(key, []):
                self.lookup[self._normalize(entry)] = category
                self.terms.append((entry, category))

    @staticmethod
    def _normalize(text):
        text = text.strip().lower()
        text = text.strip(".,!?;:'\"()[]{}")
        return text

    def classify(self, entity_text):
        """Exact-match lookup on an already-extracted entity string."""
        return self.lookup.get(self._normalize(entity_text))

    def find_spans_in_text(self, text):
        """
        Scans the full original text for known gazetteer terms, returning
        non-overlapping matches (longest-preferred at each position) with
        exact character offsets INTO THE GIVEN TEXT -- so these can be
        directly compared against/merged with model-predicted entity spans.
        """
        if not self._pattern:
            return []
        results = []
        for match in self._pattern.finditer(text):
            matched_text = match.group(1)
            category = self._term_to_category.get(self._normalize(matched_text))
            if category is None:
                continue
            results.append({
                "start": match.start(1),
                "end": match.end(1),
                "text": matched_text,      # exact casing as it appeared in the input
                "category": category,
                "entity_group": CATEGORY_TO_ENTITY_GROUP.get(category, "ORG"),
            })
        return results


if __name__ == "__main__":
    gaz = Gazetteer("./gazetteers")

    print("\nExact-match classify() test:")
    for text in ["Lagos", "APC", "INEC", "UNILAG", "Dangote Group",
                 "Super Eagles", "EndSARS", "Burna Boy", "Random Text Here"]:
        print(f"  {text!r:30s} -> {gaz.classify(text)}")

    print("\nFull-text find_spans_in_text() test (fixes truncation/fragmentation):")
    sample = ("Buhari of APC don comot go Aso Rock, meanwhile Dangote Group and "
              "GTBank dey sponsor EndSARS memorial for Kano wella.")
    for span in gaz.find_spans_in_text(sample):
        print(f"  [{span['start']}:{span['end']}] {span['text']!r} "
              f"-> {span['category']} ({span['entity_group']})")
