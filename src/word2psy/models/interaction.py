"""Social-interaction scalar rates per chunk — pure Python via nltk.

The readability pattern applied to interactional language: closed-list
rates of person deixis (who is being talked to/about), person-referring
nouns, and conversational discourse markers, plus the fraction of
question sentences. Analytic — the word lists are in this module, so a
score is always explainable by pointing at the tokens that produced it.

Rates are per word token (nltk word_tokenize, punctuation-only tokens
dropped, lowercased — the same tokenization as the pipeline's word
tables); ``interaction_question`` is per sentence. A chunk with no word
tokens scores NaN throughout. Short single-word chunks yield degenerate
values (0, or 1 for a pronoun) — like readability, these metrics are
meaningful for sentence-or-longer chunks.
"""

from word2psy.models.base import BaseModel

FIRST_PERSON_SING = frozenset("i me my mine myself".split())
FIRST_PERSON_PLUR = frozenset("we us our ours ourselves".split())
SECOND_PERSON = frozenset("you your yours yourself yourselves".split())
THIRD_PERSON = frozenset(
    # person-referring only: "it"/"its"/"itself" deliberately excluded
    "he him his himself she her hers herself they them their theirs themselves".split()
)
PERSON_NOUNS = frozenset(
    """person people man men woman women guy guys girl girls boy boys kid
    kids child children baby babies friend friends family mother mom mommy
    father dad daddy parent parents brother brothers sister sisters son
    daughter grandma grandpa grandmother grandfather aunt uncle cousin
    husband wife neighbor neighbors teacher doctor stranger lady ladies
    gentleman gentlemen sir madam mister miss folks everyone everybody
    someone somebody anyone anybody nobody""".split()
)
DISCOURSE_MARKERS = frozenset(
    """hello hi hey yeah yes no okay ok oh wow huh hmm please thanks thank
    bye goodbye well right really sorry excuse pardon anyway alright""".split()
)


class InteractionModel(BaseModel):
    """Seven social-interaction rates per chunk (closed word lists)."""

    name = "interaction"
    nulls = {c: {"means": "undefined",
                 "when": "the chunk has no word tokens (a rate over zero words)"
                 + ("; also when it has no sentences" if c == "interaction_question" else "")}
             for c in ("interaction_first_person_sing", "interaction_first_person_plur",
                       "interaction_second_person", "interaction_third_person",
                       "interaction_person_noun", "interaction_discourse_marker",
                       "interaction_question")}
    level = "chunk"
    checkpoint = None  # analytic — closed lists above, no learned weights

    def load(self) -> None:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        self.model = nltk

    def predict(self, text: str) -> dict[str, float]:
        import re

        nltk = self.model
        sentences = nltk.sent_tokenize(text)
        tokens = [
            t.lower()
            for s in sentences
            for t in nltk.word_tokenize(s)
            if not re.fullmatch(r"[^\w]+", t)
        ]
        n = len(tokens)

        def rate(wordlist: frozenset) -> float:
            return sum(t in wordlist for t in tokens) / n if n else float("nan")

        question = (
            sum(s.rstrip().endswith("?") for s in sentences) / len(sentences)
            if sentences
            else float("nan")
        )
        return {
            "interaction_first_person_sing": rate(FIRST_PERSON_SING),
            "interaction_first_person_plur": rate(FIRST_PERSON_PLUR),
            "interaction_second_person": rate(SECOND_PERSON),
            "interaction_third_person": rate(THIRD_PERSON),
            "interaction_person_noun": rate(PERSON_NOUNS),
            "interaction_discourse_marker": rate(DISCOURSE_MARKERS),
            "interaction_question": question if n else float("nan"),
        }
