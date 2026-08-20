"""
Rule-Based NLP Chatbot Engine.

A fully self-contained conversational intake engine. No external API calls.
Uses regex patterns and keyword rules to:
  1. Detect patient intent (greeting, farewell, confusion, data-giving, off-topic)
  2. Extract clinical fields from free-text messages
  3. Generate warm, contextual, varied follow-up questions
  4. Acknowledge what the patient said before asking the next question

This module implements the same interface as gemini_service.extract_information
so main.py can use it as a drop-in replacement.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

def _search(pattern: str, text: str, flags: int = re.IGNORECASE) -> Optional[re.Match]:
    return re.search(pattern, text, flags)


def _num(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "."))
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

_GREETING_PATTERNS = [
    r"\b(hi|hello|hey|howdy|greetings|good\s*(morning|afternoon|evening|day))\b",
    r"^(hi|hello|hey)\s*[!.]*$",
]

_FAREWELL_PATTERNS = [
    r"\b(bye|goodbye|see\s+you|thank\s+you|thanks|that'?s?\s+all|done|finished)\b",
]

_CONFUSION_PATTERNS = [
    r"\b(what\s+do\s+you\s+mean|i\s+don'?t\s+understand|confused|clarify|can\s+you\s+repeat|pardon|sorry\s*\?|huh|not\s+sure\s+what\s+you|what\s+is\s+that)\b",
    r"\?{2,}",
]

_AFFIRMATION_PATTERNS = [
    r"^(yes|yeah|yep|yup|correct|right|sure|absolutely|of\s+course|confirmed|ok|okay|that'?s?\s+right)\s*[.!]*$",
]

_NEGATION_PATTERNS = [
    r"^(no|nope|nah|not\s+really|i\s+don'?t\s+think\s+so|incorrect|wrong)\s*[.!]*$",
]

_HOWRU_PATTERNS = [
    r"how\s+are\s+you",
    r"how\s+do\s+you\s+do",
    r"what'?s?\s+up",
]

_OFFTRACK_PATTERNS = [
    r"\b(weather|sport|football|cricket|movie|film|news|politics|joke|recipe|food|travel)\b",
]


def _detect_intent(text: str) -> str:
    """
    Return one of:
      'greeting', 'farewell', 'confusion', 'howru',
      'affirmation', 'negation', 'off_topic', 'data'
    """
    t = text.strip().lower()

    for p in _GREETING_PATTERNS:
        if _search(p, t):
            return "greeting"

    for p in _HOWRU_PATTERNS:
        if _search(p, t):
            return "howru"

    for p in _FAREWELL_PATTERNS:
        if _search(p, t):
            return "farewell"

    for p in _CONFUSION_PATTERNS:
        if _search(p, t):
            return "confusion"

    for p in _AFFIRMATION_PATTERNS:
        if re.fullmatch(p, t, re.IGNORECASE):
            return "affirmation"

    for p in _NEGATION_PATTERNS:
        if re.fullmatch(p, t, re.IGNORECASE):
            return "negation"

    for p in _OFFTRACK_PATTERNS:
        if _search(p, t):
            return "off_topic"

    return "data"


# ---------------------------------------------------------------------------
# Clinical field extractors
# ---------------------------------------------------------------------------

def _extract_age(text: str) -> Optional[int]:
    """Extract age in years from free text."""
    patterns = [
        r"i(?:'?m| am)\s+(\d{1,3})\s*(?:years?\s*old|yrs?\.?\s*old|y/?o)?",
        r"(?:my\s+)?age\s*(?:is|:|=)\s*(\d{1,3})",
        r"(\d{1,3})\s*years?\s*old",
        r"aged?\s+(\d{1,3})",
        r"\b(\d{1,3})\s*(?:yrs?|y\.?o\.?)\b",
        r"^(\d{1,3})\s*$",
    ]
    for p in patterns:
        m = _search(p, text)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 120:
                return val

    # Written-out numbers (common in voice-to-text)
    word_map = {
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    }
    ones = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19,
    }
    t = text.lower()
    for tens_word, tens_val in word_map.items():
        for ones_word, ones_val in ones.items():
            phrase = f"{tens_word}[-\\s]+{ones_word}"
            if re.search(phrase, t):
                age = tens_val + ones_val
                if 1 <= age <= 120:
                    return age
        if re.search(rf"\b{tens_word}\b", t):
            if 1 <= tens_val <= 120:
                return tens_val
    for ones_word, ones_val in ones.items():
        if re.search(rf"\b{ones_word}\b", t):
            if 1 <= ones_val <= 120:
                return ones_val
    return None


def _extract_diabetes_duration(text: str) -> Optional[float]:
    """Extract diabetes duration in years from free text."""
    # "X and a half years"
    m = _search(r"(\d+)\s+and\s+a\s+half\s+years?", text)
    if m:
        return float(m.group(1)) + 0.5

    # "half a year"
    if _search(r"\bhalf\s+a\s+year\b", text):
        return 0.5

    # "just diagnosed / recently diagnosed / newly diagnosed"
    if _search(r"\b(just|recently|newly)\s+diagnosed\b", text):
        return 0.0

    # Long patterns with context words
    m = _search(
        r"(?:had\s+diabetes|been\s+diabetic|diagnosed|living\s+with\s+diabetes|"
        r"diabetes\s+(?:for|since)|diabetic\s+for|have\s+(?:had\s+)?diabetes)"
        r"\s*(?:for\s+)?(\d+(?:\.\d+)?)\s*years?",
        text,
    )
    if m:
        return float(m.group(1))

    m = _search(
        r"(?:had\s+diabetes|been\s+diabetic|diagnosed|diabetic\s+for|diabetes\s+for)"
        r"\s*(?:for\s+)?(\d+)\s*months?",
        text,
    )
    if m:
        return round(int(m.group(1)) / 12, 2)

    # Fallback: if "diabet" appears in the sentence, grab any year/month number
    if re.search(r"diabet", text, re.I):
        m = _search(r"(\d+(?:\.\d+)?)\s*years?", text)
        if m:
            val = float(m.group(1))
            if 0 < val < 80:
                return val
        m = _search(r"(\d+)\s*months?", text)
        if m:
            return round(int(m.group(1)) / 12, 2)

    # Standalone matches
    m = _search(r"^(\d+(?:\.\d+)?)\s*years?$", text)
    if m:
        val = float(m.group(1))
        if 0 < val < 80:
            return val
    m = _search(r"^(\d+)\s*months?$", text)
    if m:
        return round(int(m.group(1)) / 12, 2)

    return None


def _extract_hba1c(text: str) -> Optional[float]:
    """Extract HbA1c percentage."""
    patterns = [
        r"(?:hba1c|hb\s*a1c|a1c|glycated\s+h[ae]moglobin|glycoh[ae]moglobin)"
        r"\s*(?:is|of|:|was|=|level\s+is|level\s*:)?\s*(\d+(?:\.\d+)?)\s*%?",
        r"(\d+(?:\.\d+)?)\s*%?\s*(?:hba1c|a1c)",
        r"a1c\s+(?:of\s+)?(\d+(?:\.\d+)?)",
    ]
    for p in patterns:
        m = _search(p, text)
        if m:
            val = _num(m.group(1))
            if val is not None and 3.0 <= val <= 20.0:
                return val
    return None


def _extract_blood_pressure(text: str) -> Optional[str]:
    """Extract blood pressure as 'systolic/diastolic'."""
    # Standard "120/80" or "120 over 80"
    m = _search(r"(\d{2,3})\s*(?:/|over|-)\s*(\d{2,3})", text)
    if m:
        sys_, dia = int(m.group(1)), int(m.group(2))
        if 60 <= sys_ <= 250 and 40 <= dia <= 150:
            return f"{sys_}/{dia}"
    # "systolic 120 diastolic 80"
    m = _search(r"systolic\s+(\d{2,3})\s+diastolic\s+(\d{2,3})", text)
    if m:
        sys_, dia = int(m.group(1)), int(m.group(2))
        if 60 <= sys_ <= 250 and 40 <= dia <= 150:
            return f"{sys_}/{dia}"
    # "bp is normal / high / low" → don't extract a number
    return None


def _extract_ldl(text: str) -> Optional[float]:
    m = _search(r"ldl\s*(?:cholesterol\s*)?(?:is|of|:|=|level\s*is)?\s*(\d+(?:\.\d+)?)", text)
    if m:
        val = _num(m.group(1))
        if val and 10 <= val <= 500:
            return val
    return None


def _extract_hdl(text: str) -> Optional[float]:
    m = _search(r"hdl\s*(?:cholesterol\s*)?(?:is|of|:|=|level\s*is)?\s*(\d+(?:\.\d+)?)", text)
    if m:
        val = _num(m.group(1))
        if val and 10 <= val <= 200:
            return val
    return None


def _extract_triglycerides(text: str) -> Optional[float]:
    m = _search(r"triglycerides?\s*(?:is|are|of|:|=|level\s*is)?\s*(\d+(?:\.\d+)?)", text)
    if m:
        val = _num(m.group(1))
        if val and 10 <= val <= 2000:
            return val
    return None


def _extract_hemoglobin(text: str) -> Optional[float]:
    m = _search(r"\bhemoglobin\b\s*(?:is|of|:|=)?\s*(\d+(?:\.\d+)?)", text)
    if m:
        val = _num(m.group(1))
        if val and 4.0 <= val <= 25.0:
            return val
    return None


def _neg_prefix(word: str, text: str) -> bool:
    """Return True if `word` appears with a negation prefix before it."""
    pattern = (
        r"(?:no|not|never|don'?t\s+have|without|denies?|absence\s+of|haven'?t\s+(?:had\s+)?)"
        r"\s+(?:\w+\s+){0,4}" + word
    )
    return bool(_search(pattern, text))


def _extract_smoking(text: str) -> Optional[bool]:
    neg = [
        r"don'?t\s+smok", r"do\s+not\s+smok", r"never\s+smok",
        r"non[- ]?smok", r"not\s+a\s+smok", r"quit\s+smok",
        r"gave?\s+up\s+smok", r"ex[- ]smok", r"stopped?\s+smok",
        r"no\s+smok", r"never\s+smoke", r"i\s+quit", r"i\s+stopped\s+smok",
    ]
    for p in neg:
        if _search(p, text):
            return False

    pos = [
        r"i\s+smok", r"am\s+a\s+smok", r"current\w*\s+smok",
        r"\bsmok(?:e|er|ing)\b", r"cigarette", r"tobacco", r"\bvape\b", r"\bvaping\b",
    ]
    for p in pos:
        if _search(p, text):
            return True
    return None


def _bool_symptom(text: str, pos_kws: List[str], neg_kws: List[str]) -> Optional[bool]:
    """Generic boolean symptom extractor with negation awareness."""
    neg_prefix_pat = (
        r"(?:no|not|don'?t\s+have|without|denies?|absence\s+of|haven'?t|"
        r"never|doesn'?t|cannot|can'?t)\s+(?:\w+\s+){0,3}"
    )
    for kw in pos_kws:
        if _search(neg_prefix_pat + kw, text):
            return False
        if _search(kw, text):
            return True
    for kw in neg_kws:
        if _search(kw, text):
            return False
    return None


def _extract_blurred_vision(text: str) -> Optional[bool]:
    return _bool_symptom(
        text,
        [r"blur(?:red|ry)?\s+vision", r"vision\s+(?:is\s+)?blur", r"blurry\s+(?:eyes?|sight)", r"hazy\s+vision", r"dim\s+vision"],
        [r"vision\s+(?:is\s+)?(?:clear|fine|normal|good|perfect)", r"see\s+clearly", r"no\s+blurring"],
    )


def _extract_floaters(text: str) -> Optional[bool]:
    return _bool_symptom(
        text,
        [r"floaters?", r"see(?:ing)?\s+spots?", r"black\s+spots?", r"floating\s+spots?", r"dark\s+spots?"],
        [r"no\s+floaters?", r"no\s+spots?"],
    )


def _extract_vision_loss(text: str) -> Optional[bool]:
    return _bool_symptom(
        text,
        [r"vision\s+loss", r"lost\s+(?:my\s+)?vision", r"losing\s+vision", r"partial(?:ly)?\s+blind", r"can'?t\s+see\s+well", r"poor\s+vision"],
        [r"no\s+vision\s+loss", r"vision\s+is\s+(?:fine|ok|normal|good)"],
    )


def _extract_eye_pain(text: str) -> Optional[bool]:
    return _bool_symptom(
        text,
        [r"eye\s+pain", r"eyes?\s+hurt", r"eyes?\s+(?:are\s+)?sore", r"pain\s+in\s+(?:my\s+)?eyes?", r"eye\s+discomfort", r"burning\s+eyes?"],
        [r"no\s+(?:eye\s+)?pain", r"eyes?\s+(?:are\s+)?(?:fine|ok|normal|good|healthy)"],
    )


def _extract_name(text: str) -> Optional[str]:
    patterns = [
        r"(?:my\s+name\s+is|i'?m\s+called|call\s+me|name\s*[=:]\s*)([A-Za-z][A-Za-z\s]{1,30})",
        r"(?:i'?m|i\s+am)\s+([A-Za-z][a-z]{1,20})\b",
        r"^([A-Za-z][a-z]{2,20})\s*$",
    ]
    skip = {
        "diabetic", "not", "sorry", "yes", "no", "old", "fine", "okay", "good",
        "here", "ready", "well", "just", "still", "also", "done", "sure", "thanks",
        "taking", "on", "going", "having", "doing", "feeling",
        "male", "female", "man", "woman", "boy", "girl"
    }
    for p in patterns:
        # Use case-insensitive search so "my name is sarah" also works
        m = _search(p, text)
        if m:
            name = m.group(1).strip().split("\n")[0].split()[0]  # take first word only
            if name.lower() not in skip and 2 <= len(name) <= 40:
                return name.title()
    return None


def _extract_gender(text: str) -> Optional[str]:
    t = text.lower()
    if re.search(r"\b(?:i'?m\s+a\s+)?male\b|^male$|\bman\b|\bgentleman\b|\bhe/him\b", t):
        return "male"
    if re.search(r"\b(?:i'?m\s+a\s+)?female\b|^female$|\bwoman\b|\blady\b|\bshe/her\b", t):
        return "female"
    return None


def _extract_dynamic(text: str) -> Dict[str, Any]:
    """Extract free-form dynamic clinical information."""
    dynamic: Dict[str, Any] = {}
    t = text.lower()

    # Medications
    m = _search(
        r"(?:taking|on|prescribed?|using|started?)\s+([\w\s,/+]+?)\s*"
        r"(?:for\s+(?:my\s+)?(?:diabet|blood|pressure|sugar|cholesterol)|$|\.|,|\band\b)",
        t,
    )
    if m:
        meds = m.group(1).strip()
        if len(meds) > 2 and meds not in ("it", "them", "this", "that"):
            dynamic["medications"] = meds.title()

    # Family history
    if _search(
        r"family\s+history|(?:mother|father|parent|sibling|brother|sister|grandparent)"
        r"\s+(?:has|had|with|also|too)",
        t,
    ):
        dynamic["family_history"] = True

    # Occupation
    m = _search(
        r"(?:i\s+(?:am|work)\s+(?:a|as\s+an?|an?)|i\s+am\s+a)\s+([\w\s]+?)"
        r"(?:\s+by|\s+and|\.|,|$)",
        t,
    )
    if m:
        occ = m.group(1).strip()
        if 2 < len(occ) < 30:
            dynamic["occupation"] = occ.title()

    # Glasses / contacts
    if _search(r"\b(?:glasses|spectacles|contacts?|contact\s+lenses?|bifocals?)\b", t):
        dynamic["vision_correction"] = True

    # Previous eye surgeries
    if _search(r"\b(?:laser\s+eye|cataract|glaucoma|retinal|vitrectomy|lasik|eye\s+surgery)\b", t):
        dynamic["eye_surgery_history"] = True

    # Physical activity
    if _search(r"\b(?:exercise|gym|running|walking|yoga|swimming|active)\b", t):
        dynamic["physically_active"] = True

    return dynamic


# ---------------------------------------------------------------------------
# Intent → scripted reply helpers
# ---------------------------------------------------------------------------

def _greeting_reply(name: Optional[str]) -> str:
    greet_name = f", {name}" if name else ""
    options = [
        f"Hello{greet_name}! 👋 I'm Dr. Rita, your virtual screening assistant. "
        f"I'll ask you a few quick questions about your health before your diabetic retinopathy screening. "
        f"Could I start by asking your name?",

        f"Hi there{greet_name}! I'm Dr. Rita, here to help with your pre-screening intake. "
        f"It'll only take a few minutes. First — what's your name?",

        f"Hey{greet_name}! Great to have you here. I'm Dr. Rita and I'll be guiding you through a "
        f"quick health check-in. May I know your name, please?",
    ]
    return random.choice(options)


def _howru_reply() -> str:
    options = [
        "I'm doing great, thank you for asking! 😊 Now, let's get started — could you tell me your name?",
        "Thanks for asking — I'm here and ready to help! Could I get your name to begin?",
        "All good on my end, thanks! Let's make sure you're all set too. What's your name?",
    ]
    return random.choice(options)


def _farewell_reply(completed: bool) -> str:
    if completed:
        return (
            "Thank you so much for completing the intake! 🙏 "
            "Your information has been recorded and the screening team will review it shortly. "
            "Take care and see you soon!"
        )
    return (
        "Thank you for chatting with me! We haven't quite finished the intake yet — "
        "feel free to come back whenever you're ready. Take care! 😊"
    )


def _confusion_reply(missing_fields: List[str], name: Optional[str]) -> str:
    name_part = f", {name}" if name else ""
    if not missing_fields:
        return f"No worries{name_part}! We've already collected everything we need. You're all set! 🎉"

    field = missing_fields[0]
    rephrase = {
        "age": f"No problem{name_part}! I just need to know how old you are — could you share your age?",
        "diabetes_duration": f"Of course{name_part}! I'm simply asking how long you've been living with diabetes. For example, '5 years' or '8 months' would be perfect.",
        "hba1c": f"Sure, let me rephrase{name_part}! HbA1c (also written as A1c) is a blood test result that shows your average blood sugar over the past 3 months. Do you have that number from your last check-up?",
        "blood_pressure": f"My apologies{name_part}! Blood pressure is the measurement from a BP machine, usually written as two numbers like 120/80. Do you happen to know yours?",
    }
    return rephrase.get(field, f"No worries{name_part}! Let me rephrase the question for you.")


def _off_topic_reply(name: Optional[str]) -> str:
    name_part = f", {name}" if name else ""
    options = [
        f"Ha, I wish I could chat about that{name_part}! But I'm a specialist — I only know about diabetes and eye health. Let's get back to your screening, shall we?",
        f"That's a bit outside my area of expertise{name_part}! I'm here specifically for your health intake. Let's continue — I was about to ask you something important.",
        f"Great topic{name_part}, but I'd better stick to what I know! Let me get back to your screening questions.",
    ]
    return random.choice(options)


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

# Template banks — multiple variants per field for conversational variety
_QUESTION_TEMPLATES: Dict[str, List[str]] = {
    "name": [
        "Before we begin, could I get your name?",
        "May I know your name, please?",
        "First things first — what's your name?",
        "Could you tell me your name?",
    ],
    "age": [
        "How old are you?",
        "Could you tell me your age?",
        "What's your age, if you don't mind me asking?",
        "May I ask how old you are?",
    ],
    "gender": [
        "Could you tell me your gender?",
        "How do you identify — male, female, or something else?",
    ],
    "diabetes_duration": [
        "How long have you been living with diabetes?",
        "How many years have you had diabetes?",
        "When were you first diagnosed with diabetes?",
        "Could you tell me how long you've been diabetic?",
    ],
    "hba1c": [
        "Do you know your most recent HbA1c (A1c) reading?",
        "What was your last HbA1c value? It's usually a number like 7.2 or 8.5.",
        "Have you had an HbA1c test recently? What was the result?",
        "Could you share your latest glycated haemoglobin (HbA1c) value?",
    ],
    "blood_pressure": [
        "Do you know your latest blood pressure reading? (e.g. 120/80)",
        "What was your last blood pressure measurement?",
        "Could you share your blood pressure — the reading with two numbers, like 130/85?",
        "Have you had your blood pressure checked recently? What did it show?",
    ],
    "ldl": [
        "Do you know your LDL cholesterol level?",
        "What is your LDL (bad cholesterol) reading, if you have it?",
    ],
    "hdl": [
        "What about your HDL (good cholesterol) level?",
        "Do you know your HDL cholesterol value?",
    ],
    "triglycerides": [
        "Do you have your triglyceride level from a recent blood test?",
        "What are your triglycerides, if you know them?",
    ],
    "hemoglobin": [
        "Do you know your haemoglobin level?",
        "What is your haemoglobin reading?",
    ],
    "smoking": [
        "Do you currently smoke or use tobacco products?",
        "Are you a smoker?",
        "Do you smoke cigarettes, use tobacco, or vape?",
    ],
    "blurred_vision": [
        "Have you been experiencing any blurred or hazy vision lately?",
        "Is your vision ever blurry or unclear?",
        "Do you notice any blurring in your eyesight?",
    ],
    "floaters": [
        "Do you ever see floating spots, lines, or specks in your vision?",
        "Have you noticed any floaters — dark spots or shapes drifting across your field of view?",
        "Any visual floaters or moving spots in your vision?",
    ],
    "vision_loss": [
        "Have you experienced any loss of vision, even partial?",
        "Do you have any vision loss or areas of blackness in your sight?",
        "Is there any part of your visual field that seems missing or dark?",
    ],
    "eye_pain": [
        "Do you experience any pain, soreness, or discomfort in your eyes?",
        "Are your eyes painful or uncomfortable at any time?",
        "Any eye pain or irritation you'd like to mention?",
    ],
}

# Priority order for asking questions
_QUESTION_PRIORITY = [
    "name",
    "age",
    "gender",
    "diabetes_duration",
    "hba1c",
    "blood_pressure",
    "smoking",
    "blurred_vision",
    "floaters",
    "vision_loss",
    "eye_pain",
    "ldl",
    "hdl",
    "triglycerides",
    "hemoglobin",
]

# The minimum required fields for completion (must mirror schemas.py REQUIRED_FIELDS)
COMPLETION_REQUIRED = ["age", "diabetes_duration", "hba1c", "blood_pressure"]


def _pick_next_field(
    known_structured: Dict[str, Any],
    newly_extracted: Dict[str, Any],
) -> Optional[str]:
    """
    Pick the next field to ask about, using priority order.
    Considers both already-known and freshly extracted data.
    """
    combined = {**known_structured, **newly_extracted}
    for field in _QUESTION_PRIORITY:
        val = combined.get(field)
        if val is None or val == "":
            return field
    return None


def _generate_question(
    field: str,
    name: Optional[str],
    newly_extracted: Dict[str, Any],
) -> str:
    """
    Generate a contextual question for the given field.
    Prepends a brief acknowledgement of what was just extracted (if anything).
    """
    # Build acknowledgement if something was just extracted
    ack = _build_acknowledgement(newly_extracted, name)

    templates = _QUESTION_TEMPLATES.get(field, [f"Could you tell me your {field.replace('_', ' ')}?"])
    question = random.choice(templates)

    # Personalise with name if known and not asking for the name itself
    if name and field != "name":
        personalise_chance = 0.4  # 40% of the time, add the name
        if random.random() < personalise_chance:
            question = f"{name}, {question[0].lower()}{question[1:]}"

    if ack:
        return f"{ack} {question}"
    return question


def _build_acknowledgement(newly_extracted: Dict[str, Any], name: Optional[str]) -> str:
    """
    Build a short, warm acknowledgement sentence based on what was just extracted.
    Returns an empty string if nothing was extracted.
    """
    if not newly_extracted:
        return ""

    keys = list(newly_extracted.keys())

    # Specific acknowledgements for key fields
    if "name" in keys:
        patient_name = newly_extracted["name"]
        return f"Nice to meet you, {patient_name}! 😊"

    if "age" in keys and "diabetes_duration" in keys:
        return "Got it, thank you!"

    if "hba1c" in keys and "blood_pressure" in keys:
        return "Perfect, I've noted those down."

    if "blurred_vision" in keys or "floaters" in keys or "vision_loss" in keys or "eye_pain" in keys:
        return "Thanks for letting me know."

    if "smoking" in keys:
        val = newly_extracted["smoking"]
        if val is False:
            return "Good to know — that's helpful!"
        return "Noted, thank you."

    ack_options = [
        "Thank you!", "Got it!", "Thanks for sharing that.", "Noted, thank you!",
        "I've noted that down.", "Great, thank you!", "Understood!",
    ]
    return random.choice(ack_options)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def extract_information(
    latest_message: str,
    conversation_history: List[Dict[str, Any]],
    known_structured: Dict[str, Any],
    known_dynamic: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Process a patient message and return structured extraction + next question.

    Returns the same schema as gemini_service.extract_information:
    {
        "structured_data": {...},
        "dynamic_data": {...},
        "next_question": "...",
        "intent": "...",
        "reply_override": "...",   # set when intent overrides normal flow
    }
    """
    text = (latest_message or "").strip()
    text_lower = text.lower()

    # Infer known patient name from structured data
    known_name: Optional[str] = known_structured.get("name")

    # Detect intent
    intent = _detect_intent(text_lower)

    # ------------------------------------------------------------------
    # Handle non-data intents: return an intent-specific reply immediately
    # ------------------------------------------------------------------
    missing_required = [
        f for f in COMPLETION_REQUIRED
        if known_structured.get(f) in (None, "")
    ]
    already_completed = len(missing_required) == 0

    if intent == "greeting":
        reply = _greeting_reply(known_name)
        return {
            "structured_data": {},
            "dynamic_data": {},
            "next_question": reply,
            "intent": intent,
            "reply_override": reply,
        }

    if intent == "howru":
        reply = _howru_reply()
        return {
            "structured_data": {},
            "dynamic_data": {},
            "next_question": reply,
            "intent": intent,
            "reply_override": reply,
        }

    if intent == "farewell":
        reply = _farewell_reply(already_completed)
        return {
            "structured_data": {},
            "dynamic_data": {},
            "next_question": reply,
            "intent": intent,
            "reply_override": reply,
        }

    if intent == "confusion":
        reply = _confusion_reply(missing_required, known_name)
        return {
            "structured_data": {},
            "dynamic_data": {},
            "next_question": reply,
            "intent": intent,
            "reply_override": reply,
        }

    if intent == "off_topic":
        # Give the off-topic reply, then follow up with the next question
        off_reply = _off_topic_reply(known_name)
        next_field = _pick_next_field(known_structured, {})
        if next_field:
            follow_up = _generate_question(next_field, known_name, {})
            reply = f"{off_reply}\n\n{follow_up}"
        else:
            reply = off_reply
        return {
            "structured_data": {},
            "dynamic_data": {},
            "next_question": reply,
            "intent": intent,
            "reply_override": reply,
        }

    # ------------------------------------------------------------------
    # "data" intent — extract all clinical fields from the message
    # ------------------------------------------------------------------
    structured: Dict[str, Any] = {}
    dynamic: Dict[str, Any] = {}

    for field, fn in [
        ("age",               _extract_age),
        ("diabetes_duration", _extract_diabetes_duration),
        ("hba1c",             _extract_hba1c),
        ("blood_pressure",    _extract_blood_pressure),
        ("ldl",               _extract_ldl),
        ("hdl",               _extract_hdl),
        ("triglycerides",     _extract_triglycerides),
        ("hemoglobin",        _extract_hemoglobin),
        ("smoking",           _extract_smoking),
        ("blurred_vision",    _extract_blurred_vision),
        ("floaters",          _extract_floaters),
        ("vision_loss",       _extract_vision_loss),
        ("eye_pain",          _extract_eye_pain),
        ("name",              _extract_name),
        ("gender",            _extract_gender),
    ]:
        val = fn(text)
        if val is None:
            # Also try on lowercase for extractors that need it
            val = fn(text_lower)
        if val is not None:
            structured[field] = val

    dynamic.update(_extract_dynamic(text_lower))

    # Context-aware single number disambiguation
    if _search(r"^(\d+(?:\.\d+)?)$", text):
        last_assistant_msg = ""
        for turn in reversed(conversation_history):
            if turn["role"] == "assistant":
                last_assistant_msg = turn["message"].lower()
                break
        
        val = float(text)
        
        if "how long" in last_assistant_msg or "how many years" in last_assistant_msg or "diagnosed" in last_assistant_msg or "diabetic" in last_assistant_msg:
            structured["diabetes_duration"] = val
            if "age" in structured:
                del structured["age"]
        elif "hba1c" in last_assistant_msg or "a1c" in last_assistant_msg or "glycated" in last_assistant_msg:
            if 3.0 <= val <= 20.0:
                structured["hba1c"] = val
            if "age" in structured:
                del structured["age"]
        elif "age" in last_assistant_msg or "old" in last_assistant_msg:
            structured["age"] = int(val)
            if "diabetes_duration" in structured:
                del structured["diabetes_duration"]

    # Handle "affirmation / negation" for pending-confirmation context
    # (main.py handles this, but we note the intent)

    # ------------------------------------------------------------------
    # Generate next question
    # ------------------------------------------------------------------
    next_field = _pick_next_field(known_structured, structured)

    if next_field:
        # Use patient name from newly extracted if not yet known
        effective_name = structured.get("name") or known_name
        next_question = _generate_question(next_field, effective_name, structured)
    else:
        next_question = ""  # all done — main.py will issue completion message

    return {
        "structured_data": structured,
        "dynamic_data": dynamic,
        "next_question": next_question,
        "intent": intent,
    }
