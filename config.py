"""
Configuration for Media Profiler
Aligned with MBFC Credibility & Freedom Methodology
"""
import os
from dataclasses import dataclass

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# =============================================================================
# FILE PATHS
# =============================================================================
FREEDOM_INDEX_FILE = "2025.csv"
FREEDOM_HOUSE_FILE = "FH_FIW.csv"  # Freedom House "Freedom in the World" (World Bank Data360 format)

# =============================================================================
# ISO MAPPING (2-Letter -> 3-Letter)
# =============================================================================
ISO_MAPPING = {
    "KZ": "KAZ", "RU": "RUS", "US": "USA", "GB": "GBR", "UK": "GBR",
    "NZ": "NZL", "CN": "CHN", "FR": "FRA", "DE": "DEU", "CA": "CAN",
    "AU": "AUS", "UA": "UKR", "IL": "ISR", "TR": "TUR", "BY": "BLR",
    "JP": "JPN", "KR": "KOR", "IN": "IND", "BR": "BRA", "ZA": "ZAF"
}

# =============================================================================
# SCORING CONSTANTS
# =============================================================================

# MBFC Credibility Points
CREDIBILITY_POINTS = {
    "factual": {
        "Very High": 4, "High": 3, "Mostly Factual": 2,
        "Mixed": 1, "Low": 0, "Very Low": 0
    },
    "bias": {
        "Least Biased": 3, "Pro-Science": 3,
        "Right-Center": 2, "Left-Center": 2,
        "Right": 1, "Left": 1,
        "Far Right": 1, "Far Left": 1,
        "Extreme": 0, "Questionable": 0
    },
    "traffic": {
        "High": 2, "Medium": 1, "Minimal": 0
    },
    "freedom_penalty": {
        "Limited Freedom": -1,
        "Total Oppression": -2
    }
}

# =============================================================================
# RSF PRESS FREEDOM INDEX — Score thresholds and labels
# =============================================================================
FREEDOM_LABELS = [
    (90, 100, "Excellent Freedom"),
    (70, 89, "Mostly Free"),
    (50, 69, "Moderate Freedom"),
    (25, 49, "Limited Freedom"),
    (0, 24, "Total Oppression"),
]

# Country name aliases — normalizes headquarters input AND Freedom House REF_AREA_LABEL
# to match RSF Country_EN values (the primary lookup key)
COUNTRY_NAME_ALIASES = {
    # Common abbreviations → RSF Country_EN
    "UK": "United Kingdom",
    "US": "United States",
    "USA": "United States",
    "UAE": "United Arab Emirates",
    # Input variants → RSF Country_EN
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire",
    "Turkey": "Türkiye",
    # Freedom House REF_AREA_LABEL → RSF Country_EN
    "Russian Federation": "Russia",
    "Korea, Rep.": "South Korea",
    "Korea, Dem. People's Rep.": "North Korea",
    "Korea, South": "South Korea",
    "Korea, North": "North Korea",
    "Bahamas, The": "Bahamas",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Brunei Darussalam": "Brunei",
    "Congo, Dem. Rep.": "DR Congo",
    "Congo, Rep.": "Congo-Brazzaville",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Egypt, Arab Rep.": "Egypt",
    "Gambia, The": "Gambia",
    "Hong Kong SAR, China": "Hong Kong",
    "Iran, Islamic Rep.": "Iran",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Lao PDR": "Laos",
    "Slovak Republic": "Slovakia",
    "Syrian Arab Republic": "Syria",
    "Taiwan, China": "Taiwan",
    "Timor-Leste": "East Timor",
    "Turkiye": "Türkiye",
    "Venezuela, RB": "Venezuela",
    "Viet Nam": "Vietnam",
    "Yemen, Rep.": "Yemen",
}

# Mapping Weighted Fact Score (0-10, Lower is Better) to Labels
# Per MBFC methodology: 0 only → Very High; 0.1–1.9 → High
FACTUALITY_RANGES = [
    (0.0, 0.0, "Very High"),
    (0.1, 1.9, "High"),
    (2.0, 4.4, "Mostly Factual"),
    (4.5, 6.4, "Mixed"),
    (6.5, 8.4, "Low"),
    (8.5, 10.0, "Very Low")
]

# Mapping Weighted Bias Score (-10 to +10) to Labels
# Per MBFC: Far Left is -7.0 to -7.9 within Left, Far Right is +7.0 to +7.9 within Right
BIAS_RANGES = [
    (-10.0, -8.0, "Extreme Left"),
    (-7.9, -7.0, "Far Left"),
    (-6.9, -5.0, "Left"),
    (-4.9, -2.0, "Left-Center"),
    (-1.9, 1.9, "Least Biased"),
    (2.0, 4.9, "Right-Center"),
    (5.0, 6.9, "Right"),
    (7.0, 7.9, "Far Right"),
    (8.0, 10.0, "Extreme Right")
]

# =============================================================================
# ECONOMIC & SOCIAL SCALES
# =============================================================================
ECONOMIC_SCALE = {
    "Communism": -10.0, "Socialism": -7.5, "Democratic Socialism": -5.0,
    "Regulated Market Economy": -2.5, "Centrism": 0.0,
    "Moderately Regulated Capitalism": 2.5, "Classical Liberalism": 5.0,
    "Libertarianism": 7.5, "Radical Laissez-Faire": 10.0
}

SOCIAL_SCALE = {
    "Strong Progressive": -10.0, "Progressive": -7.5, "Moderate Progressive": -5.0,
    "Mild Progressive": -2.5, "Balanced": 0.0,
    "Mild Conservative": 2.5, "Moderate Conservative": 5.0,
    "Traditional Conservative": 7.5, "Strong Traditional Conservative": 10.0
}

# =============================================================================
# NEWS REPORTING BALANCE SCALE (15% of Bias Score)
# =============================================================================
# Measures how well a source reports all sides in its straight news stories
NEWS_REPORTING_SCALE = {
    "Extreme Left Reporting": -10.0,
    "Strong Left Reporting": -7.5,
    "Moderate Left Reporting": -5.0,
    "Mild Left Reporting": -2.5,
    "Neutral/Balanced": 0.0,
    "Mild Right Reporting": 2.5,
    "Moderate Right Reporting": 5.0,
    "Strong Right Reporting": 7.5,
    "Extreme Right Reporting": 10.0
}

# =============================================================================
# EDITORIAL BIAS SCALE (15% of Bias Score)
# =============================================================================
# Evaluates bias in opinion pieces, editorials, and use of loaded language
EDITORIAL_BIAS_SCALE = {
    "Extreme Left Editorial": -10.0,
    "Strong Left Editorial": -7.5,
    "Moderate Left Editorial": -5.0,
    "Mild Left Editorial": -2.5,
    "Neutral/Balanced Editorial": 0.0,
    "Mild Right Editorial": 2.5,
    "Moderate Right Editorial": 5.0,
    "Strong Right Editorial": 7.5,
    "Extreme Right Editorial": 10.0
}

# =============================================================================
# SOURCING QUALITY SCALE (25% of Factuality Score)
# =============================================================================
# Scale 0-10 where 0 is perfect sourcing, 10 is no sourcing
SOURCING_DESCRIPTIONS = {
    0: "Perfect sourcing; highly credible references with clear citations",
    1: "Almost perfect sourcing with minor inconsistencies",
    2: "Mostly credible sourcing but occasional lapses",
    3: "Generally credible but sourcing issues occur more frequently",
    4: "Mostly credible but noticeable reliance on less trustworthy sources",
    5: "Mixed sourcing including credible and questionable references",
    6: "Moderate sourcing issues with frequent reliance on less credible sources",
    7: "Limited sourcing mostly relying on questionable references",
    8: "Very limited sourcing heavily reliant on discredited sources",
    9: "Minimal sourcing using widely discredited sources",
    10: "No sourcing or complete reliance on discredited sources"
}

# =============================================================================
# IFCN APPROVED FACT CHECKERS
# =============================================================================
# International Fact-Checking Network approved organizations.
# NOTE: mediabiasfactcheck.com is intentionally EXCLUDED — using MBFC data
# as an input would contaminate our independent MBFC-methodology evaluation.
IFCN_FACT_CHECKERS = [
    "politifact.com",
    "factcheck.org",
    "snopes.com",
    "apnews.com/ap-fact-check",
    "reuters.com/fact-check",
    "washingtonpost.com/news/fact-checker",
    "fullfact.org",
    "checkyourfact.com",
    "leadstories.com",
    "africacheck.org",
    "truthorfiction.com",
]

# Sites to search for fact checks on a specific outlet.
# Imported by FactCheckSearcher — single source of truth.
# mediabiasfactcheck.com is EXCLUDED (anti-contamination).
FACTCHECK_SEARCH_SITES = [
    "politifact.com",
    "snopes.com",
    "factcheck.org",
    "fullfact.org",
    "reuters.com/fact-check",
    "apnews.com/ap-fact-check",
    "leadstories.com",
    "factcheck.kz",
]

# Search queries for fact checking
FACT_CHECK_SEARCH_TERMS = [
    "fact check",
    "false claim",
    "misinformation",
    "debunked",
    "misleading",
]

# When True, FactCheckSearcher additionally classifies WHO published each
# fact-checked claim and counts a failure ONLY when the outlet itself published it
# (claims merely about / falsely attributed to the outlet are not counted).
FACTCHECK_ATTRIBUTE_CLAIM_SOURCE = True

# =============================================================================
# MINIMUM EVIDENCE REQUIREMENTS (MBFC methodology)
# =============================================================================
MIN_HEADLINES = 10   # Minimum headline observations before scoring
MIN_FULL_STORIES = 5 # Minimum full-text stories before scoring

# =============================================================================
# QUESTIONABLE SOURCE THRESHOLDS (MBFC methodology)
# =============================================================================
# Sources displaying extreme bias, propaganda, unreliable sourcing, or lack
# of transparency. Sources lacking transparency in mission, ownership, or
# authorship are automatically categorized as questionable.
QUESTIONABLE_EXTREME_BIAS_THRESHOLD = 8.0   # abs(bias_score) >= this
QUESTIONABLE_TRANSPARENCY_THRESHOLD = 8.0   # transparency_score >= this
QUESTIONABLE_PROPAGANDA_THRESHOLD = 7.0     # one_sidedness_score >= this

# =============================================================================
# SOURCE CATEGORIES (MBFC classification)
# =============================================================================
SOURCE_CATEGORIES = [
    "News",
    "Questionable",
    "Conspiracy/Pseudoscience",
    "Pro-Science",
    "Satire",
]

# =============================================================================
# ANTI-CONTAMINATION — Domains to exclude from search results
# =============================================================================
EXCLUDED_DOMAINS = [
    "mediabiasfactcheck.com",
    "allsides.com",
    "adfontesmedia.com",
    "adfontes.media",
    "wikipedia.org",
    "rationalwiki.org",
    "ground.news",
    "thecredibilitycoalition.org",
    "newsguardtech.com",
    "realorsatire.com",
    "fakenewscodex.com",
]

SOCIAL_MEDIA_DOMAINS = [
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "tiktok.com",
    "pinterest.com",
    "linkedin.com",
    "reddit.com",
    "youtube.com",
]

# =============================================================================
# SCORING WEIGHT DATACLASSES
# =============================================================================

@dataclass
class BiasWeights:
    """MBFC 2025 media outlet bias scoring weights."""
    economic: float = 0.35
    social: float = 0.35
    reporting: float = 0.15
    editorial: float = 0.15

@dataclass
class FactualWeights:
    """MBFC 2025 media outlet factuality scoring weights."""
    failed_fact_checks: float = 0.40
    sourcing: float = 0.25
    transparency: float = 0.25
    bias_propaganda: float = 0.10

@dataclass
class PoliticianBiasWeights:
    """MBFC Politician Rating: Economic (50%) + Social (50%)."""
    economic: float = 0.50
    social: float = 0.50

@dataclass
class PollsterBiasWeights:
    """MBFC 2026 Pollster Rating: Polling Bias (70%) + Editorial Bias (30%)."""
    polling: float = 0.70
    editorial: float = 0.30

# =============================================================================
# MBFC CATEGORY DESCRIPTIONS (For Report Generation)
# =============================================================================

BIAS_CATEGORY_DESCRIPTIONS = {
    "Extreme Left": (
        "EXTREME LEFT\n"
        "These sources exclusively promote left-wing policies and rarely cite credible "
        "sources. They may use strong loaded language and appeal to emotion. Most fail "
        "fact checks and do not correct errors."
    ),
    "Far Left": (
        "FAR LEFT\n"
        "These sources strongly favor liberal causes through story selection and/or "
        "political affiliation. They may utilize strong loaded words (wording that "
        "attempts to influence an audience by appeals to emotion or stereotypes), "
        "publish misleading reports, and omit information that may damage liberal "
        "causes. Some sources in this category may be untrustworthy."
    ),
    "Left": (
        "LEFT BIAS\n"
        "These sources moderately to strongly favor liberal perspectives. They may "
        "utilize strong loaded words (wording that attempts to influence an audience "
        "by appeals to emotion or stereotypes), publish misleading reports, and omit "
        "information that may damage liberal causes. Some sources in this category "
        "may be untrustworthy."
    ),
    "Left-Center": (
        "LEFT-CENTER BIAS\n"
        "These sources have a slight to moderate liberal bias. They often publish "
        "factual information that utilizes loaded words (wording that attempts to "
        "influence an audience by appeals to emotion or stereotypes) to favor liberal "
        "causes. These sources are generally trustworthy for information but may "
        "require further investigation."
    ),
    "Least Biased": (
        "LEAST BIASED\n"
        "These sources have minimal bias and use very few loaded words (wording that "
        "attempts to influence an audience by appeals to emotion or stereotypes). "
        "The reporting is factual and usually sourced. These are the most credible "
        "media sources."
    ),
    "Right-Center": (
        "RIGHT-CENTER BIAS\n"
        "These sources have a slight to moderate conservative bias. They often publish "
        "factual information that utilizes loaded words (wording that attempts to "
        "influence an audience by appeals to emotion or stereotypes) to favor "
        "conservative causes. These sources are generally trustworthy for information "
        "but may require further investigation."
    ),
    "Right": (
        "RIGHT BIAS\n"
        "These sources moderately to strongly favor conservative perspectives. They may "
        "utilize strong loaded words (wording that attempts to influence an audience "
        "by appeals to emotion or stereotypes), publish misleading reports, and omit "
        "information that may damage conservative causes. Some sources in this category "
        "may be untrustworthy."
    ),
    "Far Right": (
        "FAR RIGHT\n"
        "These sources strongly favor conservative causes through story selection and/or "
        "political affiliation. They may utilize strong loaded words (wording that "
        "attempts to influence an audience by appeals to emotion or stereotypes), "
        "publish misleading reports, and omit information that may damage conservative "
        "causes. Some sources in this category may be untrustworthy."
    ),
    "Extreme Right": (
        "EXTREME RIGHT\n"
        "These sources exclusively promote right-wing policies and rarely cite credible "
        "sources. They may use strong loaded language and appeal to emotion. Most fail "
        "fact checks and do not correct errors."
    )
}

FACTUALITY_DESCRIPTIONS = {
    "Very High": "Sources that always use credible sources, are well-sourced, and have a clean fact check record.",
    "High": "Sources that are generally reliable with minimal failed fact checks and good sourcing practices.",
    "Mostly Factual": "Sources that are generally reliable but may have occasional minor errors or unsourced claims.",
    "Mixed": "Sources that do not always use proper sourcing or have multiple failed fact checks.",
    "Low": "Sources that rarely use credible sources and have numerous failed fact checks.",
    "Very Low": "Sources that consistently fail fact checks and promote misinformation."
}

CREDIBILITY_DESCRIPTIONS = {
    "High Credibility": "This source has earned a high credibility rating based on factual reporting, minimal bias, and transparent practices.",
    "Medium Credibility": "This source has a moderate credibility rating. While generally reliable, some caution is advised.",
    "Low Credibility": "This source has a low credibility rating due to failed fact checks, high bias, or lack of transparency."
}

# =============================================================================
# POLLSTER RATING CONSTANTS (MBFC 2026 Methodology)
# =============================================================================

# Polling bias calibration scale: maps absolute mean-reverted bias to score
POLLING_BIAS_CALIBRATION = [
    (0.0, 0.5, 1),
    (0.6, 1.0, 2),
    (1.1, 1.5, 3),
    (1.6, 2.0, 4),
    (2.1, 2.5, 5),
    (2.6, 3.0, 6),
    (3.1, 3.5, 7),
    (3.6, 4.0, 8),
    (4.1, 4.5, 9),
    (4.6, 10.0, 10),
]

# Silver Bulletin letter grade → MBFC factuality mapping
POLLSTER_FACTUALITY_GRADES = {
    "A+": "Very High",
    "A": "High",
    "A-": "High",
    "B+": "Mostly Factual",
    "B": "Mostly Factual",
    "B-": "Mostly Factual",
    "C+": "Mixed",
    "C": "Mixed",
    "C-": "Mixed",
    "D+": "Low",
    "D": "Low",
    "D-": "Low",
    "F": "Very Low",
}

# Pollster bias category thresholds (same as media but applied to final pollster score)
POLLSTER_BIAS_RANGES = [
    (0.0, 1.9, "Least Biased"),
    (2.0, 4.9, "Left-Center / Right-Center"),
    (5.0, 7.9, "Left / Right"),
    (8.0, 10.0, "Extreme Left / Extreme Right"),
]

# =============================================================================
# POLITICIAN RATING CONSTANTS (MBFC Methodology)
# =============================================================================

# Politician factuality: based solely on failed fact checks count.
# NOTE: MBFC text has an overlap at count 3 ("2-3: Mostly Factual" and
# "3-7: Mixed"). We preserve this verbatim. Sequential iteration means
# count=3 resolves to "Mostly Factual" (checked first), which aligns with
# the charitable interpretation of a borderline count.
POLITICIAN_FACTUALITY_RANGES = [
    (0, 0, "Very High"),
    (1, 1, "High"),
    (2, 3, "Mostly Factual"),
    (3, 7, "Mixed"),
    (8, 9, "Low"),
    (10, 999, "Very Low"),
]

# =============================================================================
# COUNTRY NAME MAPPING
# =============================================================================
COUNTRY_NAMES = {
    "US": "United States", "GB": "United Kingdom", "CA": "Canada",
    "AU": "Australia", "DE": "Germany", "FR": "France", "IT": "Italy",
    "ES": "Spain", "NL": "Netherlands", "BE": "Belgium", "CH": "Switzerland",
    "AT": "Austria", "SE": "Sweden", "NO": "Norway", "DK": "Denmark",
    "FI": "Finland", "IE": "Ireland", "NZ": "New Zealand", "JP": "Japan",
    "KR": "South Korea", "CN": "China", "IN": "India", "BR": "Brazil",
    "MX": "Mexico", "RU": "Russia", "ZA": "South Africa", "KZ": "Kazakhstan",
    "PL": "Poland", "CZ": "Czech Republic", "HU": "Hungary", "RO": "Romania",
    "GR": "Greece", "PT": "Portugal", "IL": "Israel", "AE": "United Arab Emirates",
    "SA": "Saudi Arabia", "EG": "Egypt", "NG": "Nigeria", "KE": "Kenya",
    "UA": "Ukraine", "TR": "Turkey", "TH": "Thailand", "VN": "Vietnam",
    "PH": "Philippines", "ID": "Indonesia", "MY": "Malaysia", "SG": "Singapore"
}

# =============================================================================
# Parallelism Settings
# =============================================================================
MAX_PARALLEL_LLM_CALLS = 3  # Cap concurrent LLM calls (rate limit safety)
