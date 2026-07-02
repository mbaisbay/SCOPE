"""
Pydantic v2 schemas for structured LLM outputs in media bias analysis.

These schemas are designed to work with LangChain's .with_structured_output()
method to ensure deterministic, type-safe responses from LLM calls.
"""

from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Article Classification Schemas
# =============================================================================


class ArticleType(str, Enum):
    """Classification of article type based on content analysis."""

    NEWS = "News"
    OPINION = "Opinion"
    SATIRE = "Satire"
    PR = "PR"  # Press Release / Promotional content


class ArticleClassification(BaseModel):
    """
    Structured output for article type classification.

    Used by OpinionAnalyzer to classify articles into distinct categories
    based on content analysis (not URL/title heuristics).
    """

    article_type: ArticleType = Field(
        description="The classified type of the article"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen"
    )


class BatchArticleClassification(BaseModel):
    """Batch classification of multiple articles in a single LLM call."""

    classifications: list[ArticleClassification] = Field(
        description="One classification per article, in the same order as the input articles"
    )


# =============================================================================
# Media Type Classification Schemas
# =============================================================================


class MediaType(str, Enum):
    """Classification of media outlet type."""

    TV = "TV"
    NEWSPAPER = "Newspaper"
    WEBSITE = "Website"
    MAGAZINE = "Magazine"
    RADIO = "Radio"
    NEWS_AGENCY = "News Agency"
    BLOG = "Blog"
    PODCAST = "Podcast"
    STREAMING = "Streaming Service"
    UNKNOWN = "Unknown"


class MediaTypeSource(str, Enum):
    """Source of media type classification."""

    LOOKUP = "Lookup"  # From known_media_types.csv (deterministic)
    LLM = "LLM"  # From search + LLM parsing
    FALLBACK = "Fallback"  # Default when no data available


class MediaTypeLLMOutput(BaseModel):
    """
    Structured output for LLM media type parsing.

    This is the schema used by the LLM when parsing search results.
    It does not include 'source' since that's determined by the analyzer.
    """

    media_type: MediaType = Field(
        description="The type of media outlet"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        description="Brief explanation of how the media type was determined from search results"
    )


class MediaTypeClassification(BaseModel):
    """
    Complete media type classification result.

    Used by MediaTypeAnalyzer to return classification results from
    either lookup table or web search + LLM.
    """

    media_type: MediaType = Field(
        description="The type of media outlet"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    source: MediaTypeSource = Field(
        default=MediaTypeSource.FALLBACK,
        description="Source of the classification (Lookup, LLM, or Fallback)"
    )
    source_snippet: Optional[str] = Field(
        default=None,
        description="The relevant snippet from search results (LLM method only)"
    )
    reasoning: str = Field(
        description="Brief explanation of how the media type was determined"
    )


# =============================================================================
# Traffic and Longevity Schemas
# =============================================================================


class TrafficTier(str, Enum):
    """Traffic level classification for websites."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    MINIMAL = "Minimal"
    UNKNOWN = "Unknown"


class TrafficEstimate(BaseModel):
    """
    Structured output for traffic level estimation from search snippets.

    Used by TrafficLongevityAnalyzer to parse traffic information
    from DuckDuckGo search results.
    """

    traffic_tier: TrafficTier = Field(
        description="Estimated traffic tier based on search results"
    )
    monthly_visits_estimate: Optional[str] = Field(
        default=None,
        description="Estimated monthly visits if mentioned (e.g., '10M', '500K')"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        description="Explanation of how traffic tier was determined from the snippet"
    )


class TrafficSource(str, Enum):
    """Source of traffic data."""

    TRANCO = "Tranco"  # Deterministic ranking from Tranco list
    LLM = "LLM"  # LLM-parsed from search results
    FALLBACK = "Fallback"  # Default when no data available


class TrafficData(BaseModel):
    """
    Complete traffic and longevity data for a domain.

    Combines deterministic data from multiple sources:
    - WHOIS for domain age
    - Tranco list for deterministic traffic ranking (when available)
    - LLM-parsed search results as fallback
    """

    domain: str = Field(
        description="The domain being analyzed"
    )
    creation_date: Optional[date] = Field(
        default=None,
        description="Domain creation date from WHOIS lookup"
    )
    age_years: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Age of the domain in years"
    )
    traffic_tier: TrafficTier = Field(
        description="Estimated traffic tier"
    )
    monthly_visits_estimate: Optional[str] = Field(
        default=None,
        description="Estimated monthly visits if available"
    )
    traffic_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in traffic tier estimate"
    )
    traffic_source: TrafficSource = Field(
        default=TrafficSource.FALLBACK,
        description="Source of the traffic data (Tranco, LLM, or Fallback)"
    )
    tranco_rank: Optional[int] = Field(
        default=None,
        description="Tranco list rank if found (1 = most popular)"
    )
    whois_success: bool = Field(
        description="Whether WHOIS lookup was successful"
    )
    whois_error: Optional[str] = Field(
        default=None,
        description="WHOIS error message if lookup failed"
    )
    traffic_search_snippet: Optional[str] = Field(
        default=None,
        description="The search snippet used for traffic estimation (LLM method only)"
    )


# =============================================================================
# Validation Dataset Schemas
# =============================================================================


class GoldenDatasetEntry(BaseModel):
    """
    Schema for entries in the opinion classification validation dataset.
    """

    url: str = Field(
        description="URL of the article"
    )
    title: str = Field(
        description="Title/headline of the article"
    )
    text_snippet: str = Field(
        description="First ~1000 characters of article text"
    )
    expected_label: ArticleType = Field(
        description="The expected/ground truth classification"
    )


class ValidationResult(BaseModel):
    """
    Result of validating a single article classification.
    """

    url: str = Field(
        description="URL of the article tested"
    )
    expected: ArticleType = Field(
        description="Expected classification"
    )
    predicted: ArticleType = Field(
        description="Predicted classification from analyzer"
    )
    confidence: float = Field(
        description="Confidence of the prediction"
    )
    is_correct: bool = Field(
        description="Whether prediction matched expected"
    )
    reasoning: str = Field(
        description="Model's reasoning for the classification"
    )


class ValidationReport(BaseModel):
    """
    Complete validation report for the golden dataset.
    """

    total_samples: int = Field(
        description="Total number of samples tested"
    )
    correct_count: int = Field(
        description="Number of correct predictions"
    )
    accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall accuracy (correct/total)"
    )
    results: list[ValidationResult] = Field(
        description="Individual results for each sample"
    )
    mismatches: list[ValidationResult] = Field(
        description="Only the incorrect predictions"
    )


# =============================================================================
# Fact Check Schemas
# =============================================================================


class FactCheckVerdict(str, Enum):
    """Verdict from a fact-checker."""

    TRUE = "True"
    MOSTLY_TRUE = "Mostly True"
    HALF_TRUE = "Half True"
    MIXED = "Mixed"
    MOSTLY_FALSE = "Mostly False"
    FALSE = "False"
    PANTS_ON_FIRE = "Pants on Fire"
    UNPROVEN = "Unproven"
    MISLEADING = "Misleading"
    NOT_RATED = "Not Rated"


class FactCheckSource(str, Enum):
    """Source of fact check data."""

    SEARCH = "Search"  # From direct fact-checker site search
    FALLBACK = "Fallback"  # No data found


class ClaimSource(str, Enum):
    """Who originally published the fact-checked claim, relative to the outlet.

    A failed fact check should only count against an outlet when the outlet itself
    published the claim (``PUBLISHED_BY_OUTLET``). Claims that are merely about the
    outlet (hoaxes, impersonation, false attribution) or that come from unrelated
    third parties must NOT be counted as the outlet's own failures.
    """

    PUBLISHED_BY_OUTLET = "published_by_outlet"  # outlet authored/published the claim
    ABOUT_OUTLET = "about_outlet"  # claim is about / falsely attributed to the outlet
    THIRD_PARTY = "third_party"  # claim merely mentions the outlet
    UNKNOWN = "unknown"  # could not be determined


class FactCheckFinding(BaseModel):
    """A single fact check finding parsed from search results."""

    source_site: str = Field(
        description="The fact-checking organization (e.g., 'PolitiFact', 'Snopes')"
    )
    claim_summary: str = Field(
        description="Brief summary of the claim that was fact-checked"
    )
    verdict: FactCheckVerdict = Field(
        description="The verdict given by the fact-checker"
    )
    url: Optional[str] = Field(
        default=None,
        description="URL to the fact-check article if available"
    )
    # --- Claim-source attribution (default-safe so legacy data still deserializes) ---
    claim_source: ClaimSource = Field(
        default=ClaimSource.UNKNOWN,
        description="Whether the checked claim was published by the outlet, is about it, "
        "or comes from a third party"
    )
    claim_source_domain: Optional[str] = Field(
        default=None,
        description="Domain that originally published the checked claim, if identifiable"
    )
    published_by_outlet: bool = Field(
        default=False,
        description="True only if the fact-checked claim was published by the outlet itself"
    )
    attribution_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the claim-source attribution"
    )


class FactCheckLLMOutput(BaseModel):
    """Structured LLM output for parsing fact-check search results."""

    findings: list[FactCheckFinding] = Field(
        default_factory=list,
        description="List of fact check findings extracted from search results"
    )
    failed_count: int = Field(
        ge=0,
        description="Number of FALSE/MOSTLY_FALSE/PANTS_ON_FIRE/MISLEADING verdicts"
    )
    total_count: int = Field(
        ge=0,
        description="Total number of fact checks found"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the parsing accuracy"
    )
    reasoning: str = Field(
        description="Explanation of findings and any ambiguities"
    )


class FactCheckAnalysisResult(BaseModel):
    """Complete fact check analysis result."""

    domain: str = Field(
        description="The domain that was analyzed"
    )
    outlet_name: Optional[str] = Field(
        default=None,
        description="Human-readable outlet name if known"
    )
    failed_checks_count: int = Field(
        ge=0,
        description="Number of failed fact checks found"
    )
    total_checks_count: int = Field(
        ge=0,
        description="Total number of fact checks found"
    )
    score: float = Field(
        ge=0.0,
        le=10.0,
        description="MBFC-style score (0=excellent, 10=very poor)"
    )
    source: FactCheckSource = Field(
        default=FactCheckSource.FALLBACK,
        description="Source of the fact check data"
    )
    findings: list[FactCheckFinding] = Field(
        default_factory=list,
        description="Individual fact check findings"
    )
    search_snippets: Optional[str] = Field(
        default=None,
        description="Combined search snippets used for analysis"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence in the analysis"
    )
    reasoning: str = Field(
        description="Explanation of the fact check analysis"
    )
    coverage_sufficient: bool = Field(
        default=True,
        description="Whether fact-check search coverage was sufficient. "
        "False when no fact-check evidence was found at all "
        "(meaning the score is a default, not an observed measurement)."
    )
    about_outlet_count: int = Field(
        default=0,
        ge=0,
        description="Number of findings whose claim is ABOUT the outlet (hoaxes, "
        "impersonation, false attribution) rather than published by it. Not counted "
        "as failed checks. Only populated when claim-source attribution is enabled."
    )


# =============================================================================
# Sourcing Quality Schemas
# =============================================================================


class SourceQuality(str, Enum):
    """Quality tier of a source."""

    PRIMARY = "Primary"  # Original documents, official statements, studies
    WIRE_SERVICE = "Wire Service"  # Reuters, AP, AFP - highly credible
    MAJOR_OUTLET = "Major Outlet"  # NYT, BBC, WSJ - established outlets
    CREDIBLE = "Credible"  # Other established outlets with standards
    UNKNOWN = "Unknown"  # Cannot assess or unfamiliar
    QUESTIONABLE = "Questionable"  # Known unreliable sources


class SourceAssessment(BaseModel):
    """Assessment of a single source."""

    domain: str = Field(
        description="The domain of the source"
    )
    quality: SourceQuality = Field(
        description="Quality tier of this source"
    )
    reasoning: str = Field(
        description="Brief explanation of the quality assessment"
    )


class SourcingLLMOutput(BaseModel):
    """Structured LLM output for sourcing analysis."""

    sources_assessed: list[SourceAssessment] = Field(
        default_factory=list,
        description="Assessment of each unique source domain OR named entity found in text"
    )
    
    # --- NEW FIELDS START ---
    vague_sourcing_detected: bool = Field(
        default=False,
        description="Whether articles rely on vague phrases like 'experts say', 'sources claim' without naming them"
    )
    vague_sourcing_examples: list[str] = Field(
        default_factory=list,
        description="Examples of vague attribution phrases found"
    )
    # --- NEW FIELDS END ---
    
    overall_quality_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Overall sourcing quality (0=excellent, 10=poor)"
    )
    has_primary_sources: bool = Field(
        description="Whether primary sources (official docs, studies) are cited"
    )
    has_wire_services: bool = Field(
        description="Whether wire services (Reuters, AP) are cited"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the assessment"
    )
    overall_assessment: str = Field(
        description="Summary assessment of sourcing practices"
    )


class SourcingAnalysisResult(BaseModel):
    """Complete sourcing analysis result."""

    score: float = Field(
        ge=0.0,
        le=10.0,
        description="MBFC-style score (0=excellent, 10=poor)"
    )
    avg_sources_per_article: float = Field(
        ge=0.0,
        description="Average number of sources cited per article"
    )
    total_sources_found: int = Field(
        ge=0,
        description="Total number of source links found"
    )
    unique_domains: int = Field(
        ge=0,
        description="Number of unique source domains"
    )
    has_hyperlinks: bool = Field(
        description="Whether articles contain hyperlinks to sources"
    )
    source_assessments: list[SourceAssessment] = Field(
        default_factory=list,
        description="Individual source quality assessments"
    )
    has_primary_sources: bool = Field(
        default=False,
        description="Whether primary sources are cited"
    )
    has_wire_services: bool = Field(
        default=False,
        description="Whether wire services are cited"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the analysis"
    )
    reasoning: str = Field(
        description="Explanation of the sourcing analysis"
    )


# =============================================================================
# Editorial Bias Schemas
# =============================================================================


class BiasDirection(str, Enum):
    """Political bias direction on the left-right spectrum."""

    EXTREME_LEFT = "Extreme Left"
    LEFT = "Left"
    LEFT_CENTER = "Left-Center"
    CENTER = "Center"
    RIGHT_CENTER = "Right-Center"
    RIGHT = "Right"
    EXTREME_RIGHT = "Extreme Right"


class PolicyDomain(str, Enum):
    """Major policy domains for bias assessment."""

    ECONOMIC = "Economic Policy"
    SOCIAL = "Social Issues"
    ENVIRONMENTAL = "Environmental Policy"
    HEALTHCARE = "Healthcare"
    IMMIGRATION = "Immigration"
    FOREIGN_POLICY = "Foreign Policy"
    GUN_RIGHTS = "Gun Rights"
    EDUCATION = "Education"


class PolicyPosition(BaseModel):
    """Assessment of outlet's position on a specific policy domain."""

    domain: PolicyDomain = Field(
        description="The policy domain being assessed"
    )
    leaning: BiasDirection = Field(
        description="The detected leaning on this policy"
    )
    indicators: list[str] = Field(
        default_factory=list,
        description="Specific indicators or quotes showing this position"
    )
    source_articles: list[str] = Field(
        default_factory=list,
        description="Titles of the articles where this evidence was found (e.g., 'Article 1: Title Here')"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this assessment"
    )


class EditorialBiasLLMOutput(BaseModel):
    """Structured LLM output for editorial bias analysis.

    Returns 4 MBFC subcategory scores for weighted bias calculation:
      Economic System (35%) + Social Values (35%) +
      Straight News Reporting Balance (15%) + Editorial/Op-Ed Bias (15%)
    """

    overall_bias: BiasDirection = Field(
        description="Overall editorial bias direction"
    )
    bias_score: float = Field(
        ge=-10.0,
        le=10.0,
        description="Overall combined bias score (for backward compat; weighted score computed externally)"
    )
    # --- MBFC 4-category subscores ---
    economic_score: float = Field(
        ge=-10.0, le=10.0,
        description="Economic System score: -10 (communism) to +10 (laissez-faire capitalism)"
    )
    social_score: float = Field(
        ge=-10.0, le=10.0,
        description="Social Progressive vs Traditional Conservative score: -10 (strong progressive) to +10 (strong traditional conservative)"
    )
    news_reporting_score: float = Field(
        ge=-10.0, le=10.0,
        description="Straight News Reporting Balance score: -10 (extreme left reporting) to +10 (extreme right reporting)"
    )
    editorial_bias_score: float = Field(
        ge=-10.0, le=10.0,
        description="Editorial/Op-Ed Bias score: -10 (extreme left editorial) to +10 (extreme right editorial)"
    )
    # --- end MBFC subscores ---
    policy_positions: list[PolicyPosition] = Field(
        default_factory=list,
        description="Positions on specific policy domains if detectable"
    )
    uses_loaded_language: bool = Field(
        description="Whether outlet uses politically loaded language"
    )
    loaded_language_examples: list[str] = Field(
        default_factory=list,
        description="Examples of loaded language found, with article reference (e.g., 'regime change (Article 3: Title)')"
    )
    story_selection_bias: Optional[str] = Field(
        default=None,
        description="Notes on biased story selection patterns if detected"
    )
    ideology_summary: str = Field(
        default="",
        description="2-3 sentence direct summary of the outlet's ideological position (e.g., 'The outlet advocates for progressive policies including expanded healthcare and environmental regulation.')"
    )
    economy_summary: str = Field(
        default="",
        description="1-2 sentence direct summary of the outlet's economic stance (e.g., 'Economically, the outlet supports free-market capitalism with minimal government intervention.')"
    )
    is_pro_science: bool = Field(
        default=False,
        description="Whether the outlet is a peer-reviewed, evidence-based scientific publication"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence in bias assessment"
    )
    reasoning: str = Field(
        description="Detailed explanation of bias assessment"
    )


class EditorialBiasResult(BaseModel):
    """Complete editorial bias analysis result with MBFC 4-category subscores."""

    domain: str = Field(
        description="The domain that was analyzed"
    )
    outlet_name: Optional[str] = Field(
        default=None,
        description="Human-readable outlet name"
    )
    overall_bias: BiasDirection = Field(
        description="Overall editorial bias direction"
    )
    bias_score: float = Field(
        ge=-10.0,
        le=10.0,
        description="Weighted composite bias score (0.35×econ + 0.35×social + 0.15×news + 0.15×editorial)"
    )
    # --- MBFC 4-category subscores ---
    economic_score: float = Field(
        default=0.0, ge=-10.0, le=10.0,
        description="Economic System score (-10 to +10)"
    )
    social_score: float = Field(
        default=0.0, ge=-10.0, le=10.0,
        description="Social Progressive vs Traditional Conservative score (-10 to +10)"
    )
    news_reporting_score: float = Field(
        default=0.0, ge=-10.0, le=10.0,
        description="Straight News Reporting Balance score (-10 to +10)"
    )
    editorial_bias_score: float = Field(
        default=0.0, ge=-10.0, le=10.0,
        description="Editorial/Op-Ed Bias score (-10 to +10)"
    )
    # --- end subscores ---
    mbfc_label: str = Field(
        description="MBFC-style label (Left, Left-Center, Center, etc.)"
    )
    policy_positions: list[PolicyPosition] = Field(
        default_factory=list,
        description="Positions on specific policy domains"
    )
    uses_loaded_language: bool = Field(
        default=False,
        description="Whether outlet uses loaded language"
    )
    loaded_language_examples: list[str] = Field(
        default_factory=list,
        description="Examples of loaded language"
    )
    story_selection_bias: Optional[str] = Field(
        default=None,
        description="Notes on story selection bias"
    )
    ideology_summary: str = Field(
        default="",
        description="Direct summary of the outlet's ideological position"
    )
    economy_summary: str = Field(
        default="",
        description="Direct summary of the outlet's economic stance"
    )
    is_pro_science: bool = Field(
        default=False,
        description="Whether the outlet is a peer-reviewed, evidence-based scientific publication"
    )
    articles_analyzed: int = Field(
        ge=0,
        description="Number of articles analyzed"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the assessment"
    )
    reasoning: str = Field(
        description="Explanation of bias assessment"
    )


# =============================================================================
# Pseudoscience Detection Schemas
# =============================================================================


class PseudoscienceCategory(str, Enum):
    """Categories of pseudoscientific content."""

    # Health-related pseudoscience
    ANTI_VACCINATION = "Anti-Vaccination"
    ALTERNATIVE_MEDICINE = "Alternative Medicine"
    CANCER_CURE_CLAIMS = "Alternative Cancer Treatments"
    AIDS_DENIALISM = "AIDS Denialism"
    COVID_MISINFORMATION = "COVID-19 Misinformation"
    MASK_MISINFORMATION = "Mask Misinformation"
    HOMEOPATHY = "Homeopathy"
    DETOX_CLAIMS = "Detoxification Claims"
    ESSENTIAL_OILS_CURE = "Essential Oils Cure Claims"

    # Climate/Environment
    CLIMATE_DENIAL = "Climate Change Denialism"
    GMO_DANGERS = "GMO Danger Claims"
    CHEMTRAILS = "Chemtrails Conspiracy"
    FIVE_G_HEALTH = "5G Health Conspiracy"

    # Paranormal/Supernatural
    ASTROLOGY = "Astrology"
    PSYCHIC_CLAIMS = "Psychic Claims"
    ANCIENT_ASTRONAUTS = "Ancient Astronauts"
    CRYSTAL_HEALING = "Crystal Healing"
    FAITH_HEALING = "Faith Healing"

    # Conspiracy theories
    FLAT_EARTH = "Flat Earth"
    MOON_LANDING_HOAX = "Moon Landing Conspiracy"
    DEEP_STATE = "Deep State Conspiracy"
    NEW_WORLD_ORDER = "New World Order"
    QAnon = "QAnon"

    # Other
    PSEUDOARCHAEOLOGY = "Pseudoarchaeology"
    CRYPTOZOOLOGY = "Cryptozoology"
    NUMEROLOGY = "Numerology"
    OTHER = "Other Pseudoscience"


class PseudoscienceSeverity(str, Enum):
    """Severity of pseudoscience promotion."""

    PROMOTES = "Promotes"  # Actively promotes pseudoscience as fact
    PRESENTS_UNCRITICALLY = "Presents Uncritically"  # Reports without debunking
    MIXED = "Mixed"  # Sometimes promotes, sometimes critical
    NONE_DETECTED = "None Detected"  # No pseudoscience found


class PseudoscienceIndicator(BaseModel):
    """A single instance of pseudoscience content detected."""

    category: PseudoscienceCategory = Field(
        description="Category of pseudoscience detected"
    )
    severity: PseudoscienceSeverity = Field(
        description="How the outlet treats this pseudoscience"
    )
    evidence: str = Field(
        description="Quote or description of the pseudoscience content"
    )
    scientific_consensus: str = Field(
        description="Brief statement of actual scientific consensus on this topic"
    )


class PseudoscienceLLMOutput(BaseModel):
    """Structured LLM output for pseudoscience detection."""

    indicators: list[PseudoscienceIndicator] = Field(
        default_factory=list,
        description="Pseudoscience indicators found in content"
    )
    promotes_pseudoscience: bool = Field(
        description="Whether the outlet actively promotes pseudoscience"
    )
    overall_severity: PseudoscienceSeverity = Field(
        description="Overall severity of pseudoscience content"
    )
    science_reporting_quality: float = Field(
        ge=0.0,
        le=10.0,
        description="Quality of science reporting (0=excellent, 10=promotes pseudoscience)"
    )
    respects_scientific_consensus: bool = Field(
        description="Whether outlet generally respects scientific consensus"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the assessment"
    )
    reasoning: str = Field(
        description="Explanation of pseudoscience assessment"
    )


class PseudoscienceAnalysisResult(BaseModel):
    """Complete pseudoscience analysis result."""

    domain: str = Field(
        description="The domain that was analyzed"
    )
    outlet_name: Optional[str] = Field(
        default=None,
        description="Human-readable outlet name"
    )
    score: float = Field(
        ge=0.0,
        le=10.0,
        description="MBFC-style score (0=pro-science, 10=promotes pseudoscience)"
    )
    promotes_pseudoscience: bool = Field(
        description="Whether outlet promotes pseudoscience"
    )
    overall_severity: PseudoscienceSeverity = Field(
        description="Overall severity classification"
    )
    categories_found: list[PseudoscienceCategory] = Field(
        default_factory=list,
        description="Categories of pseudoscience found"
    )
    indicators: list[PseudoscienceIndicator] = Field(
        default_factory=list,
        description="Detailed pseudoscience indicators"
    )
    respects_scientific_consensus: bool = Field(
        default=True,
        description="Whether outlet respects scientific consensus"
    )
    articles_analyzed: int = Field(
        ge=0,
        description="Number of articles analyzed"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the assessment"
    )
    reasoning: str = Field(
        description="Explanation of pseudoscience assessment"
    )


# =============================================================================
# Research Module Schemas
# =============================================================================


class HistoryLLMOutput(BaseModel):
    """Structured LLM output for outlet history extraction."""

    official_name: Optional[str] = Field(
        default=None,
        description="The proper, official name of the organization (e.g., 'The Associated Press' instead of 'apnews', 'Wall Street Journal' instead of 'wsj')"
    )
    founding_year: Optional[int] = Field(
        default=None,
        description="Year the outlet was founded"
    )
    founder: Optional[str] = Field(
        default=None,
        description="Name of founder(s)"
    )
    original_name: Optional[str] = Field(
        default=None,
        description="Original name if different from current"
    )
    key_events: list[str] = Field(
        default_factory=list,
        description="Key events in the outlet's history"
    )
    summary: str = Field(
        description="2-3 sentence history summary"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in extracted information"
    )


class OwnershipLLMOutput(BaseModel):
    """Structured LLM output for ownership/funding extraction."""

    owner: Optional[str] = Field(
        default=None,
        description="Current owner name"
    )
    parent_company: Optional[str] = Field(
        default=None,
        description="Parent company if applicable"
    )
    funding_model: Optional[str] = Field(
        default=None,
        description="Funding model: advertising, subscription, public, nonprofit, mixed"
    )
    headquarters: Optional[str] = Field(
        default=None,
        description="Headquarters location (city, country)"
    )
    country: Optional[str] = Field(
        default=None,
        description="Country where the outlet is based or headquartered"
    )
    notes: str = Field(
        default="",
        description="Additional notes about ownership/funding"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in extracted information"
    )


class ExternalAnalysisItem(BaseModel):
    """A single external analysis/criticism of an outlet."""

    source_name: str = Field(
        description="Name of the source (e.g., 'Columbia Journalism Review')"
    )
    source_url: Optional[str] = Field(
        default=None,
        description="URL of the source if available"
    )
    summary: str = Field(
        description="Brief summary of the analysis/criticism"
    )
    sentiment: str = Field(
        description="Sentiment: positive, negative, neutral, or mixed"
    )


class ExternalAnalysisLLMOutput(BaseModel):
    """Structured LLM output for external analysis extraction."""

    analyses: list[ExternalAnalysisItem] = Field(
        default_factory=list,
        description="List of external analyses found"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in extracted information"
    )


# =============================================================================
# Transparency Analysis Schemas (MBFC Factuality: 25% weight)
# =============================================================================

class TransparencyLLMOutput(BaseModel):
    """Structured LLM output for transparency analysis.

    Evaluates 5 elements per MBFC methodology:
    about page, ownership, funding, authors, location.
    """
    has_about_page: bool = Field(
        description="Whether the outlet has a clear About page"
    )
    discloses_ownership: bool = Field(
        description="Whether the outlet discloses ownership information"
    )
    discloses_funding: bool = Field(
        description="Whether the outlet discloses funding sources or model"
    )
    identifies_authors: bool = Field(
        description="Whether articles identify authors by name"
    )
    discloses_location: bool = Field(
        description="Whether the outlet discloses its physical location/headquarters"
    )
    transparency_score: float = Field(
        ge=0.0, le=10.0,
        description="Overall transparency score: 0=fully transparent, 10=no transparency"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in assessment"
    )
    reasoning: str = Field(
        description="Explanation of transparency assessment"
    )


class TransparencyResult(BaseModel):
    """Complete transparency analysis result."""
    domain: str = Field(description="Domain analyzed")
    outlet_name: Optional[str] = Field(default=None, description="Outlet name")
    has_about_page: bool = Field(default=False)
    discloses_ownership: bool = Field(default=False)
    discloses_funding: bool = Field(default=False)
    identifies_authors: bool = Field(default=False)
    discloses_location: bool = Field(default=False)
    score: float = Field(
        ge=0.0, le=10.0,
        description="Transparency score: 0=fully transparent, 10=no transparency"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = Field(default="")


# =============================================================================
# One-Sidedness / Propaganda Analysis Schemas (MBFC Factuality: 10% weight)
# =============================================================================

class PropagandaTechnique(str, Enum):
    """SemEval-style propaganda technique labels used for evidence display."""

    APPEAL_TO_AUTHORITY = "Appeal_to_Authority"
    APPEAL_TO_FEAR_PREJUDICE = "Appeal_to_fear-prejudice"
    BANDWAGON_REDUCTIO_AD_HITLERUM = "Bandwagon,Reductio_ad_Hitlerum"
    BLACK_AND_WHITE_FALLACY = "Black-and-White_Fallacy"
    CAUSAL_OVERSIMPLIFICATION = "Causal_Oversimplification"
    DOUBT = "Doubt"
    EXAGGERATION_MINIMISATION = "Exaggeration,Minimisation"
    FLAG_WAVING = "Flag-Waving"
    LOADED_LANGUAGE = "Loaded_Language"
    NAME_CALLING_LABELING = "Name_Calling,Labeling"
    REPETITION = "Repetition"
    SLOGANS = "Slogans"
    THOUGHT_TERMINATING_CLICHES = "Thought-terminating_Cliches"
    WHATABOUTISM_STRAW_MEN_RED_HERRING = "Whataboutism,Straw_Men,Red_Herring"


class PropagandaTechniqueFinding(BaseModel):
    """Article-level evidence for a detected propaganda technique."""

    technique: PropagandaTechnique = Field(
        description="One of the 14 supported propaganda technique labels"
    )
    text_snippet: str = Field(
        description="Exact quote/span from the article that demonstrates the technique"
    )
    context: str = Field(
        default="",
        description="Short surrounding context for the quoted span"
    )
    article_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-based article number from the analyzed article list, if known"
    )
    article_title: str = Field(
        default="",
        description="Article title associated with the finding, if known"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence that this quote matches the technique"
    )
    explanation: str = Field(
        default="",
        description="Brief explanation of why the span matches the technique"
    )


class OneSidednessLLMOutput(BaseModel):
    """Structured LLM output for one-sidedness/propaganda analysis.

    Evaluates balance, emotional language, and propaganda per MBFC methodology.
    """
    one_sidedness_score: float = Field(
        ge=0.0, le=10.0,
        description="0=perfect balance, 10=extreme bias/propaganda"
    )
    uses_emotional_language: bool = Field(
        description="Whether the outlet uses emotional or loaded language to persuade"
    )
    propaganda_level: str = Field(
        description="Level of propaganda: none, mild, moderate, heavy"
    )
    presents_opposing_views: bool = Field(
        description="Whether the outlet presents opposing viewpoints"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in assessment"
    )
    reasoning: str = Field(
        description="Explanation of one-sidedness assessment"
    )
    propaganda_findings: list[PropagandaTechniqueFinding] = Field(
        default_factory=list,
        description="Article-level propaganda technique findings with exact quoted evidence"
    )


class OneSidednessResult(BaseModel):
    """Complete one-sidedness/propaganda analysis result."""
    score: float = Field(
        ge=0.0, le=10.0,
        description="One-sidedness score: 0=balanced, 10=extreme propaganda"
    )
    uses_emotional_language: bool = Field(default=False)
    propaganda_level: str = Field(default="none")
    presents_opposing_views: bool = Field(default=True)
    articles_analyzed: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = Field(default="")
    propaganda_findings: list[PropagandaTechniqueFinding] = Field(default_factory=list)


# =============================================================================
# Comprehensive Report Data
# =============================================================================

class ComprehensiveReportData(BaseModel):
    """Complete data for generating an MBFC-style report."""

    # Target info
    target_url: str = Field(description="URL of the outlet")
    target_domain: str = Field(description="Domain name")
    outlet_name: str = Field(description="Human-readable outlet name")

    # Overall ratings
    bias_label: str = Field(description="MBFC-style bias label")
    bias_score: float = Field(
        ge=-10.0, le=10.0,
        description="Bias score (-10=far left, +10=far right)"
    )
    factuality_label: str = Field(description="Factuality rating label")
    factuality_score: float = Field(
        ge=0.0, le=10.0,
        description="Factuality score (0=excellent, 10=very poor)"
    )
    credibility_label: str = Field(description="Overall credibility label")
    credibility_score: float = Field(
        ge=0.0, le=15.0,
        description="Credibility point total (MBFC point-based system)"
    )

    # Source category (MBFC classification)
    source_category: str = Field(
        default="News",
        description="MBFC source category: News, Questionable, Conspiracy/Pseudoscience, Pro-Science, Satire"
    )

    # MBFC bias subscores (for transparency in reporting)
    economic_score: float = Field(default=0.0, ge=-10.0, le=10.0, description="Economic System subscore")
    social_score: float = Field(default=0.0, ge=-10.0, le=10.0, description="Social Progressive vs Conservative subscore")
    news_reporting_score: float = Field(default=0.0, ge=-10.0, le=10.0, description="Straight News Reporting Balance subscore")
    editorial_score: float = Field(default=0.0, ge=-10.0, le=10.0, description="Editorial/Op-Ed Bias subscore")

    # Traffic and metadata
    media_type: str = Field(description="Type of media outlet")
    traffic_tier: str = Field(description="Traffic tier (HIGH/MEDIUM/LOW/MINIMAL)")
    domain_age_years: Optional[float] = Field(
        default=None,
        description="Age of domain in years"
    )

    # Component analysis results
    editorial_bias_result: Optional["EditorialBiasResult"] = Field(
        default=None,
        description="Editorial bias analysis result"
    )
    fact_check_result: Optional["FactCheckAnalysisResult"] = Field(
        default=None,
        description="Fact check search result"
    )
    sourcing_result: Optional["SourcingAnalysisResult"] = Field(
        default=None,
        description="Sourcing quality analysis result"
    )
    pseudoscience_result: Optional["PseudoscienceAnalysisResult"] = Field(
        default=None,
        description="Pseudoscience analysis result"
    )
    transparency_result: Optional["TransparencyResult"] = Field(
        default=None,
        description="Transparency analysis result (MBFC: 25% of factuality)"
    )
    one_sidedness_result: Optional["OneSidednessResult"] = Field(
        default=None,
        description="One-sidedness/propaganda analysis result (MBFC: 10% of factuality)"
    )

    # Research results
    history_summary: Optional[str] = Field(
        default=None,
        description="Brief history of the outlet"
    )
    founding_year: Optional[int] = Field(
        default=None,
        description="Year founded"
    )
    founder: Optional[str] = Field(
        default=None,
        description="Founder(s) of the outlet"
    )
    original_name: Optional[str] = Field(
        default=None,
        description="Original name if different from current"
    )
    key_events: list[str] = Field(
        default_factory=list,
        description="Key events in the outlet's history"
    )
    owner: Optional[str] = Field(
        default=None,
        description="Owner/parent company"
    )
    parent_company: Optional[str] = Field(
        default=None,
        description="Parent company if applicable"
    )
    funding_model: Optional[str] = Field(
        default=None,
        description="Funding model"
    )
    headquarters: Optional[str] = Field(
        default=None,
        description="Headquarters location"
    )
    ownership_notes: Optional[str] = Field(
        default=None,
        description="Additional notes about ownership/funding"
    )

    # External analyses
    external_analyses: list[ExternalAnalysisItem] = Field(
        default_factory=list,
        description="External analyses from media watchdogs"
    )

    # Country Freedom Rating (RSF Press Freedom Index)
    country: Optional[str] = Field(
        default=None,
        description="Country where the outlet is headquartered"
    )
    freedom_score: Optional[float] = Field(
        default=None,
        description="RSF Press Freedom score (0-100, higher=freer)"
    )
    freedom_label: Optional[str] = Field(
        default=None,
        description="Freedom label (Excellent Freedom / Mostly Free / etc.)"
    )
    freedom_rank: Optional[int] = Field(
        default=None,
        description="Global freedom rank"
    )

    # Articles index (for linking source articles in bias detail)
    articles_index: list[dict] = Field(
        default_factory=list,
        description="Index of analyzed articles [{number, title, url}]"
    )

    # Evidence sufficiency (MBFC methodology requires minimum observations)
    headline_count: int = Field(
        default=0, ge=0,
        description="Number of headlines observed (MBFC requires >= 10)"
    )
    full_story_count: int = Field(
        default=0, ge=0,
        description="Number of full-text stories reviewed (MBFC requires >= 5)"
    )
    evidence_sufficient: bool = Field(
        default=True,
        description="Whether minimum evidence requirements were met for scoring"
    )
    insufficient_evidence_reason: Optional[str] = Field(
        default=None,
        description="Reason evidence was insufficient, if applicable"
    )

    # Intermediate calculation values (for recalculation/audit)
    formula_factuality_score: Optional[float] = Field(
        default=None,
        description="Factuality score from weighted formula before research signal adjustment and LLM calibration"
    )
    research_factuality_signal: Optional[float] = Field(
        default=None,
        description="Research-based factuality signal used for 60/40 blending (None if not applied)"
    )
    formula_bias_score: Optional[float] = Field(
        default=None,
        description="Bias score from editorial analysis before LLM calibration"
    )

    # Metadata
    analysis_date: str = Field(description="Date of analysis")
    articles_analyzed: int = Field(
        ge=0,
        description="Number of articles analyzed"
    )
class CalibratedScores(BaseModel):
    """LLM-calibrated bias and factuality scores.

    Used as structured output for the calibration step that adjusts
    formula-based scores using holistic evidence assessment.
    """
    bias_score: float = Field(description="Calibrated bias score from -10.0 to +10.0")
    factuality_score: float = Field(description="Calibrated factuality score from 0.0 to 10.0 (lower is better)")
    bias_reasoning: str = Field(description="Brief explanation of bias calibration decision")
    factuality_reasoning: str = Field(description="Brief explanation of factuality calibration decision")


class EvidenceSource(BaseModel):
    """A single evidence source with URL and context."""
    url: str = Field(description="URL of the source article or document")
    title: Optional[str] = Field(default=None, description="Title of the source")
    snippet: Optional[str] = Field(default=None, description="Relevant excerpt or summary from this source")


class MBFCTargetSchema(BaseModel):
    """
    The exact JSON structure required for the benchmark evaluation.
    Matches the mbfc_data_test.json format.
    """
    mbfc_url: Optional[str] = Field(default=None, description="URL to the MBFC page if known, else null")
    name: str = Field(description="Name of the news outlet")
    source_url: str = Field(description="URL of the news outlet")

    # UPDATED: Made Optional for MBC mode
    bias_rating: Optional[str] = Field(default=None, description="Label: EXTREME LEFT, etc.")
    bias_score: Optional[float] = Field(default=None, description="Score from -10 to +10")

    # UPDATED: Made Optional for MBC mode
    factual_reporting: Optional[str] = Field(default=None, description="Label: VERY HIGH, etc.")
    factual_score: Optional[float] = Field(default=None, description="Score from 0 to 10")

    credibility_rating: Optional[str] = Field(default=None, description="Label: HIGH/MEDIUM/LOW CREDIBILITY")

    country: Optional[str] = Field(default=None, description="Country of origin")
    country_freedom_rating: Optional[str] = Field(default=None, description="RSF Freedom rating if known")
    media_type: Optional[str] = Field(default=None, description="Website, TV, Newspaper, etc.")
    traffic_popularity: Optional[str] = Field(default=None, description="Traffic description")

    bias_category_description: Optional[str] = Field(default=None, description="Boilerplate description")
    overall_summary: str = Field(description="Summary paragraph starting with 'Overall, we rate...'")

    history: str = Field(description="History of the outlet")
    ownership: str = Field(description="Ownership and funding details")
    analysis: str = Field(description="Detailed analysis of bias and reporting")

    failed_fact_checks: List[str] = Field(description="List of specific failed fact checks")
    last_updated: str = Field(description="Current date")

    evidence_sources: List[EvidenceSource] = Field(
        default_factory=list,
        description="List of source URLs and articles used as evidence"
    )