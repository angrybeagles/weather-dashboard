# Workflow S: Content Strategy

- **Purpose:** Transform raw keyword research into strategic content opportunities
- **Input:** JSON response from SEO Boost API
- **Output:** Single comprehensive Markdown artifact with full analysis
- **Capacity:** 200-300 keywords per execution

---

## Execution Mode: Autonomous + Single Artifact Output

**Process Flow:**

1. Receive JSON response from SEO Boost API (paste or reference)
2. Filter irrelevant keywords in pre-processing (SILENTLY - never mention in output)
3. Process all stages in memory (no external API calls)
4. Output complete analysis as single Markdown artifact

**No external dependencies** - all processing happens in Claude's context.

**Execution Principles:**

- Filter irrelevant keywords SILENTLY before categorization - never mention filtering in artifact
- Apply categorization rules in exact sequence - first match wins
- Use precise boundary conditions (>= not >, <= not <)
- Never categorize with qualifiers like "technically should be X" - get it right the first time
- Include complete data tables - this artifact is a standalone resource for downstream workflows
- Validate categorization logic before generating artifact

---

## Input Requirements

### JSON API Response Format

The input is a JSON response from the SEO Boost API with this structure:

```json
{
  "primary_keyword": "string",
  "execution_time": "string",
  "total_keywords": number,
  "total_paa_questions": number,
  "total_subtopics": number,
  "total_serp_results": number,
  "keywords_by_type": {
    "KW Related": number,
    "KW Suggestion": number,
    "KW Idea": number
  },
  "keywords": [
    ["keyword", msv, difficulty, "intent", "competition", cpc, "type"],
    ...
  ],
  "paa_questions": ["question 1", "question 2", ...],
  "subtopics": ["subtopic 1", "subtopic 2", ...],
  "serp_results": [
    ["url", "domain", "title", "description", "type"],
    ...
  ]
}
```

### Keywords Array Format

Each keyword entry is an array with these values in order:

| Index | Field       | Description                                                          |
| ----- | ----------- | -------------------------------------------------------------------- |
| 0     | Keyword     | The keyword phrase                                                   |
| 1     | MSV         | Monthly search volume (numeric or null)                              |
| 2     | Difficulty  | Keyword difficulty 0-100 (numeric or null)                           |
| 3     | Intent      | Intent type (informational, commercial, transactional, navigational) |
| 4     | Competition | LOW, MEDIUM, HIGH                                                    |
| 5     | CPC         | Cost per click (numeric or null)                                     |
| 6     | Type        | Keyword type (KW Suggestion, KW Related, KW Idea)                    |

**Example keyword entry:**

```json
["claude code extension", 110, 10, "commercial", "LOW", 14.92, "KW Suggestion"]
```

The Primary Keyword is extracted from the `primary_keyword` field in the JSON response.

---

## Processing Stages (Autonomous Execution)

### STAGE 1: Data Ingestion & Validation

1. Parse JSON response from SEO Boost API
2. Extract Primary Keyword from `primary_keyword` field
3. Count total keywords from `keywords` array
4. Validate data quality:
   - Count keywords with full metrics (MSV + Difficulty)
   - Count keywords with partial metrics
   - Count keywords by Type from `keywords_by_type`

### STAGE 1.5: Relevance Filtering (SILENT - NOT MENTIONED IN OUTPUT)

**Purpose:** Remove obviously irrelevant keywords to improve data quality and reduce noise.

**CRITICAL: Filtering happens in background. Never mention filtered keywords, filtering process, or filtered count anywhere in the artifact output.**

**Filter out keywords that are clearly unrelated to the Primary Keyword:**

**Detection Heuristics:**

1. **Celebrity/Person Names:**

   - Contains full names pattern: [First Name] + [Last Name]
   - Examples: "claude monet", "claude debussy", "claude rains", "jean claude van damme"
   - Detection: Check if keyword contains common first/last name combinations

2. **Geographic Locations:**

   - Contains city, country, or place names unrelated to the product
   - Examples: "claude cormier montreal", "claude bosi esher"
   - Detection: Check for location keywords + name patterns

3. **Unrelated Products/Brands:**

   - Contains product names clearly unrelated to primary keyword domain
   - Examples: "louis code vein" (video game), "jules code" (unrelated product)
   - Detection: Check for brand/product names from different industries

4. **Off-Topic Terms:**
   - Keyword exists but topic is completely unrelated to primary keyword
   - Example: If primary keyword is "Claude Code" (software), filter "claude chappe code" (historical figure)

**Action:**

- Remove these keywords from ALL subsequent processing (categorization, clustering, content ideas)
- DO NOT create any "Filtered Keywords" list in the output
- DO NOT mention filtering in Executive Summary
- DO NOT include filtered keywords section anywhere in artifact
- Present the artifact as if you only received the relevant keywords to begin with

**Filtering Principles:**

- **Be conservative** - when in doubt, keep the keyword
- **Focus on obvious cases** - clear celebrity names, locations, unrelated products
- **Context matters** - "claude" alone could be relevant, "claude debussy" is not

### STAGE 2: Strategic Categorization

**CRITICAL:** Apply rules in exact order listed. First matching rule wins. Do not skip ahead.

Apply this decision tree to EVERY keyword in sequence:

```
RULE 1: PAA Questions
IF Type = "People Also Ask":
  -> Category: "Intent Signals"
  -> Reasoning: "PAA question - excellent for featured snippets"

RULE 2: Question Keywords
ELSE IF keyword starts with (how/what/why/when/where/who/which/is/are/can/do/does) OR contains "?":
  -> Category: "Intent Signals"
  -> Reasoning: "Question format ([first word]) - ideal for SERP features"

RULE 3: Low-Volume Semantic Keywords
ELSE IF Type = "autocomplete" OR "Subtopic" OR "KW Related" AND (MSV is null OR MSV < 50):
  -> Category: "Semantic Topics"
  -> Reasoning: "[Type] keyword - builds topical authority"

RULE 4: Quick Wins (PRIORITY CHECK)
ELSE IF MSV >= 101 AND Difficulty <= 29 AND Difficulty is not null:
  -> Category: "Quick Wins"
  -> Reasoning: "Low difficulty ([Difficulty]) with good volume ([MSV])"

  EXAMPLES:
  - MSV: 210, Difficulty: 2 -> Quick Wins
  - MSV: 110, Difficulty: 10 -> Quick Wins
  - MSV: 320, Difficulty: 5 -> Quick Wins

RULE 5: High-Difficulty Authority Targets
ELSE IF Difficulty >= 51 AND MSV >= 201 AND Difficulty is not null:
  -> Category: "Authority Builders"
  -> Reasoning: "Competitive ([Difficulty]) but valuable ([MSV])"

RULE 6: Emerging Topics (Low Volume)
ELSE IF MSV <= 100 AND MSV >= 1 AND MSV is not null:
  -> Category: "Emerging Topics"
  -> Reasoning: "Low volume ([MSV]) - early mover advantage"

  EXAMPLES:
  - MSV: 90, Difficulty: 10 -> Emerging Topics
  - MSV: 70, Difficulty: 5 -> Emerging Topics
  - MSV: 50, Difficulty: 0 -> Emerging Topics

RULE 7: Moderate Metrics (100-200 MSV, 30-50 Difficulty)
ELSE IF MSV >= 101 AND MSV <= 200 AND Difficulty >= 30 AND Difficulty <= 50:
  IF Intent = "informational":
    -> Category: "Authority Builders"
    -> Reasoning: "Informational with moderate metrics"
  ELSE:
    -> Category: "Quick Wins"
    -> Reasoning: "Commercial/transactional with moderate metrics"

RULE 8: No Metrics Available
ELSE IF MSV is null AND Difficulty is null:
  -> Category: "Semantic Topics"
  -> Reasoning: "No volume data - supporting term for topic authority"

RULE 9: High Volume Default
ELSE IF MSV >= 201:
  -> Category: "Authority Builders"
  -> Reasoning: "High volume ([MSV]) - authority target"

RULE 10: Catch-All
ELSE:
  -> Category: "Unknown"
  -> Reasoning: "Insufficient data - manual review recommended"
```

**Validation Check:**
After categorization, verify:

- Keywords with MSV 1-100 -> Should be Emerging Topics (unless Rule 1-3 caught them)
- Keywords with MSV 101+ AND Difficulty 0-29 -> Should be Quick Wins (unless Rule 1-3 caught them)
- No keyword should be "technically" wrong category

**Count by category after categorization.**

### STAGE 3: Semantic Clustering

**Purpose:** Group keywords into 4-6 thematic clusters based on semantic relationships.

**Clustering Scope:**
Only cluster keywords suitable for content creation:

- **Include:** Quick Wins, Authority Builders, Intent Signals, Emerging Topics (if relevant)
- **Exclude:** Semantic Topics (autocomplete, subtopics), Unknown category
- **Target:** 60-80% of total keywords clustered (the rest are supporting/tangential terms)

**Clustering Approach:**

1. Identify broad themes in the keyword set
2. Group keywords by shared user intent and topic
3. Create 4-6 clusters (minimum 5 keywords per cluster)
4. Each keyword belongs to exactly ONE cluster

**For each cluster, define:**

- **Cluster Name:** 2-4 word descriptive name
- **Core Topic:** One sentence describing the theme
- **Intent Pattern:** Primary user intent
- **Keywords:** All keywords in this cluster
- **Reasoning:** Why these keywords group together

**Common cluster patterns for SEO keyword research:**

- Pricing & Cost Investigation
- Product/Tool Comparisons
- Feature & Capability Exploration
- Setup & Implementation Guides
- General Platform Information
- Best Practices & Optimization

### STAGE 4: Content Ideas Generation

**Content Idea Generation Scope:**

Generate content ideas ONLY for:

- Quick Wins
- Authority Builders
- Intent Signals
- Emerging Topics (if relevant to primary keyword)

**Do NOT generate content ideas for:**

- Semantic Topics (supporting terms only - used for topical authority, not primary content)
- Unknown category

**Final Count Calculation:**
Total content ideas = (category-based ideas for relevant keywords) + (hub articles) + (spoke articles)

#### Part A: Category-Based Content Ideas

For each categorized keyword (in scope per above), generate:

**Title (under 60 characters):**

- Use title templates based on keyword pattern:
  - **"vs" keywords:** "[A] vs [B]: Complete Comparison 2025"
  - **"what is" keywords:** "What Is [Topic]? Complete Guide 2025"
  - **"how to" keywords:** "How to [Action]: Step-by-Step Guide"
  - **Question keywords:** "[Question]: Everything You Need to Know"
  - **Pricing keywords:** "[Product] Pricing Guide 2025: Plans & Costs"
  - **Best practice keywords:** "[Topic] Best Practices: Expert Tips 2025"
  - **Generic keywords:** "[Keyword]: Complete Guide 2025"

**Description (50-120 words):**

- 2-3 sentences outlining main topics
- Indicates specific value for readers
- Incorporates keyword naturally

#### Part B: Hub/Spoke Strategy

For EACH cluster, create:

**1 Hub Article (pillar content):**

- Title: Comprehensive guide covering entire cluster theme
- Description: 3-4 sentences covering full scope
- Type: "hub"
- Keyword: Most relevant keyword from cluster

**3 Spoke Articles (supporting content):**

- Title: Specific subtopic within cluster
- Description: 2-3 sentences on focused aspect
- Type: "spoke"
- Keyword: Specific keyword from cluster

**Select spokes to cover:**

1. Different approach/angle
2. Specific use case
3. Detailed how-to
4. Comparison/options (if applicable)
5. Advanced/specific aspect (if 5 spokes instead of 3)

**Total articles per cluster:** 1 hub + 3 spokes = 4 articles
**Total articles for 6 clusters:** 24 articles

---

## Output Format: Single Markdown Artifact

Generate ONE comprehensive Markdown document with the following structure:

# Strategic Keyword Organization Results

**Primary Keyword:** [keyword]
**Date Analyzed:** [date]
**Total Keywords Analyzed:** [count after filtering - present as if this is what we started with]
**Execution Time:** [time]

---

## Executive Summary

- Keywords Analyzed: [count after filtering]
- Categories Identified: [count with breakdown]
- Clusters Created: [count]
- Keywords Clustered: [count] ([%]% of analyzed keywords)
- Content Ideas Generated: [total count]
  - Category-based ideas: [count]
  - Hub articles: [count]
  - Spoke articles: [count]
- Recommended Priority: [Quick Wins -> Intent Signals -> Authority Builders]

**Key Insights:**

- [Notable patterns in data]
- [Top opportunities identified]
- [Strategic recommendations]

---

## Stage 2: Categorization Results

### Overview

| Category           | Count | Percentage | Priority    |
| ------------------ | ----- | ---------- | ----------- |
| Quick Wins         | [X]   | [%]        | HIGH        |
| Authority Builders | [X]   | [%]        | MEDIUM-HIGH |
| Intent Signals     | [X]   | [%]        | HIGH        |
| Emerging Topics    | [X]   | [%]        | MEDIUM      |
| Semantic Topics    | [X]   | [%]        | LOW         |
| Unknown            | [X]   | [%]        | REVIEW      |

### Top Opportunities by Category

#### Quick Wins (Low Difficulty, Good Volume)

1. [keyword] - MSV: [X], Difficulty: [X]
2. [keyword] - MSV: [X], Difficulty: [X]
   [list top 10]

#### Intent Signals (SERP Features)

1. [question keyword]
2. [question keyword]
   [list all]

#### Authority Builders (High Value Targets)

1. [keyword] - MSV: [X], Difficulty: [X]
2. [keyword] - MSV: [X], Difficulty: [X]
   [list top 5]

### Complete Categorization Table

| Keyword   | Category   | MSV   | Difficulty | Intent   | Reasoning   |
| --------- | ---------- | ----- | ---------- | -------- | ----------- |
| [keyword] | [category] | [msv] | [diff]     | [intent] | [reasoning] |

[full table with ALL keywords - only show keywords that passed filtering]

---

## Stage 3: Semantic Clusters

### Cluster Overview

| #   | Cluster Name | Keywords | Intent Pattern |
| --- | ------------ | -------- | -------------- |
| 1   | [name]       | [count]  | [pattern]      |
| 2   | [name]       | [count]  | [pattern]      |

[all clusters]

**Note:** Clustered [X] keywords ([%]% of analyzed keywords). Remaining keywords are Semantic Topics, Unknown, or supporting terms not suitable for primary content creation.

### Detailed Cluster Breakdown

#### Cluster 1: [Cluster Name]

**Core Topic:** [one sentence]
**Intent Pattern:** [user intent description]
**Total Keywords:** [count]
**Reasoning:** [why these keywords group together]

**Keywords in this cluster:**

- [keyword 1] ([MSV] MSV, Difficulty: [X])
- [keyword 2] ([MSV] MSV, Difficulty: [X])
- [keyword 3] ([MSV] MSV, Difficulty: [X])
  [all keywords in cluster with metrics]

[Repeat for each cluster]

---

## Stage 4A: Category-Based Content Ideas

### Quick Wins Content Ideas

| Title   | Keyword   | MSV   | Difficulty | Description   |
| ------- | --------- | ----- | ---------- | ------------- |
| [title] | [keyword] | [msv] | [diff]     | [description] |

[all Quick Win ideas]

### Intent Signals Content Ideas

| Title   | Keyword   | MSV   | Description   |
| ------- | --------- | ----- | ------------- |
| [title] | [keyword] | [msv] | [description] |

[all Intent Signal ideas]

### Authority Builders Content Ideas

| Title   | Keyword   | MSV   | Difficulty | Description   |
| ------- | --------- | ----- | ---------- | ------------- |
| [title] | [keyword] | [msv] | [diff]     | [description] |

[all Authority Builder ideas]

### Emerging Topics Content Ideas

| Title   | Keyword   | MSV   | Description   |
| ------- | --------- | ----- | ------------- |
| [title] | [keyword] | [msv] | [description] |

[all Emerging Topic ideas for relevant keywords]

---

## Stage 4B: Hub/Spoke Content Strategy

### Cluster 1: [Cluster Name]

#### HUB ARTICLE (Pillar Content)

**Title:** [comprehensive title]
**Keyword:** [primary keyword]
**Description:** [3-4 sentence comprehensive description]
**Type:** Hub

#### SPOKE ARTICLES (Supporting Content)

**Spoke 1:**
**Title:** [specific subtopic title]
**Keyword:** [keyword]
**Description:** [2-3 sentence focused description]
**Type:** Spoke

**Spoke 2:**
**Title:** [specific subtopic title]
**Keyword:** [keyword]
**Description:** [2-3 sentence focused description]
**Type:** Spoke

**Spoke 3:**
**Title:** [specific subtopic title]
**Keyword:** [keyword]
**Description:** [2-3 sentence focused description]
**Type:** Spoke

[Repeat for each cluster]

### Hub/Spoke Summary Table

| Cluster | Hub Title | Spoke 1 | Spoke 2 | Spoke 3 |
| ------- | --------- | ------- | ------- | ------- |
| [name]  | [title]   | [title] | [title] | [title] |

[all clusters]

---

## Master Keyword Assignment Table

**CRITICAL:** Include ALL analyzed keywords (after silent filtering) in this table. This is a complete reference for downstream workflows.

Complete table showing all assignments:

| Keyword   | Category   | Category Reasoning | Cluster Name | MSV   | Difficulty | Intent   | Content Status |
| --------- | ---------- | ------------------ | ------------ | ----- | ---------- | -------- | -------------- |
| [keyword] | [category] | [reasoning]        | [cluster]    | [msv] | [diff]     | [intent] | Categorized    |

[ALL keywords with full assignments - do not truncate]

---

## Execution Statistics

### Processing Summary

- Keywords Analyzed: [X]
- Successfully Categorized: [X] ([%]%)
- Clustered: [X] ([%]% of analyzed)
- Unknown/Review Needed: [X] ([%]%)

### Content Opportunities Created

- Category-Based Ideas: [X]
- Hub Articles: [X]
- Spoke Articles: [X]
- **Total Content Ideas:** [X]

### Category Distribution

- Quick Wins: [X] keywords ([%]%)
- Authority Builders: [X] keywords ([%]%)
- Emerging Topics: [X] keywords ([%]%)
- Intent Signals: [X] keywords ([%]%)
- Semantic Topics: [X] keywords ([%]%)
- Unknown: [X] keywords ([%]%)

### Cluster Distribution

- [Cluster Name]: [X] keywords ([%]%)
  [all clusters]

---

## Recommended Action Plan

### Week 1-2: Quick Wins Priority

Target these high-ROI keywords first:

1. [keyword] - [reasoning]
2. [keyword] - [reasoning]
   [top 10 Quick Wins]

### Week 3-4: Intent Signals for SERP Features

Create content for these question keywords:

1. [keyword]
2. [keyword]
   [all Intent Signals]

### Month 2: Authority Building Campaign

Begin long-term authority content:

1. [keyword] - [MSV] volume, [difficulty] difficulty
2. [keyword] - [MSV] volume, [difficulty] difficulty
   [top 5 Authority Builders]

### Ongoing: Hub/Spoke Architecture

Implement structured content strategy:

- Start with [cluster name] hub
- Complete 3 spokes for depth
- Link internally for topical authority
- Repeat for remaining clusters

---

## Notes & Considerations

### Data Quality Observations

- [Any patterns noticed in the data]
- [Keywords that may need further review]
- [Suggestions for improvement]

### Strategic Insights

- [Key opportunities identified]
- [Market gaps or niches]
- [Competition analysis observations]

---

## Workflow C Ready Content Ideas

**Instructions:** Copy any content idea block below and paste into a new Claude chat with: "Execute Workflow C on this content idea:"

### Quick Wins Ideas (Ready for Workflow C)

```
Title: [title from table]
Keyword: [keyword from table]
Description: [description from table]
Primary Keyword: [primary keyword from report header]
Category: Quick Wins
```

[Generate one block per Quick Win content idea]

### Authority Builders Ideas (Ready for Workflow C)

```
Title: [title from table]
Keyword: [keyword from table]
Description: [description from table]
Primary Keyword: [primary keyword from report header]
Category: Authority Builders
```

[Generate one block per Authority Builder content idea]

### Intent Signals Ideas (Ready for Workflow C)

```
Title: [title from table]
Keyword: [keyword from table]
Description: [description from table]
Primary Keyword: [primary keyword from report header]
Category: Intent Signals
```

[Generate one block per Intent Signal content idea]

### Hub Articles (Ready for Workflow C)

```
Title: [hub title]
Keyword: [hub keyword]
Description: [hub description]
Primary Keyword: [primary keyword from report header]
Category: Authority Builders
Type: Hub
Cluster: [cluster name]
```

[Generate one block per hub article]

### Spoke Articles (Ready for Workflow C)

```
Title: [spoke title]
Keyword: [spoke keyword]
Description: [spoke description]
Primary Keyword: [primary keyword from report header]
Category: Quick Wins
Type: Spoke
Cluster: [cluster name]
Hub Article: [related hub title]
```

[Generate one block per spoke article]

---

_End of Content Strategy Report_

---

## Execution Checklist

Before generating the artifact, ensure:

- [ ] JSON API response successfully parsed
- [ ] Primary keyword identified from `primary_keyword` field
- [ ] Irrelevant keywords filtered SILENTLY (no mention in output)
- [ ] All remaining keywords categorized using exact rule order (no "technically should be X" notes)
- [ ] Validation: MSV 1-100 keywords are Emerging Topics (unless Intent Signals/Semantic)
- [ ] Validation: MSV 101+ with Difficulty 0-29 keywords are Quick Wins (unless Intent Signals/Semantic)
- [ ] Target: <5% Unknown category
- [ ] 4-6 semantic clusters created
- [ ] Only relevant categories included in content ideas (Quick Wins, Authority Builders, Intent Signals, relevant Emerging Topics)
- [ ] Hub/spoke articles created for each cluster
- [ ] **Complete Master Keyword Assignment Table included with ALL analyzed keywords**
- [ ] Statistics calculated accurately (showing only analyzed keywords)
- [ ] NO mention of filtering anywhere in artifact
- [ ] Recommendations prioritized strategically

---

## Usage

1. **Call SEO Boost API** to get keyword research data
2. **Paste the JSON response into Claude**
3. **Say:** "Execute Workflow S on this keyword data"
4. **Receive:** Complete analysis as single comprehensive Markdown artifact
5. **Use content idea blocks for Workflow C content generation**

**Expected execution time:** 2-5 minutes for 100-200 keywords
