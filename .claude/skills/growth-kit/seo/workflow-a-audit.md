# Workflow A: Audit & Opportunity Analysis

- **Purpose:** Evaluate existing content, competitive positioning, and content gaps BEFORE keyword research
- **Platform:** Claude Desktop (or Claude Code) with web_search
- **Input:** Website URL + Google Search Console data (optional) + business context
- **Output:** Single comprehensive Markdown artifact with audit findings and prioritized opportunities
- **Capacity:** 1 domain per execution

---

## Why Workflow A Exists

Most SEO workflows start with keyword research. This is backwards.

**The Problem:**

- Keyword research without context produces generic recommendations
- You may already have content assets that need optimization, not replacement
- Competitor gaps are invisible without systematic analysis
- Without positioning evaluation, you're "ranking in a vacuum"

**The Solution:**
Workflow A audits what exists BEFORE discovering new keywords. It answers:

1. What content do we already have, and what should we do with it?
2. Where do we stand vs. competitors in search results?
3. What gaps exist—both vs. competitors AND in the market overall?
4. What should we prioritize first?

**Pipeline Position:**

```
Workflow A (Audit) → SEO Boost API (Keyword Discovery) → Workflow S (Content Strategy) → Workflow C (Content Generation)
                  ↘ Workflow R (Rank Enhancement) ← existing content optimization loop
```

---

## The Three Pillars Framework

Workflow A is structured around three audit dimensions:

| Pillar                     | Focus                 | Key Question                            |
| -------------------------- | --------------------- | --------------------------------------- |
| **Positioning Evaluation** | Competitive landscape | Where do we stand vs. competitors?      |
| **Content Gap Analysis**   | Missing opportunities | What topics are we missing?             |
| **Content Inventory**      | Existing assets       | What should we do with current content? |

**Critical Insight (Lidia Infante, Wix):**

> "There is a ceiling to how much optimizing one aspect of your SEO can improve your rankings. Be it content, technical SEO, or links, the returns on your SEO activities diminish as you become better and better at one particular area."

This means: if your content is excellent but not ranking, more content improvement yields diminishing returns. Focus shifts to technical SEO or link building instead.

---

## Execution Mode: Autonomous + Single Artifact Output

**Process Flow:**

1. Receive website URL and business context
2. Gather GSC data if available (or use web_search for competitive intelligence)
3. Execute all audit stages sequentially
4. Output complete audit as single Markdown artifact

**Dependencies:**

- web_search tool for SERP analysis and competitor research
- Optional: Google Search Console data export (CSV or manual input)
- Optional: SEO tool exports (Ahrefs, Semrush) for deeper analysis

**Execution Principles:**

- Audit existing state before recommending new content
- Apply Three Pillars framework systematically
- Use decision trees for consistent recommendations
- Produce actionable output that feeds into SEO Boost API or Workflow R
- Prioritize by Impact × Effort

---

## Input Requirements

### Required Information

```
Website URL: [full domain, e.g., https://example.com]
Business Description: [1-2 sentences on what the business does]
Primary Topic/Niche: [main subject area for SEO]
Target Audience: [who you're trying to reach]
Business Goals: [what success looks like - traffic, leads, sales, authority]
```

### Optional (Enhances Analysis)

```
Google Search Console Export: [CSV with queries, pages, clicks, impressions, position]
Top 3-5 Competitors: [domains of main SEO competitors]
Existing Content List: [URLs of current blog/content pages]
Current Rankings: [any known keyword positions]
Previous SEO Work: [what's been tried before]
```

### Minimal Viable Input

At minimum, provide:

- Website URL
- Primary topic/niche
- Target audience

All other information can be inferred through web_search analysis.

---

## Processing Stages (Autonomous Execution)

### STAGE 1: Context Gathering & Competitor Identification

**Step 1.1: Parse Input**

1. Extract website URL and strip to root domain
2. Note business description and goals
3. Identify primary topic/niche for SERP analysis
4. Record target audience for intent matching

**Step 1.2: Identify SEO Competitors**

Execute web_search for 3-5 core topic queries:

```
web_search("[primary topic] guide")
web_search("[primary topic] best practices")
web_search("[primary topic] for [target audience]")
```

**From search results, identify:**

- Which domains consistently appear in top 10
- Note: SEO competitors may differ from business competitors
- Select 3-5 domains that compete for same keywords

**Output: Competitor List**
| Competitor | Domain | Appears in X/5 Searches | Notes |
|------------|--------|------------------------|-------|

**Step 1.3: Baseline Assessment**

If GSC data available:

- Total organic clicks (last 3 months)
- Total impressions
- Average position
- Number of ranking queries
- Top 10 pages by traffic

If no GSC data:

- Use web_search to check if site appears for core queries
- Note approximate visibility level (strong/moderate/weak/invisible)

---

### STAGE 2: Positioning Evaluation

**Purpose:** Understand where you stand relative to competitors in search results.

**Step 2.1: SERP Landscape Analysis**

For 10-15 core keywords in your niche, execute web_search and document:

| Keyword | Your Position | Top Competitor | SERP Features | Content Type Winning |
| ------- | ------------- | -------------- | ------------- | -------------------- |

**SERP Features to Note:**

- Featured snippets (present/absent, who owns it)
- People Also Ask boxes
- AI Overviews (and which sources cited)
- Video carousels
- Local pack
- Knowledge panels
- Image packs

**Content Types Winning:**

- Ultimate guides (3000+ words)
- Listicles
- How-to tutorials
- Comparison pages
- Tools/calculators
- Video content

**Step 2.2: Competitive Scorecard**

For each competitor (3-5), assess:

| Metric                        | Competitor A | Competitor B | Competitor C | Your Site |
| ----------------------------- | ------------ | ------------ | ------------ | --------- |
| Estimated organic traffic     |              |              |              |           |
| Content volume (approx pages) |              |              |              |           |
| Domain authority (if known)   |              |              |              |           |
| Content quality (1-5)         |              |              |              |           |
| SERP feature ownership        |              |              |              |           |

**Quality Assessment Criteria:**

- 5: Comprehensive, well-structured, unique insights, strong E-E-A-T signals
- 4: Thorough coverage, good structure, some unique value
- 3: Adequate coverage, standard approach, no differentiation
- 2: Thin content, missing key subtopics, outdated
- 1: Poor quality, keyword-stuffed, unhelpful

**Step 2.3: Position Gap Identification**

Categorize your keyword positions:

| Position Range | Count | Strategic Implication                         |
| -------------- | ----- | --------------------------------------------- |
| 1-3            |       | Defend and maintain                           |
| 4-10           |       | Optimize for quick wins (Workflow R)          |
| 11-20          |       | Striking distance—priority optimization       |
| 21-50          |       | Requires significant investment               |
| Not ranking    |       | Net new content opportunities (API -> S -> C) |

**Step 2.4: Competitive Advantage Assessment**

Based on analysis, identify:

**Your Advantages:**

- Topics where you outrank competitors
- Unique angles or expertise you have
- Content formats you do better
- E-E-A-T signals (credentials, experience, authority)

**Your Disadvantages:**

- Topics where competitors dominate
- Content depth gaps
- Technical or UX weaknesses
- Authority/backlink gaps

---

### STAGE 3: Content Gap Analysis

**Purpose:** Identify missing topic opportunities—both vs. competitors and in the market overall.

**Step 3.1: Competitor Content Mapping**

For each competitor, categorize their content:

```
web_search("site:competitor.com [primary topic]")
```

**Document:**

- Main topic categories covered
- Estimated number of pieces per category
- Notable content series or hubs
- Content you have vs. don't have

**Output: Content Coverage Matrix**

| Topic Category | Competitor A | Competitor B | Competitor C | Your Site   | Gap?    |
| -------------- | ------------ | ------------ | ------------ | ----------- | ------- |
| [Topic 1]      | ✅ 12 pieces | ✅ 8 pieces  | ✅ 5 pieces  | ❌ 0        | YES     |
| [Topic 2]      | ✅ 6 pieces  | ❌ 0         | ✅ 3 pieces  | ✅ 2 pieces | Partial |

**Step 3.2: Standard Content Gap (vs. Competitors)**

Identify topics where:

- **All competitors rank, you don't** = Critical gap
- **Multiple competitors rank, you don't** = Important gap
- **One competitor ranks, you don't** = Opportunity gap

**Classification:**

| Gap Type | Definition                                             | Priority |
| -------- | ------------------------------------------------------ | -------- |
| Missing  | Competitors cover extensively, you have nothing        | High     |
| Weak     | You have content but competitors significantly outrank | Medium   |
| Partial  | You cover some subtopics but not comprehensively       | Medium   |
| Format   | Competitors use different format (video, tool, etc.)   | Varies   |

**Step 3.3: Double Content Gap (Blue Ocean Analysis)**

**Critical Addition:** Find topics that BOTH you AND competitors are missing.

**Discovery Methods:**

1. **People Also Ask Mining**

   - Search core topics and expand PAA boxes
   - Note questions without satisfying answers in results
   - These are untapped demand signals

2. **Related Searches Analysis**

   - Bottom of SERP "Related searches"
   - Check if any lead to weak/thin content

3. **Pain Point Indicators**

   - If you have access: customer support tickets, sales call notes
   - Community mining: Reddit, forums, Discord for unanswered questions
   - Comment sections on competitor content

4. **SERP Quality Assessment**
   - For potential topics, analyze top 3 results
   - If results are outdated, thin, or off-topic = opportunity

**Blue Ocean Criteria:**

- Real pain point (heard from multiple sources)
- SERP shows unsatisfying results
- You can provide genuinely better/different value
- Aligns with your expertise and product
- Serves users who could become customers

**Output: Double Content Gap Opportunities**

| Topic/Question | Evidence of Demand | Current SERP Quality | Our Advantage | Priority |
| -------------- | ------------------ | -------------------- | ------------- | -------- |

**Step 3.4: Intent Gap Analysis**

Check if you're missing entire intent categories:

| Intent Type                      | Your Coverage | Competitor Coverage | Gap? |
| -------------------------------- | ------------- | ------------------- | ---- |
| Informational (how/what/why)     |               |                     |      |
| Commercial (best/compare/review) |               |                     |      |
| Transactional (buy/pricing/get)  |               |                     |      |
| Navigational (brand + keyword)   |               |                     |      |

**Step 3.5: E-E-A-T Assessment**

Evaluate your site's Experience, Expertise, Authoritativeness, and Trust signals:

**Experience Signals:**
| Signal | Present? | Quality (1-5) | Improvement Needed |
|--------|----------|---------------|-------------------|
| First-hand experience demonstrated | | | |
| Original photos/screenshots | | | |
| Personal case studies with results | | | |
| Behind-the-scenes insights | | | |
| "What I learned" narratives | | | |

**Expertise Signals:**
| Signal | Present? | Quality (1-5) | Improvement Needed |
|--------|----------|---------------|-------------------|
| Author credentials displayed | | | |
| Professional background relevant | | | |
| Industry certifications | | | |
| Regular content in domain | | | |
| Technical depth appropriate | | | |

**Authoritativeness Signals:**
| Signal | Present? | Quality (1-5) | Improvement Needed |
|--------|----------|---------------|-------------------|
| External backlinks quality | | | |
| Brand mentions across web | | | |
| Citations in publications | | | |
| Awards or recognition | | | |
| Speaking/thought leadership | | | |

**Trust Signals:**
| Signal | Present? | Quality (1-5) | Improvement Needed |
|--------|----------|---------------|-------------------|
| Transparent sourcing/citations | | | |
| Content update timestamps | | | |
| Clear contact info/about page | | | |
| HTTPS and security | | | |
| Privacy policy/terms | | | |
| Review platform presence | | | |

**E-E-A-T Score:** Sum all quality scores. Interpret:
- 80+: Strong E-E-A-T, focus on content
- 50-79: Moderate E-E-A-T, improve signals alongside content
- <50: Weak E-E-A-T, prioritize trust-building before content scale

**Step 3.6: Site Type-Specific Issues**

Apply site type-specific audit considerations:

**SaaS/Product Sites:**
- [ ] Product pages have sufficient content depth (not just features)
- [ ] Blog/content is integrated with product pages
- [ ] Comparison/alternative pages exist
- [ ] Feature pages have substantial content
- [ ] Glossary/educational content supports buyer journey

**E-commerce:**
- [ ] Category pages have unique intro content (not just product grid)
- [ ] Product descriptions are unique (not manufacturer boilerplate)
- [ ] Product schema implemented correctly
- [ ] Faceted navigation handled (canonicals, noindex filters)
- [ ] Out-of-stock pages managed properly

**Content/Blog Sites:**
- [ ] Content freshness maintained (outdated posts updated)
- [ ] No keyword cannibalization between posts
- [ ] Topical clustering implemented
- [ ] Internal linking connects related content
- [ ] Author pages exist with E-E-A-T signals

**Local Business:**
- [ ] NAP (Name, Address, Phone) consistent across web
- [ ] LocalBusiness schema implemented
- [ ] Google Business Profile optimized
- [ ] Location-specific landing pages exist
- [ ] Local content strategy in place

---

### STAGE 4: Content Inventory Assessment

**Purpose:** Evaluate existing content and determine action for each piece.

**Step 4.1: Content Collection**

If site has existing content, list all content pages:

- URL
- Title
- Publish date (if visible)
- Topic/category
- Estimated word count

**Step 4.2: Performance Classification**

If GSC data available, classify each page:

| Performance Tier  | Criteria                             | Typical Action                   |
| ----------------- | ------------------------------------ | -------------------------------- |
| Top Performers    | Position 1-3, consistent traffic     | KEEP (monitor)                   |
| Near Winners      | Position 4-10, good impressions      | IMPROVE (Workflow R)             |
| Striking Distance | Position 11-20, moderate impressions | IMPROVE (priority)               |
| Underperformers   | Position 21-50, low traffic          | EVALUATE (improve or remove)     |
| Zero Traffic      | No impressions for 6+ months         | EVALUATE (consolidate or remove) |

**Step 4.3: Quick-Win Category Classification**

When GSC + analytics data is available, classify pages into specific quick-win categories using these filter criteria:

| Category              | Filter Criteria                                       | Typical Fix                               | Time    |
| --------------------- | ----------------------------------------------------- | ----------------------------------------- | ------- |
| **CTR Optimization**  | Impressions >1,000/mo + CTR <5% + Position 5-15       | Title/meta rewrite                        | 30 min  |
| **Striking Distance** | Position 8-15 + Impressions >500/mo + Words >1,000    | Add 300-500 words, internal links, FAQ    | 90 min  |
| **Engagement Fix**    | Position 1-5 + Bounce >70% + Time on page <1 min      | Improve intro, add TOC, better formatting | 45 min  |
| **Thin Authority**    | Position 4-10 + Words <800 + Backlinks >3             | Expand to 1,500-2,000 words               | 2 hours |
| **Orphan Rescue**     | Sessions >100/mo + Internal links 0-1 + Age >6 months | Add 5-8 contextual internal links         | 20 min  |

**Data Sources Required:**

- **Google Analytics (GA4):** Sessions, bounce rate, average time on page, conversions
- **Google Search Console:** Impressions, clicks, CTR, average position
- **SEO Tool (Ahrefs/Semrush):** Backlink count, word count, internal link count

**Classification Process:**

1. Export data from all three sources
2. Combine into single spreadsheet by URL
3. Apply filter criteria for each category
4. Pages may qualify for multiple categories - prioritize by highest impact fix
5. Document category assignment in Content Actions Summary

**Output: Quick-Win Classification**

| URL | Category | Filter Match | Priority Fix | Est. Time |
| --- | -------- | ------------ | ------------ | --------- |

**Handoff to Workflow R:**

When routing pages to Workflow R, include the category classification:

```
URL: [page URL]
Category: [CTR Optimization / Striking Distance / Engagement Fix / Thin Authority / Orphan Rescue]
Current Position: [from GSC]
Key Metrics: [the filter values that triggered classification]
Priority: [Tier from audit]
```

This enables Workflow R to apply category-specific optimization protocols rather than generic enhancement.

---

**Step 4.4: KICP Decision Tree**

Apply this decision tree to each piece of content:

```
START: Is the content still relevant to our business?
│
├─ NO → Is there a redirect target?
│        ├─ YES → REMOVE + 301 redirect
│        └─ NO → REMOVE (410 or soft 404)
│
└─ YES → Is it performing adequately?
         │
         ├─ YES → KEEP (monitor annually)
         │
         └─ NO → Is the core topic still relevant?
                  │
                  ├─ NO → Is there similar content to combine?
                  │        ├─ YES → CONSOLIDATE into stronger piece
                  │        └─ NO → REMOVE + redirect
                  │
                  └─ YES → Can it be meaningfully improved?
                           ├─ YES → IMPROVE (refresh content)
                           └─ NO → REMOVE + redirect
```

**Step 4.5: Content Actions Summary**

| Action      | Count | Description                                       |
| ----------- | ----- | ------------------------------------------------- |
| KEEP        |       | Pages performing well, monitor only               |
| IMPROVE     |       | Pages with optimization potential (→ Workflow R)  |
| CONSOLIDATE |       | Multiple weak pages to merge into one strong page |
| PRUNE       |       | Pages to remove or redirect                       |

**For IMPROVE pages, note specific opportunities:**

- Title/meta optimization needed
- Content expansion needed
- Internal linking gaps
- Outdated information
- Missing SERP feature optimization

**For CONSOLIDATE pages, identify:**

- Target page (the "winner" to merge into)
- Pages to merge
- Redirect mapping

---

### STAGE 5: Opportunity Prioritization

**Purpose:** Rank all opportunities by Impact × Effort to create actionable roadmap.

**Step 5.1: Compile All Opportunities**

Gather opportunities from all stages:

| Source                      | Opportunity Type                | Count |
| --------------------------- | ------------------------------- | ----- |
| Positioning (Stage 2)       | Striking distance optimizations |       |
| Content Gap (Stage 3)       | Competitor gaps to fill         |       |
| Content Gap (Stage 3)       | Blue ocean opportunities        |       |
| Content Inventory (Stage 4) | Content improvements            |       |
| Content Inventory (Stage 4) | Consolidations                  |       |

**Step 5.2: Score Each Opportunity**

**Impact Score (1-5):**

- 5: High traffic potential + high conversion relevance
- 4: Good traffic potential + good relevance
- 3: Moderate traffic + moderate relevance
- 2: Low traffic but strategic value
- 1: Minimal expected impact

**Effort Score (1-5):**

- 5: Minimal effort (quick optimization, <1 day)
- 4: Low effort (minor content update, 1-2 days)
- 3: Moderate effort (significant update, 3-5 days)
- 2: High effort (new content piece, 1-2 weeks)
- 1: Major effort (pillar content, multiple weeks)

**Priority Score = Impact × Effort**

| Priority | Score Range | Action                    |
| -------- | ----------- | ------------------------- |
| Tier 1   | 15-25       | Execute immediately       |
| Tier 2   | 10-14       | Schedule for next sprint  |
| Tier 3   | 5-9         | Plan for future           |
| Tier 4   | 1-4         | Deprioritize or eliminate |

**Step 5.3: Create Prioritized Roadmap**

**Tier 1: Quick Wins (This Week)**

- Striking distance optimizations (→ Workflow R)
- Meta title/description updates
- Internal linking fixes
- Content consolidations

**Tier 2: Short-Term (This Month)**

- Content improvements requiring moderate effort
- Blue ocean content opportunities
- Key competitor gap content

**Tier 3: Medium-Term (This Quarter)**

- New content pillars
- Topic cluster buildout
- Authority building content

**Tier 4: Long-Term (Next Quarter+)**

- Major site restructuring
- Comprehensive link building
- New content formats (tools, videos)

---

### STAGE 6: Output Generation

**Purpose:** Compile all findings into actionable audit report.

Generate the complete audit artifact following the Output Format Template below.

---

## Output Format: Markdown Artifact Template

# SEO Audit Report: [Website Domain]

**Audit Date:** [current date]
**Primary Topic/Niche:** [from input]
**Target Audience:** [from input]

---

## Executive Summary

[2-3 paragraph summary of key findings, including:]

- Current positioning status (strong/moderate/weak)
- Most critical gaps identified
- Biggest quick win opportunities
- Recommended priority focus

---

## Part 1: Positioning Evaluation

### SEO Competitor Landscape

| Competitor | Domain | Estimated Traffic | Content Volume | Strengths |
| ---------- | ------ | ----------------- | -------------- | --------- |

[competitor data]

### SERP Landscape Analysis

| Keyword | Your Position | Top Competitor | SERP Features | Content Type Winning |
| ------- | ------------- | -------------- | ------------- | -------------------- |

[keyword position data]

### Position Distribution

| Range       | Count | Implication                  |
| ----------- | ----- | ---------------------------- |
| 1-3         |       | Defend                       |
| 4-10        |       | Quick win candidates         |
| 11-20       |       | Striking distance (priority) |
| 21-50       |       | Requires investment          |
| Not ranking |       | New content needed           |

### Competitive Advantages & Disadvantages

**Advantages:**

- [list advantages]

**Disadvantages:**

- [list disadvantages]

---

## Part 2: Content Gap Analysis

### Content Coverage Matrix

| Topic Category | Comp A | Comp B | Comp C | You | Gap Status |
| -------------- | ------ | ------ | ------ | --- | ---------- |

[coverage data]

### Standard Content Gaps (vs. Competitors)

**Critical Gaps (all competitors have, you don't):**

1. [gap 1]
2. [gap 2]

**Important Gaps (most competitors have, you don't):**

1. [gap 1]
2. [gap 2]

### Blue Ocean Opportunities (Market Gaps)

| Topic/Question | Demand Evidence | SERP Quality | Priority |
| -------------- | --------------- | ------------ | -------- |

[blue ocean opportunities]

### Intent Coverage Gaps

| Intent Type   | Your Coverage | Gap? |
| ------------- | ------------- | ---- |
| Informational |               |      |
| Commercial    |               |      |
| Transactional |               |      |

---

## Part 3: Content Inventory Assessment

### Current Content Overview

- **Total content pages:** [X]
- **Estimated total words:** [X]
- **Average publish age:** [X months]

### Performance Distribution

| Status                        | Count | Percentage |
| ----------------------------- | ----- | ---------- |
| Top Performers (pos 1-3)      |       |            |
| Near Winners (pos 4-10)       |       |            |
| Striking Distance (pos 11-20) |       |            |
| Underperformers (pos 21-50)   |       |            |
| Zero Traffic                  |       |            |

### Content Actions Required

| Action      | Count | Priority      |
| ----------- | ----- | ------------- |
| KEEP        |       | Low (monitor) |
| IMPROVE     |       | High          |
| CONSOLIDATE |       | Medium        |
| PRUNE       |       | Medium        |

### Pages to Improve (Workflow R Candidates)

| URL | Current Position | Issue | Recommended Action |
| --- | ---------------- | ----- | ------------------ |

[improvement candidates]

### Pages to Consolidate

| Target Page | Pages to Merge | Redirect Plan |
| ----------- | -------------- | ------------- |

[consolidation plan]

### Pages to Prune

| URL | Reason | Action |
| --- | ------ | ------ |

[prune list]

---

## Part 4: Prioritized Opportunity Roadmap

### Tier 1: Quick Wins (Execute This Week)

| Opportunity | Type | Impact | Effort | Score | Action |
| ----------- | ---- | ------ | ------ | ----- | ------ |

[tier 1 items]

**Recommended First Action:** [specific actionable recommendation]

### Tier 2: Short-Term (This Month)

| Opportunity | Type | Impact | Effort | Score | Action |
| ----------- | ---- | ------ | ------ | ----- | ------ |

[tier 2 items]

### Tier 3: Medium-Term (This Quarter)

| Opportunity | Type | Impact | Effort | Score |
| ----------- | ---- | ------ | ------ | ----- |

[tier 3 items]

### Tier 4: Long-Term (Next Quarter+)

| Opportunity | Type | Impact | Effort | Score |
| ----------- | ---- | ------ | ------ | ----- |

[tier 4 items]

---

## Part 5: Next Steps

### Immediate Actions

1. **[Action 1]** - [specific instruction]
2. **[Action 2]** - [specific instruction]
3. **[Action 3]** - [specific instruction]

### Workflow Routing

**For Existing Content Optimization:**
→ Take "Pages to Improve" list to **Workflow R: Rank Enhancement**

**For New Content Creation:**
→ Take "Content Gap" topics to **SEO Boost API**
→ Research [specific topic] as primary keyword

**For Content Consolidation:**
→ Execute consolidation plan, then route merged pages to Workflow R

### Recommended Primary Keywords for API Research

Based on gap analysis, research these keywords first:

1. **[keyword 1]** - [rationale]
2. **[keyword 2]** - [rationale]
3. **[keyword 3]** - [rationale]

---

## Audit Metadata

- **Audit Type:** Comprehensive SEO Audit
- **Methodology:** Three Pillars Framework + KICP Decision Tree
- **Data Sources:** [list sources used]
- **Limitations:** [note any data gaps]

---

_Generated by Workflow A: Audit & Opportunity Analysis_

---

## Integration with Other Workflows

### Feeding SEO Boost API (Discovery)

The audit identifies primary keywords for research:

**From Content Gaps:**

- Topics competitors cover that you don't → research as primary keywords
- Blue ocean opportunities → research to validate demand

**From Positioning:**

- Topic clusters where you're weakest → research for expansion

**Handoff Format:**

```

Primary Keyword: [from audit gap analysis]
Rationale: [why this keyword matters based on audit]
Priority: [Tier 1/2/3 from prioritization]

```

### Feeding Workflow R (Rank Enhancement)

The audit identifies pages for optimization:

**From Content Inventory:**

- Pages marked IMPROVE → direct candidates for Workflow R
- Striking distance pages (position 11-20) → priority optimization
- Near winners (position 4-10) → quick win optimization

**Handoff Format:**

```

URL: [page URL]
Current Position: [from GSC or estimate]
Target Keyword: [primary keyword for page]
Issues Identified: [from audit]
Priority: [Tier 1/2/3]

```

### Audit Cadence

| Audit Type      | Frequency      | Scope                                       |
| --------------- | -------------- | ------------------------------------------- |
| Full Audit      | Annually       | Complete 6-stage process                    |
| Quarterly Check | Every 3 months | Stages 2 + 4 only (positioning + inventory) |
| Monthly Monitor | Monthly        | GSC trends + top performers check           |

---

## Execution Checklist

Before generating the audit artifact, ensure:

- [ ] Website URL and business context captured
- [ ] 3-5 SEO competitors identified
- [ ] SERP landscape analyzed for 10-15 core keywords
- [ ] Competitive scorecard completed
- [ ] Position distribution mapped
- [ ] Content coverage matrix created
- [ ] Standard content gaps identified
- [ ] Blue ocean opportunities explored
- [ ] Intent coverage assessed
- [ ] Content inventory collected (if applicable)
- [ ] KICP decision tree applied to all content
- [ ] All opportunities scored (Impact × Effort)
- [ ] Prioritized roadmap created with 4 tiers
- [ ] Workflow routing recommendations provided
- [ ] Primary keywords for API research identified

---

## Example Execution

### User Input

```

Website: https://example-saas.com
Business: Project management software for remote teams
Primary Topic: Remote team productivity
Target Audience: Team leads and managers at companies with 10-100 remote employees
Goals: Increase organic signups, establish thought leadership
Competitors: monday.com, asana.com, notion.so

```

### Claude Response

```

I'll conduct a comprehensive SEO audit for example-saas.com focusing on remote team productivity.

[Executes web_search for core topic queries]
[Identifies SERP landscape and competitor positions]
[Maps content coverage across competitors]
[Identifies content gaps and blue ocean opportunities]
[Assesses existing content with KICP framework]
[Scores and prioritizes all opportunities]

[Generates complete audit artifact]

**Key Findings:**

- You're invisible for 80% of core keywords (competitors dominate)
- Biggest gap: No comparison/alternative content (competitors have 10+ pieces each)
- Blue ocean opportunity: "Async communication for remote teams" has weak SERP
- 15 existing pages identified; 8 need improvement, 3 should consolidate

**Recommended First Action:**
Run Workflow R on your "Remote Team Guide" page (position 12 → potential top 5)

**Recommended Keywords for API Research:**

1. "project management for remote teams" (competitor gap)
2. "async communication tools" (blue ocean)
3. "remote team productivity tips" (intent gap - informational)

```

---

## Summary

Workflow A is the **strategic foundation** of the SEO system. It:

- ✅ Precedes keyword research with strategic audit
- ✅ Evaluates competitive positioning systematically
- ✅ Identifies both competitor gaps AND market gaps (Blue Ocean)
- ✅ Assesses existing content with KICP decision tree
- ✅ Prioritizes opportunities by Impact × Effort
- ✅ Routes findings to appropriate downstream workflow (API or Workflow R)
- ✅ Uses Three Pillars framework (Content, Technical, Backlinks)
- ✅ Produces actionable roadmap with 4 priority tiers
- ✅ Execution time: 15-30 minutes depending on site complexity
- ✅ Recommended cadence: Full audit annually, quarterly checks

**Key Innovation:** Transforms SEO from "keyword-first" to "strategy-first" by auditing what exists before researching what to create.

**Philosophical Shift:**

> "You're not ranking in a vacuum. Ranking is actually about outranking your search competitors." — Lidia Infante

Workflow A ensures you understand the battlefield before deploying resources.
