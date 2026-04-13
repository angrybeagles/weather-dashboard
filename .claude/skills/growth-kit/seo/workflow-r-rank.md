# Workflow R: Rank

- **Purpose:** Optimize existing pages for improved rankings without creating new content
- **Platform:** Claude Desktop (or Claude Code) with web_search + SEO Boost API (optional)
- **Input:** Page URL + current position data + target keyword
- **Output:** Single comprehensive Markdown artifact with optimization plan and implementation
- **Capacity:** 1-5 pages per execution

> **API Synergy:** This workflow can be enhanced with SEO Boost API data for keyword metrics (MSV, difficulty, intent) and PAA questions. The API provides quantitative data that web_search alone cannot deliver, while web_search excels at visual SERP feature detection.

> **Prerequisite Check:** Before optimizing, verify your page is indexed by Google. Search `site:yourdomain.com/page-url` in Google. If your page does not appear, use **Workflow I: Indexing** first to submit it for indexing. Optimizing content that Google has not discovered is wasted effort.

---

## Why Rank Enhancement Matters

> "Focus your content update efforts on pages ranking in positions 4-20, on the first and second page of search results. These 'striking distance' pages are already somewhat relevant to Google, making them prime candidates for improvement." — Tom Winter, SEOwind

**The ROI Case:**

- Content refresh produces faster results than new content creation
- Pages ranking 11-20 need relatively small improvements to break into top 10
- Moving from position 8 to position 3 dramatically increases visibility
- Position #1 has 25% chance of being cited in AI Overviews

**What This Workflow Does:**

1. Analyzes existing page vs. top-ranking competitors
2. Identifies specific enhancement opportunities
3. Optimizes meta titles/descriptions using semantic principles
4. Expands and improves content strategically
5. Restructures internal linking for authority flow
6. Tracks expected timeline and measurement approach

---

## The Striking Distance Paradigm

### Definition

**Striking Distance Keywords:** Search terms where your website ranks between positions 5-20 (some extend to 5-30). These represent highest-value optimization targets because:

1. Google already recognizes your content as relevant
2. Small improvements can yield significant gains
3. ROI typically exceeds new content creation
4. Results appear faster (weeks vs. months)

### Position-Based Strategy

| Position Range | Priority | Typical Action                     | Expected Effort |
| -------------- | -------- | ---------------------------------- | --------------- |
| 1-3            | Low      | Maintain only, minor tweaks        | Minimal         |
| 4-10           | High     | Aggressive optimization            | Medium          |
| 11-20          | High     | Content refresh + optimization     | Medium-High     |
| 21-50          | Medium   | Evaluate ROI vs. new content       | High            |
| 50+            | Low      | Consider major rewrite or new page | Very High       |

### Why Positions 5-20 Matter More Now

Research from SEOgets (2024) on AI search inclusion:

- Position #1: 25% chance of being cited in AI Overviews
- Top 3 positions: Significantly higher likelihood across AI platforms
- Moving up just a few positions dramatically transforms visibility in both traditional search and AI-powered search

---

## Category-Based Optimization Protocols

When pages arrive from Workflow A with a category classification, apply the corresponding protocol for targeted, efficient optimization.

### Protocol 1: CTR Optimization (30 minutes)

**Trigger:** Impressions >1,000/mo + CTR <5% + Position 5-15

**Problem:** Page is visible but not getting clicked. Users see it but choose competitors.

**Fix Protocol:**

1. **Analyze competing titles** (5 min) - What makes top 3 titles more clickable?
2. **Rewrite title tag** (10 min) - Add power words, numbers, brackets, urgency
3. **Rewrite meta description** (10 min) - Clear value prop, call-to-action, differentiation
4. **Verify H1 alignment** (5 min) - H1 should reinforce title promise

**Success Metrics:** CTR improvement 2-3x within 2-4 weeks

**Example Transformation:**

- Before: "Guide to Remote Work Tools" (CTR 2.1%)
- After: "15 Remote Work Tools That Actually Work [2025 Tested]" (CTR 5.8%)

---

### Protocol 2: Striking Distance Expansion (90 minutes)

**Trigger:** Position 8-15 + Impressions >500/mo + Words >1,000

**Problem:** Good content ranking on page 2 or bottom of page 1. Close to breakthrough.

**Fix Protocol:**

1. **Competitor content analysis** (15 min) - What do top 3 have that you don't?
2. **Add 300-500 words** (30 min) - Fill topic gaps identified in analysis
3. **Add FAQ section** (20 min) - Use PAA questions from SERP or API
4. **Add 5-8 internal links TO this page** (15 min) - From high-authority pages
5. **Add 3-5 internal links FROM this page** (10 min) - To related content

**Success Metrics:** Move to position 1-7 within 4-8 weeks

**Key Insight:** These pages need depth and authority signals, not complete rewrites.

---

### Protocol 3: Engagement Fix (45 minutes)

**Trigger:** Position 1-5 + Bounce >70% + Time on page <1 min

**Problem:** Ranking well but failing to satisfy users. Google will eventually demote.

**Fix Protocol:**

1. **Improve introduction** (15 min) - Hook reader in first 100 words, promise value
2. **Add table of contents** (5 min) - Jump links for scanability
3. **Improve formatting** (15 min) - Headers, bullets, whitespace, callout boxes
4. **Add visual elements** (10 min) - Images, diagrams, or embedded media breaks

**Success Metrics:** Bounce rate <55%, time on page >2 min within 2-4 weeks

**Warning Signs to Address:**

- Wall of text with no breaks
- Intro that buries the lede
- Missing jump navigation on long content
- No visual relief for reader

---

### Protocol 4: Thin Authority Expansion (2 hours)

**Trigger:** Position 4-10 + Words <800 + Backlinks >3

**Problem:** Page has earned authority (backlinks) but content is too thin to rank higher.

**Fix Protocol:**

1. **Audit current content** (15 min) - What's covered vs. what's missing?
2. **Research competitor depth** (15 min) - What topics do top 3 cover?
3. **Expand to 1,500-2,000 words** (60 min) - Add missing subtopics, examples, data
4. **Add expert quotes or citations** (15 min) - E-E-A-T signals
5. **Restructure headings** (15 min) - Proper H2/H3 hierarchy

**Success Metrics:** Move to position 1-3 within 6-10 weeks

**Key Insight:** This page already has authority from backlinks. The constraint is content depth.

---

### Protocol 5: Orphan Rescue (20 minutes)

**Trigger:** Sessions >100/mo + Internal links 0-1 + Age >6 months

**Problem:** Valuable content that's disconnected from site structure. PageRank isn't flowing to it.

**Fix Protocol:**

1. **Identify high-authority source pages** (5 min) - Homepage, pillar content, top traffic pages
2. **Add 5-8 contextual internal links** (15 min) - Natural anchor text, in-content placement

**Link Placement Priority:**

1. Homepage (if topically relevant)
2. Related pillar/hub pages
3. Other posts in same topic cluster
4. Resource/tools pages
5. Recent related articles

**Success Metrics:** Position improvement within 4-6 weeks as PageRank flows

**Why This Works:** Internal links are the fastest way to signal page importance to Google.

---

### Protocol Selection Flowchart

```
Page arrives from Workflow A
│
├─ Has Category?
│   YES
│   └─ Apply corresponding protocol
│
└─ No Category?
    │
    └─ Run full Stage 1-6 analysis
        (generic enhancement)
```

**Multiple Categories:** If page qualifies for multiple categories, prioritize:

1. Engagement Fix (if ranking 1-5, protect position first)
2. CTR Optimization (quick win, high visibility)
3. Orphan Rescue (quick win, low effort)
4. Striking Distance (medium effort, high impact)
5. Thin Authority (high effort, high impact)

---

## Input Requirements

### Required Information

```
Page URL: [full URL of page to optimize]
Target Keyword: [primary keyword this page should rank for]
Current Position: [current ranking position, if known]
```

### Optional (Enhances Analysis)

```
Secondary Keywords: [other keywords this page ranks for]
GSC Data: [impressions, clicks, CTR for this page]
Current Word Count: [approximate length]
Last Updated: [when page was last modified]
Backlink Count: [number of referring domains, if known]
Business Priority: [how important is this page to revenue/goals]
```

### Minimal Viable Input

At minimum, provide:

- Page URL
- Target keyword

Current position and other data can be estimated through SERP analysis.

---

## Processing Stages (Autonomous Execution)

### STAGE 0: SEO Data Enrichment (Optional)

> **When to use:** Run Stage 0 when you have access to the SEO Boost API and want quantitative keyword metrics. Skip if working without API access - the workflow functions fully with web_search alone.

**Step 0.1: API Call**

Call the SEO Boost API with your target keyword:

```
POST /api/v1/seo
Authorization: Bearer sk_seoboost_xxx
{
  "keyword": "[target keyword]",
  "serp_depth": 20,
  "ideas_limit": 50,
  "language_code": "en",
  "location_name": "United States"
}
```

**Step 0.2: Extract Enrichment Data**

From the API response, capture:

| Data Point         | Location in Response            | Use In Workflow                |
| ------------------ | ------------------------------- | ------------------------------ |
| Search Volume      | `keywords[0][1]` (msv)          | Set realistic expectations     |
| Keyword Difficulty | `keywords[0][2]`                | Estimate effort required       |
| Search Intent      | `keywords[0][3]`                | Validate content alignment     |
| Competition        | `keywords[0][4]`                | Gauge competitive landscape    |
| CPC                | `keywords[0][5]`                | Understand commercial value    |
| PAA Questions      | `paa_questions[]`               | Feed FAQ section (Stage 3)     |
| Related Keywords   | `keywords` (type: "KW Related") | Secondary optimization targets |
| SERP Competitors   | `serp_results[]`                | Pre-populate competitor list   |

**Step 0.3: Document Baseline Metrics**

```
Target Keyword: [keyword]
Monthly Search Volume: [msv]
Keyword Difficulty: [0-100]
Primary Intent: [informational/commercial/transactional/navigational]
Competition Level: [LOW/MEDIUM/HIGH]
Estimated CPC: $[value]

PAA Questions Found: [count]
Related Keywords: [count]
SERP Results Retrieved: [count]
```

**Step 0.4: Prepare Stage 1 Inputs**

- Use `serp_results` to identify top competitors (validates web_search)
- Save `paa_questions` for FAQ planning in Stage 3
- Note high-value related keywords for secondary targeting
- Use difficulty score to set realistic timeline expectations

**Credit Cost:** ~7 credits per API call (varies by result volume)

---

### STAGE 1: Page Analysis & Competitor Benchmarking

**Step 1.1: Current State Assessment**

Fetch and analyze the target page:

```
web_fetch("[page URL]")
```

**Document:**

- Current title tag
- Current meta description
- H1 and heading structure
- Approximate word count
- Content sections/topics covered
- Internal links present
- Schema markup (if any)
- Publish/update date

**Step 1.2: SERP Analysis**

Execute web_search for target keyword:

```
web_search("[target keyword]")
```

**Document for top 5 results:**

| Position | URL | Title | Description | Content Type |
| -------- | --- | ----- | ----------- | ------------ |
| 1        |     |       |             |              |
| 2        |     |       |             |              |
| 3        |     |       |             |              |
| 4        |     |       |             |              |
| 5        |     |       |             |              |

**Note:**

- SERP features present (featured snippet, PAA, AI Overview)
- Content format winning (guide, listicle, tool, video)
- Average estimated word count of top 3
- Common topics/sections covered

**Step 1.3: Competitor Content Analysis**

For top 3 competitors, analyze:

```
web_fetch("[competitor URL]")
```

**Document:**

- Heading structure (H2s, H3s)
- Topics/sections covered
- Unique value proposition
- Content depth and examples
- Internal/external links
- Schema markup used
- E-E-A-T signals (author, credentials, sources)

**Step 1.4: Gap Identification**

Compare your page vs. competitors:

| Element          | Your Page | Competitor Avg | Gap? |
| ---------------- | --------- | -------------- | ---- |
| Word count       |           |                |      |
| Sections covered |           |                |      |
| Heading depth    |           |                |      |
| Examples/data    |           |                |      |
| Visual elements  |           |                |      |
| Internal links   |           |                |      |
| Schema markup    |           |                |      |
| E-E-A-T signals  |           |                |      |

---

### STAGE 2: Meta Optimization (Macro + Microsemantics)

**The Constellation of Signals Approach:**

> "My goal isn't just to optimise the <title> tag in isolation. It's to create a 'constellation of signals' – the <title>, the <h1>, the URL, the intro paragraph – that are so thematically coherent that they leave no room for algorithmic ambiguity." — Hobo Web

**Step 2.1: Title Tag Optimization**

**Current Title Analysis:**

- Does it include target keyword?
- Is keyword in first 3-5 words?
- Does it match search intent?
- Is it compelling (would you click)?
- Length (aim for 50-60 characters)

**Title Optimization Framework:**

1. **RDF Thinking:** Consider subject-predicate-object relationships
2. **Semantic Similarity:** Title should convey same meaning as content
3. **Entity Recognition:** Include core entities Google expects
4. **Contextual Vectors:** Align with broader topical context

**Generate 3 Title Options:**

```
Option 1 (SEO-focused): [keyword first, clear intent]
Option 2 (CTR-focused): [hook, curiosity, power words]
Option 3 (Balanced): [keyword + compelling element]
```

**Power Words to Consider:**

- Guide, Complete, Ultimate, Proven, Best
- 2025, Updated, New
- Step-by-Step, Easy, Quick, Simple
- Free, Essential, Expert

**Step 2.2: Meta Description Optimization**

**Current Description Analysis:**

- Does it include target keyword?
- Does it match search intent?
- Does it have a clear value proposition?
- Does it include a call-to-action?
- Length (aim for 150-160 characters)

**Generate Optimized Description:**

- Include target keyword naturally
- State the main benefit/value
- Match the search intent
- Add urgency or differentiation
- End with implicit or explicit CTA

**Step 2.3: H1 Alignment**

Ensure H1:

- Matches or closely aligns with title
- Contains target keyword
- Is the only H1 on page
- Accurately describes content

**Step 2.4: URL Assessment**

Check if URL:

- Contains target keyword (don't change if already indexed)
- Is reasonably short
- Uses hyphens, not underscores
- Is lowercase

**Note:** Only recommend URL change if critical — redirects have transition cost.

---

### STAGE 3: Content Enhancement

**Step 3.1: Content Expansion Strategy**

Based on competitor gap analysis, identify:

**Topics to Add:**
| Topic | Why Missing Matters | Recommended Words |
|-------|--------------------|--------------------|
| | | |

**Sections to Expand:**
| Section | Current Coverage | Recommended Expansion |
|---------|------------------|----------------------|
| | | |

**Step 3.2: Content Quality Improvements**

**Checklist:**

- [ ] Add specific examples (competitors have X, we have Y)
- [ ] Include current data/statistics (with sources)
- [ ] Add expert quotes or citations
- [ ] Include step-by-step instructions where appropriate
- [ ] Add FAQ section if competitors have one
- [ ] Improve introduction hook
- [ ] Strengthen conclusion with clear takeaway

> **API Synergy (Stage 0):** If you ran Stage 0, use the `paa_questions` array directly for FAQ section planning. These are real questions users ask about your target keyword - answering them improves topical coverage and featured snippet eligibility.

**Step 3.3: E-E-A-T Signal Enhancement**

**Experience Signals:**

- Add first-hand experience mentions
- Include original screenshots/photos
- Reference personal case studies or results

**Expertise Signals:**

- Author bio with relevant credentials
- Technical depth appropriate to topic
- Industry-specific terminology used correctly

**Authoritativeness Signals:**

- Citations to authoritative sources
- Mention of relevant credentials/awards
- Original data or research if available

**Trust Signals:**

- Clear, transparent sourcing
- Updated date visible
- Contact information accessible
- Accurate, verifiable claims

**Step 3.4: Content Structure Optimization**

**Heading Hierarchy Review:**

- Logical H2 → H3 → H4 progression
- Headings use keywords naturally
- Headings are descriptive (not clever)
- Each major section has clear heading

**Scanability Improvements:**

- Key points in bold or callout boxes
- Bulleted lists for series of items
- Tables for comparison data
- Short paragraphs (3-4 sentences max)

---

### STAGE 4: Internal Linking Restructure

**Step 4.1: Current Internal Link Audit**

**Links FROM this page:**
| Anchor Text | Target URL | Relevant? |
|-------------|------------|-----------|
| | | |

**Links TO this page:**
| Source URL | Anchor Text | Authority Level |
|------------|-------------|-----------------|
| | | |

**Step 4.2: Internal Link Opportunities**

**Find pages to link FROM (boost this page):**

- High-authority pages on your site
- Topically related content
- Pages with existing organic traffic

**Query to identify:**

```
site:yourdomain.com [target keyword topic]
```

**Recommended new links TO this page:**
| Source Page | Suggested Anchor Text | Location in Content |
|-------------|----------------------|---------------------|
| | | |

**Step 4.3: Links FROM this page (distribute authority)**

**Add links to:**

- Related supporting content (spoke → spoke)
- Pillar/hub page if exists (spoke → hub)
- Conversion pages where relevant (content → product)

**Anchor Text Best Practices:**

- Descriptive, not generic ("click here")
- Include target keyword variations naturally
- Vary anchor text (not all exact match)
- Contextually placed within sentences

---

### STAGE 5: Technical & Schema Enhancements

**Step 5.1: Quick Technical Checks**

**Verify:**

- [ ] Page is indexable (no noindex)
- [ ] Page loads quickly (<3 seconds)
- [ ] Mobile-friendly rendering
- [ ] No broken links on page
- [ ] Images have alt text
- [ ] Canonical tag correct

**Step 5.2: Schema Markup Recommendations**

Based on content type, recommend appropriate schema:

| Content Type   | Schema Type   | Priority |
| -------------- | ------------- | -------- |
| How-to guide   | HowTo         | High     |
| FAQ content    | FAQPage       | High     |
| Product review | Review        | High     |
| Article/blog   | Article       | Medium   |
| Local content  | LocalBusiness | High     |
| Recipe         | Recipe        | High     |

**Generate Schema Example:**

```json
{
  "@context": "https://schema.org",
  "@type": "[appropriate type]",
  ...
}
```

**Step 5.3: Featured Snippet Optimization**

If featured snippet opportunity exists:

**For Definition Snippets:**

- Add clear definition in first paragraph
- Use "What is [keyword]?" as H2
- Keep definition to 40-60 words

**For List Snippets:**

- Use numbered or bulleted lists
- Clear H2 heading before list
- 5-8 items typically captured

**For Table Snippets:**

- Use HTML tables
- Clear column headers
- Comparative data works well

---

### STAGE 6: Implementation Plan & Tracking

**Step 6.1: Prioritized Action List**

Organize all recommendations by priority:

**Tier 1: Quick Wins (Implement Today)**
| Action | Effort | Expected Impact |
|--------|--------|-----------------|
| Update title tag | 5 min | High (CTR) |
| Update meta description | 5 min | Medium (CTR) |
| Fix H1 alignment | 5 min | Low-Medium |
| Add schema markup | 15 min | Medium |

**Tier 2: Content Updates (This Week)**
| Action | Effort | Expected Impact |
|--------|--------|-----------------|
| Expand [section] | 1-2 hours | High |
| Add [missing topic] | 1 hour | Medium |
| Update statistics | 30 min | Medium |
| Add internal links | 30 min | Medium |

**Tier 3: Major Enhancements (If Needed)**
| Action | Effort | Expected Impact |
|--------|--------|-----------------|
| Full content refresh | 4-8 hours | High |
| Add new sections | 2-4 hours | Medium-High |
| Create supporting visuals | 2-4 hours | Medium |

**Step 6.2: Implementation Checklist**

```
□ Update title tag
□ Update meta description
□ Verify H1 alignment
□ Expand content sections
□ Add missing topics
□ Update statistics/dates
□ Add internal links TO this page
□ Add internal links FROM this page
□ Implement schema markup
□ Optimize for featured snippet (if applicable)
□ Update publish date
□ Request reindexing in GSC
```

**Step 6.3: Tracking Setup**

**Metrics to Monitor:**

- Position for target keyword
- Position for secondary keywords
- Organic clicks to this page
- Impressions
- CTR
- Average position (aggregate)

**Timeline Expectations:**

| Change Type        | Expected Timeline                    |
| ------------------ | ------------------------------------ |
| Title/meta changes | 1-2 weeks for CTR impact             |
| Content expansion  | 4-8 weeks for ranking impact         |
| Internal linking   | 4-6 weeks for authority flow         |
| Schema markup      | 2-4 weeks for rich result appearance |
| Full refresh       | 8-12 weeks for full effect           |

**Step 6.4: Iteration Decision Points**

**After 4-6 weeks, evaluate:**

If position improved but goal not achieved:

- Add more internal links
- Consider backlink acquisition
- Expand content further

If position unchanged:

- Re-analyze competitor content
- Check for technical issues
- Evaluate if wrong content type for intent
- Consider more significant rewrite

If position declined:

- Check for algorithm update impact
- Review changes made — any negative impact?
- Analyze SERP changes

**When to move on:**

- 3+ optimization attempts with no movement
- Competitors significantly stronger (insurmountable gap)
- Search intent has shifted
- Better ROI available elsewhere

---

## Output Format: Markdown Artifact Template

# Rank Enhancement Plan: [Page Title]

**Page URL:** [URL]
**Target Keyword:** [keyword]
**Current Position:** [position]
**Audit Date:** [date]

---

## Executive Summary

[2-3 sentences on key findings and biggest opportunities]

**Primary Recommendation:** [single most important action]

**Expected Outcome:** [realistic prediction based on changes]

---

## Part 1: Current State Analysis

### Page Snapshot

| Element          | Current State         |
| ---------------- | --------------------- |
| Title            | [current title]       |
| Meta Description | [current description] |
| H1               | [current H1]          |
| Word Count       | [approximate]         |
| Last Updated     | [date if visible]     |

### Competitive Landscape

| Position  | Domain     | Title | Key Differentiator |
| --------- | ---------- | ----- | ------------------ |
| 1         |            |       |                    |
| 2         |            |       |                    |
| 3         |            |       |                    |
| Your Page | [position] |       |                    |

### Gap Analysis

| Element        | Your Page | Top 3 Average | Gap |
| -------------- | --------- | ------------- | --- |
| Word count     |           |               |     |
| Topics covered |           |               |     |
| Examples/data  |           |               |     |
| Internal links |           |               |     |
| Schema         |           |               |     |

---

## Part 2: Meta Optimization

### Title Tag

**Current:** [current title]

**Recommended Options:**

1. **[Option 1]** — [rationale]
2. **[Option 2]** — [rationale]
3. **[Option 3]** — [rationale]

**Selected:** [recommended choice]

### Meta Description

**Current:** [current description]

**Recommended:** [new description]

**Rationale:** [why this is better]

### H1 Tag

**Current:** [current H1]
**Recommended:** [new H1 if different]

---

## Part 3: Content Enhancement Plan

### Topics to Add

| Topic | Why It Matters | Suggested Word Count |
| ----- | -------------- | -------------------- |
|       |                |                      |

### Sections to Expand

| Section | Current State | Enhancement Needed |
| ------- | ------------- | ------------------ |
|         |               |                    |

### Content Quality Checklist

- [ ] Add [X] specific examples
- [ ] Update statistics to [year]
- [ ] Add FAQ section with [X] questions
- [ ] Strengthen introduction with [hook type]
- [ ] Add [E-E-A-T signal type]

---

## Part 4: Internal Linking Plan

### Links TO Add (Boost This Page)

| Source Page | Anchor Text | Priority |
| ----------- | ----------- | -------- |
|             |             |          |

### Links FROM This Page

| Target Page | Anchor Text | Context |
| ----------- | ----------- | ------- |
|             |             |         |

---

## Part 5: Technical & Schema

### Technical Checklist

- [ ] Page indexable: [Yes/No]
- [ ] Mobile-friendly: [Yes/No]
- [ ] Page speed: [Fast/Medium/Slow]
- [ ] Images optimized: [Yes/No]

### Schema Recommendation

**Type:** [schema type]

```json
[schema markup]
```

### Featured Snippet Opportunity

**Type:** [definition/list/table/none]
**Optimization:** [specific recommendation]

---

## Part 6: Implementation Roadmap

### Tier 1: Quick Wins (Today)

| Action | Time | Impact |
| ------ | ---- | ------ |
|        |      |        |

### Tier 2: Content Updates (This Week)

| Action | Time | Impact |
| ------ | ---- | ------ |
|        |      |        |

### Tier 3: Major Work (If Needed)

| Action | Time | Impact |
| ------ | ---- | ------ |
|        |      |        |

### Complete Checklist

```
□ Title tag updated
□ Meta description updated
□ H1 aligned
□ Content expanded: [sections]
□ Internal links added: [count] TO, [count] FROM
□ Schema implemented
□ Publish date updated
□ Reindexing requested
```

---

## Part 7: Tracking Plan

### Metrics to Monitor

| Metric   | Current | Target | Timeline |
| -------- | ------- | ------ | -------- |
| Position | [X]     | [Y]    | [weeks]  |
| Clicks   | [X]     | [Y]    | [weeks]  |
| CTR      | [X%]    | [Y%]   | [weeks]  |

### Check-In Schedule

- **Week 2:** Check CTR changes from meta updates
- **Week 4:** Initial position movement assessment
- **Week 8:** Full impact evaluation
- **Week 12:** Final assessment and iteration decision

### Iteration Triggers

**If no movement by Week 6:**

- [ ] Re-analyze competitor changes
- [ ] Evaluate backlink gap
- [ ] Consider content type mismatch

---

_Generated by Workflow R: Rank_

---

## Batch Processing (Multiple Pages)

For optimizing multiple pages in one session:

### Input Format

```

Page 1:

- URL: [url]
- Target Keyword: [keyword]
- Current Position: [position]

Page 2:

- URL: [url]
- Target Keyword: [keyword]
- Current Position: [position]

[...up to 5 pages]

```

### Batch Output

Generate individual enhancement plans, then summary:

## Batch Summary: [X] Pages Optimized

| Page | Keyword | Current | Target | Top Action |
| ---- | ------- | ------- | ------ | ---------- |
|      |         |         |        |            |

### Priority Order

1. [Page X] — Highest impact, lowest effort
2. [Page Y] — High impact, medium effort
3. [Page Z] — Medium impact, low effort

### Aggregate Time Estimate

- Quick wins (all pages): [X hours]
- Full implementation: [X hours]

---

## Integration with Other Workflows

### From Workflow A (Audit)

Workflow A identifies pages marked "IMPROVE" - these are direct candidates for Workflow R.

**Handoff format from W0:**

```
URL: [from audit]
Current Position: [from audit]
Target Keyword: [from audit]
Issues Identified: [from audit gap analysis]
Priority: [Tier from audit]
```

### Connection to Workflows K-S-C (New Content)

If Workflow R analysis reveals:

- Content type mismatch → Consider new page (Workflows K-S-C)
- Major rewrite needed → May be faster to start fresh
- Topic cannibalization → Consolidate, then optimize winner

### Iteration Loop

```
Workflow R → Implement → Monitor (4-8 weeks) → Evaluate
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    │                              │                              │
                    ▼                              ▼                              ▼
             Goal Achieved                  Partial Progress              No Progress
                    │                              │                              │
                    ▼                              ▼                              ▼
               MAINTAIN                    RE-RUN WORKFLOW 4               ESCALATE
          (move to next page)              (deeper optimization)       (new page or link building)
```

---

## Execution Checklist

Before generating enhancement plan, ensure:

- [ ] Page URL accessible and indexed
- [ ] Target keyword identified
- [ ] Current position known or estimated
- [ ] Top 3-5 competitors analyzed
- [ ] Content gaps documented
- [ ] Meta tags evaluated
- [ ] Internal linking audited
- [ ] Schema opportunities identified
- [ ] Implementation prioritized by effort/impact
- [ ] Tracking plan established
- [ ] Timeline expectations set

---

## Summary

Workflow R is the **existing content optimization engine**. It:

- ✅ Analyzes pages vs. top-ranking competitors
- ✅ Identifies specific gaps and opportunities
- ✅ Optimizes meta titles using Constellation of Signals approach
- ✅ Provides content expansion recommendations
- ✅ Restructures internal linking for authority flow
- ✅ Recommends appropriate schema markup
- ✅ Optimizes for featured snippets where applicable
- ✅ Creates prioritized implementation roadmap
- ✅ Establishes tracking and iteration plan
- ✅ Handles batch processing (1-5 pages)
- ✅ Execution time: 10-20 minutes per page
- ✅ Optional SEO Boost API integration for quantitative metrics (MSV, KD, intent, PAA)

**Key Insight:**

> "Content refresh can produce impressive results in a matter of days. They will also continue to show increased gains over several months." — Big Sea Agency

**Recommended Cadence:**

- Run monthly on striking distance pages (positions 11-20)
- Run quarterly on near-winners (positions 4-10)
- Run as-needed on underperformers flagged in Workflow A
