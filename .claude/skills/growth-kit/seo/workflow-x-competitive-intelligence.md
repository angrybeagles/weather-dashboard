# Workflow X: Competitive Intelligence

- **Purpose:** Systematic competitive analysis, gap identification, and ongoing monitoring
- **Platform:** Claude Desktop (or Claude Code) with web_search + SEO tools (optional)
- **Input:** Website URL + target niche/topic + known competitors (optional)
- **Output:** Comprehensive competitive intelligence report with actionable strategy
- **Capacity:** 1 domain with 3-5 competitor comparisons per execution

---

## Why a Dedicated Competitor Workflow?

**The Problem:**

Competitor analysis was scattered across multiple workflows without a dedicated entry point. SERP analysis in Workflows C and R is competitor analysis in disguise but wasn't framed that way. Users asking "help me analyze my competitors" had no clear path.

**Workflow X provides:**

1. Dedicated entry point for competitive intelligence
2. Share of Voice as the primary competitive metric
3. Complete gap analysis methodology (keywords, content, backlinks, technical)
4. Structured response frameworks for competitor movements
5. Ongoing monitoring system with defined cadences
6. AI visibility tracking for the evolving search landscape

---

## When to Use This Workflow

**Use Workflow X when:**

- Starting competitive analysis for the first time
- Refreshing competitor intelligence (quarterly recommended)
- Investigating why competitors outrank you
- Planning competitive content strategy
- Responding to competitor ranking gains
- Setting up systematic monitoring

**Use other workflows when:**

- Page-level optimization (Workflow R handles immediate SERP competition)
- Content creation (Workflow C does tactical SERP analysis)
- Full audit (Workflow A includes competitor analysis as one pillar)

**Relationship to Workflow A:**

- Workflow A produces competitor analysis as part of a broader audit
- Workflow X goes deeper on competitive intelligence specifically
- Run Workflow X before Workflow A for comprehensive competitor profile, OR
- Extract competitor data from Workflow A and use Workflow X for deeper analysis

---

## Input Requirements

### Required Information

```
Website URL: [your domain, e.g., https://example.com]
Primary Topic/Niche: [main subject area for competition analysis]
Target Keywords: [3-5 core keywords you're competing for]
```

### Optional (Enhances Analysis)

```
Known Competitors: [domains you consider competitors]
Current Rankings: [your positions for target keywords]
GSC Export: [queries, pages, clicks, impressions, position data]
SEO Tool Access: [Ahrefs, SEMrush, SE Ranking, etc.]
```

### Minimal Viable Input

At minimum, provide:

- Website URL
- Primary topic/niche

Competitors and other data can be discovered through SERP analysis.

---

## Processing Stages (Autonomous Execution)

### STAGE 1: Competitor Discovery & Classification

**Purpose:** Identify who you're actually competing against in SERPs

**Step 1.1: SERP-Based Competitor Identification**

Execute web_search for 7-10 core topic queries:

```
web_search("[primary topic] guide")
web_search("[primary topic] best practices")
web_search("[primary topic] for [target audience]")
web_search("best [primary topic] tools")
web_search("[primary topic] comparison")
web_search("how to [primary topic]")
web_search("[primary topic] examples")
```

**Document for each search:**

- All unique domains in top 10 results
- Frequency of appearance across all searches
- Average position per domain
- SERP features owned (snippets, PAA, video)

**Critical Insight:**

> "SEO competitors may differ from business competitors. Focus on who appears for YOUR target keywords, not who you consider market competitors."

**Step 1.2: Hierarchical Competitor Classification**

Apply this classification framework:

| Competitor Type          | Definition                             | Monitoring Level | Analysis Depth |
| ------------------------ | -------------------------------------- | ---------------- | -------------- |
| **Direct Competitors**   | Appear in top 10 for 4+ of 7 queries   | High (weekly)    | Full analysis  |
| **Partial Competitors**  | Appear in top 10 for 1-3 queries       | Medium (monthly) | Key metrics    |
| **Domain Authorities**   | High-authority sites (news, Wikipedia) | Low (quarterly)  | Tactics only   |
| **Emerging Competitors** | New domains gaining visibility         | High (weekly)    | Trajectory     |

**Output: Competitor Watch List**

| Rank | Competitor | Domain | Appears in X/7 | Avg Position | Type | Threat Level |
| ---- | ---------- | ------ | -------------- | ------------ | ---- | ------------ |
| 1    |            |        |                |              |      |              |
| 2    |            |        |                |              |      |              |
| 3    |            |        |                |              |      |              |

**Step 1.3: Competitor Quick Profile**

For top 5 competitors, assess via site: searches:

```
web_search("site:[competitor.com] [primary topic]")
```

**Document:**

- Estimated content volume (indexed pages on topic)
- Content depth indicators (word counts, comprehensiveness)
- Publishing frequency (recent publication dates)
- Specialization level (niche focus vs broad)
- Notable content series or topic hubs

---

### STAGE 2: Competitive Scorecard & Share of Voice

**Purpose:** Quantify your competitive position with objective metrics

**Step 2.1: Share of Voice Calculation (Primary Metric)**

Share of Voice (SOV) measures your organic visibility relative to competitors.

**SOV Formula:**

```
SEO Share of Voice = (Your Visibility Score / Total Market Visibility) x 100
```

**Calculation Methods:**

**Method 1: Position-Based (Without Premium Tools)**

1. Define 50-100 target keywords from your niche
2. Assign visibility score per position:
   - Position 1: 30%
   - Position 2: 15%
   - Position 3: 10%
   - Positions 4-5: 5% each
   - Positions 6-10: 2% each
   - Positions 11+: 0%
3. Sum visibility scores across all keywords
4. Calculate: Your total / Maximum possible total

**Method 2: Tool-Based (With SEO Tools)**

| Tool       | SOV Feature                         | Access      |
| ---------- | ----------------------------------- | ----------- |
| Semrush    | "Market Share" in Position Tracking | $129+/month |
| Ahrefs     | "Visibility" metric in Rank Tracker | $129+/month |
| SE Ranking | Visibility tracking                 | $52+/month  |

**SOV Interpretation:**

| SOV Range | Position           | Strategic Implication    |
| --------- | ------------------ | ------------------------ |
| >30%      | Market Leader      | Defend and expand        |
| 15-30%    | Strong Competitor  | Targeted offensive       |
| 5-15%     | Emerging Player    | Build authority          |
| <5%       | Minimal Visibility | Significant gap to close |

**Step 2.2: AI Visibility Tracking**

AI Overviews now appear in 15-50% of searches. Track separately:

```
For each target keyword:
- Is AI Overview present?
- Which domains are cited?
- Are you cited?
- Are competitors cited?
```

**AI Citation Indicators:**

- Clear, factual, definitive statements
- Well-structured content with headers
- Strong E-E-A-T signals
- Content that directly answers the query
- Recent/updated content

**Step 2.3: Content Velocity Comparison**

Publishing frequency impacts topical authority.

**Tracking Method:**

```
web_search("site:[competitor.com] [topic]" filtered by past month)
```

**Document:**

| Competitor | Posts/Month | Refresh Rate | Content Expansion | Velocity Level |
| ---------- | ----------- | ------------ | ----------------- | -------------- |
|            |             |              |                   |                |

**Velocity Benchmarks:**

- High (20+ pieces/month): Major publishers, difficult to match
- Medium (5-20 pieces/month): Active programs, compete on quality
- Low (<5 pieces/month): Opportunity to outpace

**Benchmark:** Match or exceed top competitor velocity. Minimum for authority: 20,000-30,000 words monthly.

**Step 2.4: Complete Competitive Scorecard**

| Dimension               | Your Site | Comp A | Comp B | Comp C | Your Rank |
| ----------------------- | --------- | ------ | ------ | ------ | --------- |
| **Share of Voice**      |           |        |        |        | /4        |
| **AI Citations**        |           |        |        |        | /4        |
| **Content Volume**      |           |        |        |        | /4        |
| **Content Velocity**    |           |        |        |        | /4        |
| **Domain Authority**    |           |        |        |        | /4        |
| **SERP Features Owned** |           |        |        |        | /4        |
| **E-E-A-T Signals**     |           |        |        |        | /4        |
| **Overall Position**    |           |        |        |        | /4        |

---

### STAGE 3: Keyword Gap Analysis

**Purpose:** Identify keyword opportunities competitors rank for that you don't

**Step 3.1: With SEO Tools**

If you have Ahrefs, SEMrush, or similar:

1. Run keyword gap report comparing your domain to 3-5 competitors
2. Filter for:
   - Keywords with 100+ monthly searches
   - Keywords where competitors rank top 20, you don't
   - Commercial/transactional intent keywords
3. Export and categorize gaps

**Step 3.2: Without SEO Tools (5-Step Method)**

**Step A: SERP Mining**

- Search 15-20 core topic queries
- Document all unique domains in top 20
- Create frequency matrix: which domains appear for which queries
- Domains appearing 5+ times are primary keyword competitors

**Step B: Site Operator Expansion**

For each competitor:

```
web_search("site:competitor.com [primary topic]")
```

- Document page titles and apparent target keywords
- Build competitor keyword inventory
- Cross-reference with your content

**Step C: People Also Ask Extraction**

- For each core query, expand PAA boxes fully
- Document all PAA questions (often 20-40 per query)
- Check which questions have competitor content
- Identify questions where you have no coverage

**Step D: Related Searches Mining**

- Document bottom-of-SERP related searches
- Identify clusters where competitors rank but you don't

**Step E: GSC Cross-Reference**

- Export queries with impressions but low/no clicks
- These are keywords where you appear but aren't competitive
- Cross-reference with competitor positions

**Step 3.3: Gap Classification**

| Gap Type            | Definition                                           | Priority | Typical Action             |
| ------------------- | ---------------------------------------------------- | -------- | -------------------------- |
| **Critical Gap**    | All competitors rank, you don't                      | Highest  | Create content immediately |
| **Important Gap**   | 2-3 competitors rank, you don't                      | High     | Schedule within 30 days    |
| **Opportunity Gap** | 1 competitor ranks, you don't                        | Medium   | Evaluate ROI               |
| **Intent Gap**      | Competitors have commercial intent coverage you lack | High     | Prioritize for revenue     |
| **Format Gap**      | Competitors use format you don't (video, tools)      | Varies   | Assess capability          |

**Step 3.4: Gap Prioritization with ICE Scoring**

| Gap Keyword | Gap Type | Impact (1-10) | Confidence (1-10) | Ease (1-10) | ICE Score | Priority |
| ----------- | -------- | ------------- | ----------------- | ----------- | --------- | -------- |
|             |          |               |                   |             |           |          |

**ICE Scoring:**

- **Impact:** How much will ranking improve your business?
- **Confidence:** How certain this is a real opportunity?
- **Ease:** How quickly can you create competitive content?

**Priority = Impact x Confidence x Ease / 100**

---

### STAGE 4: Content Gap Analysis

**Purpose:** Map topic coverage and identify authority gaps

**Step 4.1: Topical Authority Mapping**

For each competitor, map their content structure:

```
web_search("site:competitor.com [primary topic]")
```

**Identify:**

- Pillar pages (comprehensive guides)
- Supporting content (related subtopics)
- Hub-and-spoke structure presence
- Topic clusters and internal linking patterns

**Topical Authority Indicators:**

- 15-30 subtopics covered per core topic
- Clear pillar-to-spoke linking
- Consistent publishing within topic area
- Depth over breadth

**Step 4.2: Content Coverage Matrix**

| Topic Category | Comp A | Comp B | Comp C | You   | Gap Status |
| -------------- | ------ | ------ | ------ | ----- | ---------- |
| [Core Topic 1] | 12 pcs | 8 pcs  | 5 pcs  | 0     | Critical   |
| [Core Topic 2] | 6 pcs  | 0      | 3 pcs  | 2 pcs | Partial    |
| [Subtopic A]   |        |        |        |       |            |
| [Subtopic B]   |        |        |        |       |            |

**Gap Status Definitions:**

- **Critical:** Competitors have extensive coverage, you have none
- **Significant:** Competitors ahead by 3x+ content volume
- **Partial:** Some coverage but competitors more comprehensive
- **Competitive:** Roughly equal coverage
- **Advantage:** You lead in this area

**Step 4.3: Format Gap Analysis**

| Content Format      | Comp A | Comp B | Comp C | You | Gap? |
| ------------------- | ------ | ------ | ------ | --- | ---- |
| Long-form guides    |        |        |        |     |      |
| Video content       |        |        |        |     |      |
| Interactive tools   |        |        |        |     |      |
| Templates/downloads |        |        |        |     |      |
| Case studies        |        |        |        |     |      |
| Original research   |        |        |        |     |      |

**Step 4.4: Freshness Analysis**

| Topic Area | Competitor Last Update | Your Last Update | Freshness Gap |
| ---------- | ---------------------- | ---------------- | ------------- |
|            |                        |                  |               |

**Freshness Benchmark:** Top performers refresh evergreen content every 6-12 months.

**Step 4.5: Blue Ocean Opportunities (Double Gap)**

Find topics BOTH you AND competitors are missing:

**Discovery Methods:**

1. PAA questions without satisfying answers
2. Reddit/forum threads with unanswered questions
3. Related searches leading to weak content
4. Emerging trends not yet covered

**Blue Ocean Criteria:**

- Real pain point (multiple sources confirm demand)
- SERP shows unsatisfying results
- You can provide genuinely better value
- Aligns with your expertise

---

### STAGE 5: Backlink Gap Analysis

**Purpose:** Identify link building opportunities from competitor profiles

**Step 5.1: Backlink Profile Overview**

**With SEO Tools:**

| Metric                    | Your Site | Comp A | Comp B | Comp C |
| ------------------------- | --------- | ------ | ------ | ------ |
| Referring Domains         |           |        |        |        |
| Total Backlinks           |           |        |        |        |
| Domain Rating/Authority   |           |        |        |        |
| Link Velocity (new/month) |           |        |        |        |

**Without Premium Tools (Free Options):**

- Ahrefs Webmaster Tools (your site only, free)
- Moz Link Explorer (limited queries, free)
- Ubersuggest (limited queries, free tier)

**Step 5.2: Link Gap Identification**

Export top 500-1000 referring domains per competitor. Categorize gaps:

| Link Gap Type         | Definition                                     | Priority | Action                      |
| --------------------- | ---------------------------------------------- | -------- | --------------------------- |
| **Common Links**      | Sites linking to 2+ competitors but not you    | Highest  | Proven linkers to your type |
| **Unique High-Value** | High-authority sites linking to one competitor | High     | Differentiation opportunity |
| **Category Leaders**  | Industry publications linking to competitors   | High     | Establishes niche authority |
| **Content-Specific**  | Sites linking to specific content types        | Medium   | Informs content strategy    |

**Step 5.3: Link Type Analysis**

Categorize how competitors acquire links:

| Acquisition Type    | Comp A | Comp B | Comp C | Replicable? |
| ------------------- | ------ | ------ | ------ | ----------- |
| Editorial mentions  |        |        |        |             |
| Guest posts         |        |        |        |             |
| Resource page links |        |        |        |             |
| Digital PR/news     |        |        |        |             |
| Tool/resource links |        |        |        |             |

**Step 5.4: Linkable Asset Analysis**

What content earns competitor links?

**Common Linkable Assets:**

- Original research and data studies
- Free tools and calculators
- Comprehensive guides (10x content)
- Infographics with original data
- Industry surveys and reports
- Controversial/contrarian takes

**Document:**

| Competitor | Top Linked Asset | Link Count | Asset Type | Can You Create Similar? |
| ---------- | ---------------- | ---------- | ---------- | ----------------------- |
|            |                  |            |            |                         |

**Step 5.5: Link Opportunity Prioritization**

| Link Opportunity | Domain Authority | Relevance | Likelihood | Effort | Priority Score |
| ---------------- | ---------------- | --------- | ---------- | ------ | -------------- |
|                  |                  |           |            |        |                |

**Scoring:** (DA x Relevance x Likelihood) / Effort

---

### STAGE 6: Technical SEO Comparison

**Purpose:** Benchmark technical performance vs competitors

**Step 6.1: Core Web Vitals Comparison**

Use PageSpeed Insights for each domain:

| Metric                          | Target | Your Site | Comp A | Comp B | Comp C |
| ------------------------------- | ------ | --------- | ------ | ------ | ------ |
| LCP (Largest Contentful Paint)  | <2.5s  |           |        |        |        |
| INP (Interaction to Next Paint) | <200ms |           |        |        |        |
| CLS (Cumulative Layout Shift)   | <0.1   |           |        |        |        |
| TTFB (Time to First Byte)       | <600ms |           |        |        |        |

**Note:** INP replaced FID in March 2024.

**Step 6.2: Crawlability Comparison**

| Factor                      | Your Site | Comp A | Comp B | Comp C |
| --------------------------- | --------- | ------ | ------ | ------ |
| Indexed Pages (site: count) |           |        |        |        |
| XML Sitemap Present         |           |        |        |        |
| Robots.txt Clean            |           |        |        |        |
| Mobile-Friendly             |           |        |        |        |
| HTTPS                       |           |        |        |        |

**Step 6.3: Schema Implementation**

Check via Rich Results Test:

| Schema Type             | Your Site | Comp A | Comp B | Comp C |
| ----------------------- | --------- | ------ | ------ | ------ |
| Organization            |           |        |        |        |
| Article/BlogPosting     |           |        |        |        |
| FAQ                     |           |        |        |        |
| HowTo                   |           |        |        |        |
| Product (if applicable) |           |        |        |        |
| Breadcrumb              |           |        |        |        |

**Step 6.4: Technical Scorecard**

Rate each domain 1-5:

| Dimension             | Your Site | Comp A | Comp B | Comp C |
| --------------------- | --------- | ------ | ------ | ------ |
| Page Speed            |           |        |        |        |
| Core Web Vitals       |           |        |        |        |
| Mobile Experience     |           |        |        |        |
| Schema Richness       |           |        |        |        |
| Site Architecture     |           |        |        |        |
| Security/HTTPS        |           |        |        |        |
| **Technical Average** |           |        |        |        |

---

### STAGE 7: Competitive Strategy Development

**Purpose:** Translate analysis into actionable strategy

**Step 7.1: RADAR Response Framework**

When competitive threats are detected, use RADAR:

**R - Recognize** the threat

- Ranking drop detected?
- Competitor entered top 10?
- SOV declining?
- SERP feature lost?

**A - Analyze** the cause

- What specific action did competitor take?
- Content improvement? New content? Links? Technical upgrade?
- One-time move or sustained campaign?

**D - Determine** priority (ICE Score)

- Impact: How much does this affect your business?
- Confidence: How certain are you of cause and solution?
- Ease: How quickly can you respond effectively?

**A - Act** with appropriate response

| Competitor Move           | Response Options                                      |
| ------------------------- | ----------------------------------------------------- |
| New comprehensive content | Improve existing OR create differentiated alternative |
| Link building campaign    | Identify link sources, pursue same opportunities      |
| Technical improvements    | Match technical performance, compete on content       |
| Featured snippet capture  | Optimize your content for snippet format              |
| New topic area entry      | Evaluate strategic fit, consider early defense        |

**R - Review** results

- Set 30/60/90 day checkpoints
- Track ranking recovery
- Document what worked

**Step 7.2: Offensive vs Defensive Balance**

**Recommended Allocation: 60% Offensive / 40% Defensive**

**Offensive Actions:**

- Close critical content gaps
- Build new topical authority areas
- Pursue link gap opportunities
- Capture unclaimed SERP features

**Defensive Actions:**

- Content freshness maintenance
- Backlink profile monitoring
- Technical health vigilance
- Brand mention monitoring

**Step 7.3: ICE-Prioritized Action Plan**

Rate each opportunity:

| Action | Impact (1-10) | Confidence (1-10) | Ease (1-10) | ICE Score | Timeline |
| ------ | ------------- | ----------------- | ----------- | --------- | -------- |
|        |               |                   |             |           |          |

**Tier 1: Quick Wins (This Week)** - ICE > 50, High Ease

- [ ] [Action]
- [ ] [Action]

**Tier 2: Short-Term Moves (This Month)** - ICE 30-50

- [ ] [Action]
- [ ] [Action]

**Tier 3: Strategic Investments (This Quarter)** - High Impact, Lower Ease

- [ ] [Action]
- [ ] [Action]

**Step 7.4: Content Calendar Integration**

Based on competitive gaps, add to content calendar:

| Content Piece | Competitive Rationale | Target Keyword | Priority | Target Date |
| ------------- | --------------------- | -------------- | -------- | ----------- |
|               |                       |                |          |             |

---

### STAGE 8: Ongoing Monitoring Setup

**Purpose:** Establish systematic competitive tracking

**Step 8.1: Monitoring Cadence Framework**

| Frequency     | Activities                                                 | Time      | Triggers for Escalation  |
| ------------- | ---------------------------------------------------------- | --------- | ------------------------ |
| **Daily**     | Top 10-20 keyword rankings, alerts                         | 5-10 min  | >5 position drop         |
| **Weekly**    | Top 50-100 keywords, competitor content, SERP features     | 30-60 min | Competitor enters top 10 |
| **Monthly**   | Full ranking report, SOV calculation, backlinks, technical | 2-4 hours | SOV drop >5%             |
| **Quarterly** | Complete re-analysis (full Workflow X), strategy review    | 1-2 days  | Major competitive shifts |

**Step 8.2: Tool Recommendations by Budget**

**Free Tier:**

| Tool                  | Purpose                   |
| --------------------- | ------------------------- |
| Google Search Console | Own site performance      |
| Google Alerts         | Competitor brand mentions |
| PageSpeed Insights    | Technical comparison      |
| HubSpot AI SOV Tool   | AI visibility tracking    |

**Budget Tier ($30-70/month):**

| Tool                   | Purpose                   | Price     |
| ---------------------- | ------------------------- | --------- |
| SE Ranking Essential   | Full competitive analysis | $52-65/mo |
| Ubersuggest            | Keyword/backlink gaps     | $29-52/mo |
| SERPWatcher (Mangools) | Rank tracking with alerts | $29.90/mo |

**Professional Tier ($100-200/month):**

| Tool           | Purpose                      | Price      |
| -------------- | ---------------------------- | ---------- |
| SE Ranking Pro | AI tracking, historical data | $119/mo    |
| Semrush Pro    | Full marketing suite         | $129.95/mo |
| Ahrefs Lite    | Backlink focus               | $129/mo    |

**Step 8.3: Alert Configuration**

Set up alerts for:

| Alert Type         | Threshold                               | Tool/Method        |
| ------------------ | --------------------------------------- | ------------------ |
| Ranking drop       | >5 positions on priority keyword        | Rank tracker       |
| Competitor content | New content on your topics              | Google Alerts      |
| Backlink change    | Sudden spike (potential negative SEO)   | Ahrefs/Semrush     |
| SERP feature loss  | Featured snippet captured by competitor | Manual check       |
| New competitor     | Unknown domain enters top 10            | Weekly SERP review |

**Step 8.4: Monitoring Templates**

**Weekly Check Template:**

```markdown
## Weekly Competitive Check - [Date]

### Ranking Changes

- [Keyword]: [Your Position] -> [New Position]

### Competitor Activity

- [Competitor]: [New content/action observed]

### SERP Feature Changes

- [Feature]: [Previous owner] -> [New owner]

### Actions Needed

- [ ] [Action item]
```

**Monthly Report Template:**

```markdown
## Monthly Competitive Report - [Month]

### Share of Voice

- Your SOV: [X%] (Change: [+/-]%)
- Comp A: [X%]
- Comp B: [X%]

### Key Movements

- [Summary of significant changes]

### Gap Closure Progress

- [Gaps addressed this month]

### Next Month Priorities

- [ ] [Priority 1]
- [ ] [Priority 2]
```

---

## Output Format: Competitive Intelligence Report

```markdown
# Competitive Intelligence Report: [Your Domain]

**Analysis Date:** [date]
**Primary Topic/Niche:** [topic]
**Competitors Analyzed:** [count]

---

## Executive Summary

**Overall Position:** [Leader / Strong Competitor / Emerging / Behind]
**Share of Voice:** [X%]
**AI Visibility:** [Strong / Moderate / Weak]

**Key Findings:**

- [Finding 1]
- [Finding 2]
- [Finding 3]

**Priority Actions:**

1. [Action 1]
2. [Action 2]
3. [Action 3]

---

## Part 1: Competitor Landscape

### Share of Voice

| Domain | SOV | AI Citations | Trend |
| ------ | --- | ------------ | ----- |
| You    |     |              |       |
| Comp A |     |              |       |
| Comp B |     |              |       |
| Comp C |     |              |       |

### Competitive Scorecard

[Full scorecard from Stage 2]

---

## Part 2: Gap Analysis

### Keyword Gaps (Top 10)

| Keyword | Gap Type | ICE Score | Action |
| ------- | -------- | --------- | ------ |
|         |          |           |        |

### Content Gaps

[Coverage matrix from Stage 4]

### Backlink Gaps

[Link opportunities from Stage 5]

### Technical Gaps

[Scorecard from Stage 6]

---

## Part 3: Strategic Recommendations

### Offensive Priorities (60%)

1. [Priority 1]
2. [Priority 2]
3. [Priority 3]

### Defensive Priorities (40%)

1. [Defense 1]
2. [Defense 2]

### Action Plan

**This Week:** [Quick wins]
**This Month:** [Short-term moves]
**This Quarter:** [Strategic investments]

---

## Part 4: Monitoring Plan

### Cadence

| Check         | Frequency | Owner |
| ------------- | --------- | ----- |
| Ranking       | Daily     |       |
| Content       | Weekly    |       |
| SOV           | Monthly   |       |
| Full Analysis | Quarterly |       |

### Alert Triggers

- [Trigger 1]: [Response]
- [Trigger 2]: [Response]

---

## Competitor Profiles

### [Competitor A]

**Domain:** [domain]
**Type:** [Direct/Partial/Authority/Emerging]
**Threat Level:** [High/Medium/Low]

**Strengths:**

- [Strength 1]
- [Strength 2]

**Weaknesses:**

- [Weakness 1]
- [Weakness 2]

**Content Strategy:** [Summary]
**Link Strategy:** [Summary]

[Repeat for each competitor]

---

_Generated by Workflow X: Competitive Intelligence_
_Analysis Date: [date]_
_Next Review: [date + 3 months]_
```

---

## Integration with Other Workflows

### Feeding Workflow A (Audit)

Workflow X provides competitor context:

- Use Competitor Watch List in Stage 1.2
- Use Competitive Scorecard in Stage 2.2
- Use Content Coverage Matrix in Stage 3.1

### Feeding Workflow R (Rank)

Workflow X informs page-level optimization:

- Which competitor pages to benchmark
- What content depth required to compete
- Which SERP features to target

### Feeding Workflow S (Strategy)

Workflow X competitive gaps inform content prioritization:

- Gap keywords become priority targets
- Competitor velocity sets publishing benchmarks

### Feeding Prioritization Framework

Workflow X outputs provide:

- ICE scores for opportunity ranking
- Urgency indicators from competitor movements
- Resource allocation guidance (offensive/defensive)

---

## Execution Checklist

Before generating competitive intelligence report:

- [ ] Website URL and topic/niche captured
- [ ] 7-10 core topic searches executed
- [ ] Competitors classified (Direct/Partial/Authority/Emerging)
- [ ] Share of Voice calculated
- [ ] AI visibility assessed
- [ ] Content velocity compared
- [ ] Competitive scorecard completed
- [ ] Keyword gaps identified with ICE scoring
- [ ] Content coverage matrix created
- [ ] Topical authority mapped
- [ ] Backlink gap analysis completed
- [ ] Link opportunities prioritized
- [ ] Technical comparison completed
- [ ] RADAR response framework applied
- [ ] Action plan prioritized (offensive/defensive)
- [ ] Monitoring cadence established
- [ ] Alert thresholds configured
- [ ] Competitor profiles created

---

## Summary

Workflow X is the **dedicated competitive intelligence workflow** for systematic competitor analysis.

**Key Metrics:**

- Share of Voice as primary competitive measure
- AI visibility as emerging KPI
- Content velocity for authority building
- ICE scoring for prioritization

**Key Frameworks:**

- Hierarchical competitor classification (Direct/Partial/Authority/Emerging)
- Three-tier gap analysis (Keywords, Content, Authority)
- RADAR response model (Recognize, Analyze, Determine, Act, Review)
- Four-tier monitoring cadence (Daily/Weekly/Monthly/Quarterly)

**Execution Time:** 30-60 minutes depending on tool access and depth

**Key Insight:**

> "You're not ranking in a vacuum. Ranking is actually about outranking your search competitors." - Lidia Infante

---

## Version History

| Version | Date     | Changes                                                                                                                                                                 |
| ------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| v1      | Jan 2026 | Initial extraction from existing workflows. Research gaps identified.                                                                                                   |
| v2      | Jan 2026 | Complete methodology: SOV calculation, AI visibility, backlink gap analysis, technical benchmarking, RADAR response framework, ICE prioritization, monitoring cadences. |
