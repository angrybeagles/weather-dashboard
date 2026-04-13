# SEO Prioritization Framework

- **Purpose:** Strategic decision layer for determining what SEO work to prioritize
- **Application:** Governs resource allocation across all SEO workflows
- **Output:** Prioritized roadmap with clear execution order

---

## Why Prioritization Matters

> "Prioritization is the apex artifact bridging strategy and execution... In practice, however, prioritization models are often used as cursory tools to get your boss or client off your back." — Alex Birkett, Omniscient Digital

Most SEO prioritization fails because it:

- Dumps all keywords into one spreadsheet with uniform scoring
- Ignores business context and strategic priorities
- Treats all "errors" as equally urgent
- Produces generic recommendations

This framework solves that with **hierarchical prioritization** — business-first, then tactics.

---

## Core Principle: Hierarchical Clustering

**The Problem with Flat Prioritization:**

Traditional approach:

```
All Keywords → Score by (Impact × Ease × Relevance) → Sort → Execute
```

This produces mediocre results because every business is unique. A $5M SaaS company and a local bakery shouldn't use identical prioritization formulas.

**Hierarchical Approach:**

```
Business Priorities → Audience Segments → Funnel Stages → Content Types → Individual Keywords
```

Prioritize higher-order dimensions FIRST, then sort within those constraints.

---

## Framework 1: Hierarchical Dimensions

Before any keyword-level decisions, determine priority at these levels:

### Dimension 1: Business Unit / Product Line

**Questions to Answer:**

- Which products are most strategic right now?
- Where does leadership want growth?
- What has highest LTV or margin?
- What's the competitive landscape per product?

**Example Prioritization:**
| Product | Revenue Contribution | Strategic Priority | SEO Priority |
|---------|---------------------|-------------------|--------------|
| Product A | 60% | Maintain | Medium |
| Product B | 30% | Grow aggressively | High |
| Product C | 10% | New launch | High |

### Dimension 2: Customer Segment / Persona

**Questions to Answer:**

- Which audience is most valuable (LTV)?
- Which segment is underserved by competitors?
- Where is competitive opportunity greatest?
- Which segment converts best from organic?

**Example Prioritization:**
| Segment | LTV | Competition | Conversion Rate | SEO Priority |
|---------|-----|-------------|-----------------|--------------|
| Enterprise | $50K | High | 2% | Medium |
| Mid-Market | $15K | Medium | 5% | High |
| SMB | $2K | Low | 8% | High |

### Dimension 3: Funnel Stage

**Questions to Answer:**

- Where do we need more pipeline? (TOFU)
- Where are we losing people? (MOFU)
- Where can we capture ready buyers? (BOFU)

| Stage | Description   | Typical Content               | When to Prioritize          |
| ----- | ------------- | ----------------------------- | --------------------------- |
| TOFU  | Awareness     | Educational, problem-focused  | Building future pipeline    |
| MOFU  | Consideration | Comparison, evaluation        | Nurturing existing interest |
| BOFU  | Decision      | Pricing, features, conversion | Capturing ready buyers      |

**Default Priority:** BOFU → MOFU → TOFU (capture demand before building awareness)

**Exception:** New market/product launch → TOFU first to create awareness

### Dimension 4: Content Type

| Type          | Purpose               | Priority When                      |
| ------------- | --------------------- | ---------------------------------- |
| Transactional | Direct conversion     | High-intent keywords available     |
| Commercial    | Evaluation/comparison | Competitive differentiation needed |
| Informational | Education/authority   | Building topical authority         |
| Navigational  | Brand capture         | Protecting brand terms             |

---

## Framework 2: Eisenhower Matrix for SEO

Categorize all SEO tasks by Urgency × Importance:

```
                    URGENT                 NOT URGENT
                ┌─────────────────────┬─────────────────────┐
                │                     │                     │
    IMPORTANT   │     DO FIRST        │      SCHEDULE       │
                │                     │                     │
                │  • Critical tech    │  • Content strategy │
                │    issues blocking  │  • Topic authority  │
                │    indexing         │    building         │
                │  • Security vulns   │  • Evergreen        │
                │  • CWV failures     │    content          │
                │  • Algorithm        │  • Link building    │
                │    penalties        │  • Site architecture│
                │                     │                     │
                ├─────────────────────┼─────────────────────┤
                │                     │                     │
  NOT IMPORTANT │   DELEGATE/QUICK    │     ELIMINATE       │
                │                     │                     │
                │  • Routine          │  • Vanity metrics   │
                │    reporting        │  • Low-value        │
                │  • Minor tech       │    keywords         │
                │    fixes            │  • Aesthetic        │
                │  • Low-priority     │    changes w/o      │
                │    content updates  │    SEO impact       │
                │                     │  • Over-optimizing  │
                │                     │    successful pages │
                │                     │                     │
                └─────────────────────┴─────────────────────┘
```

### Quadrant Definitions

**Q1: Urgent + Important (DO FIRST)**

- Critical technical issues blocking indexing
- Security vulnerabilities affecting trust
- Core Web Vitals failures impacting rankings
- Responding to algorithm penalties
- Misconfigured robots.txt blocking key pages
- Canonical issues causing duplicate content penalties

**Q2: Not Urgent + Important (SCHEDULE)**

- Long-term content strategy development
- Topic authority building (hub/spoke)
- Evergreen content creation
- Link building campaigns
- Site architecture improvements
- E-E-A-T signal building

**Q3: Urgent + Not Important (DELEGATE/QUICK)**

- Routine weekly/monthly reporting
- Minor technical fixes (non-blocking)
- Low-priority content date updates
- Social signals maintenance
- Small metadata tweaks

**Q4: Not Urgent + Not Important (ELIMINATE)**

- Vanity metric optimization (DA chasing)
- Low-KD keywords with no business value
- Aesthetic changes without SEO impact
- Over-optimizing pages already ranking #1
- "SEO theater" — activities that look like SEO but don't move metrics

---

## Framework 3: Impact × Effort Scoring

For tactical prioritization within strategic constraints:

### Scoring Dimensions

**Impact Score (1-5):**

| Score | Traffic Potential | Business Value  | Description                           |
| ----- | ----------------- | --------------- | ------------------------------------- |
| 5     | 10K+ monthly      | Direct revenue  | High-volume, high-converting keywords |
| 4     | 1K-10K monthly    | Strong lead gen | Good volume, clear business relevance |
| 3     | 100-1K monthly    | Moderate value  | Decent volume, supports authority     |
| 2     | <100 monthly      | Indirect value  | Low volume but strategic              |
| 1     | Minimal           | Marginal        | Little expected impact                |

**Effort Score (1-5):**

| Score | Time Required | Resources | Description                      |
| ----- | ------------- | --------- | -------------------------------- |
| 5     | <1 day        | Solo      | Quick optimization, minor tweaks |
| 4     | 1-2 days      | Solo      | Small content update             |
| 3     | 3-5 days      | Solo/pair | Moderate content refresh         |
| 2     | 1-2 weeks     | Team      | New content piece                |
| 1     | 2+ weeks      | Team      | Pillar content, major project    |

**Confidence Score (1-5):**

| Score | Certainty | Evidence                                |
| ----- | --------- | --------------------------------------- |
| 5     | Very high | Proven tactic, clear data               |
| 4     | High      | Strong indicators, similar past success |
| 3     | Medium    | Reasonable expectation                  |
| 2     | Low       | Experimental, uncertain outcome         |
| 1     | Very low  | Speculative, no precedent               |

### Priority Formula

```
Priority Score = (Impact + Effort + Confidence) / 3
```

Or weighted version:

```
Priority Score = (Impact × 0.4) + (Effort × 0.35) + (Confidence × 0.25)
```

### Priority Tiers

| Tier   | Score Range | Action               | Typical Tasks                       |
| ------ | ----------- | -------------------- | ----------------------------------- |
| Tier 1 | 4.0-5.0     | Execute immediately  | Quick wins, striking distance pages |
| Tier 2 | 3.0-3.9     | Schedule this sprint | Content improvements, gap filling   |
| Tier 3 | 2.0-2.9     | Plan for future      | New content, link building          |
| Tier 4 | 1.0-1.9     | Deprioritize         | Low-value, high-effort tasks        |

---

## Framework 4: The 70/30 Rule

### Resource Allocation by Work Type

**Initial Phase (Months 1-6):**

```
Quick Wins & Maintenance: 70%
Long-Term Strategic: 30%
```

**Mature Phase (Month 6+):**

```
Quick Wins & Maintenance: 50%
Long-Term Strategic: 50%
```

### Quick Wins (70% initially)

Definition: Tasks that are simple to implement and produce noticeable results

**Examples:**

- Title tag optimization on striking distance pages
- Internal linking improvements
- Meta description updates for CTR
- Fixing broken links
- Adding schema markup
- Updating dates/statistics on performing content

**Characteristics:**

- Completed in 1-5 days
- Measurable movement in 2-4 weeks
- Doesn't require major resources
- Builds momentum and stakeholder confidence

### Long-Term Investments (30% initially)

Definition: Strategic projects requiring significant investment but building lasting advantage

**Examples:**

- New content pillar development
- Topic cluster buildout
- Site architecture restructuring
- E-E-A-T authority building
- Comprehensive link building campaigns
- Technical infrastructure improvements

**Characteristics:**

- Takes 2-6 months to implement
- Results appear over 6-18 months
- Requires dedicated resources
- Creates competitive moats

### Content Budget Split

> "One effective approach is the 70/30 rule: allocate 70% of your content budget for new creation, 30% for refreshing existing content." — Kate Kandefer, SEOwind

**Adjustment Factors:**

| Site State             | New Content | Refresh | Rationale                          |
| ---------------------- | ----------- | ------- | ---------------------------------- |
| New site (<50 pages)   | 90%         | 10%     | Need content foundation            |
| Growing (50-500 pages) | 70%         | 30%     | Balance growth + optimization      |
| Mature (500+ pages)    | 50%         | 50%     | Significant optimization potential |
| Post-update recovery   | 30%         | 70%     | Focus on fixing what exists        |

---

## Resource Allocation Models

### Model 1: By Site Maturity

**New Site (<50 pages, <1K monthly organic visits):**

| Activity             | Allocation |
| -------------------- | ---------- |
| Technical foundation | 20%        |
| Keyword research     | 20%        |
| New content creation | 40%        |
| On-page optimization | 15%        |
| Link building        | 5%         |

**Growing Site (50-500 pages, 1K-50K monthly):**

| Activity              | Allocation |
| --------------------- | ---------- |
| Technical maintenance | 10%        |
| Content gap analysis  | 10%        |
| New content creation  | 30%        |
| Content optimization  | 25%        |
| Internal linking      | 10%        |
| Link building         | 15%        |

**Mature Site (500+ pages, 50K+ monthly):**

| Activity              | Allocation |
| --------------------- | ---------- |
| Technical maintenance | 10%        |
| Content audit/pruning | 15%        |
| Content refresh       | 30%        |
| New strategic content | 20%        |
| Internal linking      | 10%        |
| Link building         | 15%        |

### Model 2: By Team Size

**Solo Practitioner (Weekly):**

| Day      | Focus                                | Allocation |
| -------- | ------------------------------------ | ---------- |
| Monday   | Analysis & planning                  | 20%        |
| Tue-Wed  | Content creation/optimization        | 40%        |
| Thursday | Technical & on-page                  | 25%        |
| Friday   | Reporting, learning, experimentation | 15%        |

**Small Team (3-5 people):**

| Role          | Focus                              | Allocation   |
| ------------- | ---------------------------------- | ------------ |
| Strategist    | Planning, analysis, prioritization | 100% of role |
| Content Lead  | Content creation, optimization     | 100% of role |
| Technical SEO | Infrastructure, performance        | 50% of role  |
| Link Builder  | Outreach, relationships            | 50% of role  |

### Model 3: Budget Allocation (External Spend)

| Category      | Allocation | Examples                        |
| ------------- | ---------- | ------------------------------- |
| Content       | 40-50%     | Writers, editors, designers     |
| Technical     | 15-20%     | Developer time, tools           |
| Link building | 20-25%     | Outreach, PR, digital PR        |
| Tools & Data  | 10-15%     | Ahrefs, Semrush, Screaming Frog |
| Strategy      | 5-10%      | Planning, coordination          |

---

## Decision Trees

### Decision Tree 1: New Content vs. Optimization

```
START: Do we have existing content on this topic?
│
├─ NO → Is this a priority topic (from hierarchical analysis)?
│        ├─ YES → CREATE NEW (Workflows K-S-C)
│        └─ NO → DEPRIORITIZE
│
└─ YES → Is existing content ranking?
          │
          ├─ NO (position 50+) → Is the content quality good?
          │                       ├─ YES → Technical/link issue → FIX TECHNICAL
          │                       └─ NO → MAJOR REWRITE or NEW PAGE
          │
          └─ YES → What position?
                    │
                    ├─ 1-3 → MAINTAIN (minor updates only)
                    ├─ 4-10 → OPTIMIZE (Workflow R - quick wins)
                    ├─ 11-20 → OPTIMIZE (Workflow R - priority)
                    └─ 21-50 → EVALUATE (optimize vs. new page)
```

### Decision Tree 2: Task Type Routing

```
START: What type of opportunity is this?
│
├─ Existing page needs improvement
│   └─ → Workflow R: Rank
│
├─ Content gap (competitor has, we don't)
│   └─ → SEO Boost API (research) → Workflow S (organize) → Workflow C (create)
│
├─ Blue ocean opportunity (no one covers well)
│   └─ → Validate demand → API-S-C
│
├─ Technical issue
│   └─ → Technical SEO checklist (outside workflow system)
│
├─ Link building need
│   └─ → Link building strategy (outside workflow system)
│
└─ Content consolidation
    └─ → Merge content → Workflow R on merged page
```

### Decision Tree 3: Quarterly Planning

```
START: Beginning of quarter planning
│
├─ STEP 1: Review business priorities
│   └─ What products/segments matter most this quarter?
│
├─ STEP 2: Run Workflow A (Audit)
│   └─ Update positioning, gaps, inventory
│
├─ STEP 3: Apply Eisenhower Matrix
│   └─ Categorize all opportunities by Urgent × Important
│
├─ STEP 4: Score Tier 1-2 opportunities
│   └─ Impact × Effort × Confidence
│
├─ STEP 5: Allocate resources
│   └─ Apply 70/30 or 50/50 based on maturity
│
└─ STEP 6: Create sprint plan
    └─ Week-by-week execution schedule
```

---

## Quarterly Planning Template

### Step 1: Business Priority Check

**Questions to Answer:**

1. What are the company's top 3 priorities this quarter?
2. Which products/services need SEO support?
3. Are there any launches, pivots, or sunsets?
4. What does success look like for SEO this quarter?

**Output:**
| Priority | Description | SEO Implication |
|----------|-------------|-----------------|
| 1 | | |
| 2 | | |
| 3 | | |

### Step 2: Current State Assessment

Run Workflow A or quick audit to update:

- Position distribution (1-3, 4-10, 11-20, 21-50, not ranking)
- Content inventory status
- Top opportunities identified
- Blockers or technical issues

### Step 3: Opportunity Scoring

Score all identified opportunities:

| Opportunity | Type | Impact | Effort | Confidence | Score | Tier |
| ----------- | ---- | ------ | ------ | ---------- | ----- | ---- |
|             |      |        |        |            |       |      |

### Step 4: Resource Allocation

Based on site maturity and team size:

| Activity             | Hours/Week | % of Total |
| -------------------- | ---------- | ---------- |
| New content creation |            |            |
| Content optimization |            |            |
| Technical SEO        |            |            |
| Link building        |            |            |
| Analysis & planning  |            |            |
| **Total**            |            | 100%       |

### Step 5: Sprint Planning

**Month 1:**

- Week 1-2: [Tier 1 quick wins]
- Week 3-4: [Tier 1-2 content work]

**Month 2:**

- Week 1-2: [Tier 2 priorities]
- Week 3-4: [New content from Workflow C]

**Month 3:**

- Week 1-2: [Continue Tier 2]
- Week 3-4: [Long-term strategic work]

---

## Common Pitfalls to Avoid

### Pitfall 1: Fixing Every Audit Error

> "If you're using an SEO audit tool, it will score and direct you to do all types of tasks. If you don't filter these tasks through this matrix, you'll end up spending a large amount of time on projects that garner little results." — Jason Montoya

**Solution:** Use prioritization framework to filter. Not every error matters equally.

### Pitfall 2: Only Chasing Quick Wins

> "Focusing solely on long-term projects can delay results, while chasing only quick wins can leave important strategic opportunities on the table." — Xponent21

**Solution:** Maintain the 70/30 or 50/50 balance. Schedule long-term work even when quick wins are tempting.

### Pitfall 3: Ignoring Business Context

> "SEO isn't a standalone strategy—it must align with your broader business objectives." — Xponent21

**Solution:** Start every prioritization cycle with business priority check-in.

### Pitfall 4: Static Priorities

Priorities established in January may not apply in July due to:

- Algorithm updates
- Competitive changes
- Business pivots
- Seasonal shifts

**Solution:** Quarterly full reprioritization. Monthly check-ins for adjustments.

### Pitfall 5: Analysis Paralysis

Spending too much time on perfect prioritization instead of doing work.

**Solution:** Time-box prioritization to 2-4 hours per quarter. Use 80% confidence decisions and iterate based on results.

---

## Integration with Workflow System

### How Prioritization Connects to Workflows

```
┌─────────────────────────────────────────────────────────────────┐
│                  SEO PRIORITIZATION FRAMEWORK                    │
│                                                                 │
│  Business Priorities → Hierarchical Dimensions → Scoring        │
│                                                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────┴────────────────┐
         │                                 │
         ▼                                 ▼
┌─────────────────┐               ┌─────────────────┐
│   NEW CONTENT   │               │ EXISTING CONTENT│
│   OPPORTUNITIES │               │  OPTIMIZATION   │
└────────┬────────┘               └────────┬────────┘
         │                                 │
         ▼                                 ▼
┌─────────────────┐               ┌─────────────────┐
│   Workflow A    │               │   Workflow R    │
│   SEO Audit     │               │ Rank Enhancement│
└────────┬────────┘               └─────────────────┘
         │
         ▼
┌─────────────────┐
│  SEO Boost API  │
│Keyword Discovery│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Workflow S    │
│Content Strategy │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Workflow C    │
│Content Generation│
└─────────────────┘
```

### When to Apply This Framework

| Situation                   | Apply Framework                  |
| --------------------------- | -------------------------------- |
| Starting new SEO engagement | Full hierarchical analysis       |
| Quarterly planning          | Complete prioritization cycle    |
| Monthly check-in            | Quick Eisenhower review          |
| After Workflow A audit      | Score and tier all opportunities |
| Resource constraint         | Rebalance allocations            |
| Business pivot              | Re-run hierarchical dimensions   |

---

## Summary

The SEO Prioritization Framework provides:

✅ **Hierarchical Clustering** — Business-first, then tactics  
✅ **Eisenhower Matrix** — Urgent × Important categorization  
✅ **Impact × Effort Scoring** — Quantified prioritization  
✅ **70/30 Rule** — Resource allocation guidance  
✅ **Decision Trees** — Clear routing for different situations  
✅ **Quarterly Planning Template** — Structured planning process  
✅ **Pitfall Avoidance** — Common mistakes to avoid

**Key Insight:**

> "This framework requires a lot more critical thinking than your typical prioritization model, which is a good thing. We've spent too many years as an industry using the same keyword research tools, content brief tools, and simplistic sorting models, which has resulted in a sea of sameness and impenetrable competition." — Alex Birkett

Prioritization isn't about scoring keywords. It's about aligning SEO with business reality.
