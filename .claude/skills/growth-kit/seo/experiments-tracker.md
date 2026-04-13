# SEO Experiments Tracker Guide

- **Purpose:** Systematic methodology for tracking SEO changes and measuring their impact
- **Use Case:** Documenting tests, validating hypotheses, proving ROI

---

## Why Track SEO Experiments

> "Gone are the days of the stereotypical SEO answer, 'It depends...' Today, we embrace a new era of 'Our tests showed...'" — Stephanie Wallace, Search Engine Land

**Benefits of Systematic Tracking:**

- Prove what actually works (vs. assumptions)
- Justify further investment with data
- Prevent negative changes from scaling
- Build institutional knowledge
- Avoid repeating failed experiments

**Without Tracking:**

- Can't attribute changes to specific actions
- Algorithm updates confuse results
- No learning accumulates
- Decisions remain opinion-based

---

## The Experiment Framework

### 7-Step Process

```
1. IDEATE → Create hypothesis
2. GROUP → Define control and variant sets
3. DEFINE → Set methodology, duration, success criteria
4. MONITOR → Set up tracking dashboard
5. IMPLEMENT → Make changes to variant only
6. ANALYZE → Wait for statistical significance
7. DETERMINE → Roll out, iterate, or abandon
```

---

## Step 1: Hypothesis Formation

### IF + THEN + BECAUSE Format

Every experiment needs a testable hypothesis:

```
IF [we make this specific change]
THEN [this metric will improve by X%]
BECAUSE [this is our reasoning/evidence]
```

**Good Hypothesis Examples:**

```
IF we add target keyword to the first 3 words of title tags
THEN CTR will increase by 15%+
BECAUSE SERP analysis shows top performers front-load keywords

IF we expand product pages from 300 to 800 words
THEN rankings will improve by 3+ positions
BECAUSE competitor analysis shows top 3 average 750+ words

IF we add FAQ schema to guide posts
THEN rich result appearances will increase by 50%
BECAUSE FAQ schema has high implementation success rate
```

**Bad Hypothesis Examples:**

- "Let's try adding more content" (no specific change)
- "This should help rankings" (no measurable outcome)
- "Everyone says this works" (no reasoning)

---

## Step 2: Control & Variant Groups

### Requirements for Valid Testing

1. **Sufficient traffic:** Minimum 100+ sessions/month per page tested
2. **Page similarity:** Templatized or structurally similar pages
3. **Even distribution:** Traffic reasonably distributed across test pages
4. **Random assignment:** Pages randomly assigned to groups
5. **No overlap:** Changes only to variant group

### Group Setup

**Control Group:** Pages that remain unchanged (baseline)
**Variant Group:** Pages receiving the experimental change

**Minimum Group Sizes:**

- Small test: 5+ pages per group
- Medium test: 10-20 pages per group
- Large test: 50+ pages per group (for statistical confidence)

**Assignment Methods:**

- Random selection from similar page pool
- Alphabetical split
- Every-other assignment
- Category-based split (if categories are balanced)

---

## Step 3: Methodology Definition

### Document Before Starting

```markdown
## Experiment: [Name]

### Hypothesis

IF: [specific change]
THEN: [expected outcome]
BECAUSE: [reasoning]

### Change Details

**What exactly changes:**

- [element 1]
- [element 2]

**What stays the same:**

- [element 1]
- [element 2]

### Groups

**Control (unchanged):**

- [URL 1]
- [URL 2]
- [URL 3]

**Variant (changed):**

- [URL 1]
- [URL 2]
- [URL 3]

### Success Criteria

**Primary metric:** [metric] improves by [X%]
**Secondary metric:** [metric] does not decline by more than [Y%]

### Duration

**Minimum runtime:** [X weeks]
**Decision date:** [date]

### External Factors

**Known risks:**

- [seasonal variation?]
- [algorithm update window?]
- [competitor activity?]
```

---

## Step 4: Monitoring Setup

### Key Metrics to Track

| Metric             | Source             | Frequency    | Notes                   |
| ------------------ | ------------------ | ------------ | ----------------------- |
| Position (average) | GSC                | Daily/Weekly | Per keyword + aggregate |
| Impressions        | GSC                | Daily        | Early signal            |
| Clicks             | GSC                | Daily        | Primary metric          |
| CTR                | GSC                | Weekly       | For meta experiments    |
| Organic sessions   | GA4                | Daily        | Traffic verification    |
| Conversions        | GA4                | Weekly       | Business impact         |
| Page speed         | PageSpeed Insights | Pre/post     | Technical changes       |

### Dashboard Setup

**Google Looker Studio Template:**

1. Connect GSC data source
2. Create filter for control URLs
3. Create filter for variant URLs
4. Build comparison charts:
   - Clicks over time (control vs. variant)
   - Impressions over time
   - CTR over time
   - Position over time

**Simple Spreadsheet Alternative:**

| Date | Control Clicks | Variant Clicks | Control CTR | Variant CTR | Notes |
| ---- | -------------- | -------------- | ----------- | ----------- | ----- |
|      |                |                |             |             |       |

### External Factor Monitoring

Track these alongside your experiment:

- **Algorithm updates:** Semrush Sensor, Mozcast, AWR
- **Competitor changes:** Weekly SERP spot checks
- **Seasonal patterns:** Year-over-year comparison
- **Site-wide issues:** Overall traffic trend

---

## Step 5: Implementation

### Change Execution Checklist

```
□ Baseline data captured (7+ days before change)
□ Control group verified unchanged
□ Variant changes implemented correctly
□ All variant pages verified
□ Implementation date recorded
□ Dashboard tracking confirmed
□ Calendar reminder set for analysis dates
```

### Documentation Requirements

**For each change, record:**

- Exact change made (before/after examples)
- Implementation date and time
- Who made the change
- Any issues during implementation
- Screenshot/archive of before state

---

## Step 6: Analysis

### Timeline Guidelines

| Test Type         | Minimum Duration | Reasoning                     |
| ----------------- | ---------------- | ----------------------------- |
| Title tag changes | 2-4 weeks        | CTR changes appear faster     |
| Meta descriptions | 2-4 weeks        | CTR-focused                   |
| Content expansion | 8-12 weeks       | Full indexing + ranking shift |
| Internal linking  | 6-8 weeks        | Authority flow takes time     |
| Full page refresh | 8-16 weeks       | Multiple signals compound     |
| Schema markup     | 2-4 weeks        | Rich result appearance        |
| Technical changes | 4-8 weeks        | Crawl + index + ranking       |

### Analysis Framework

**Week 2 Check (Early Signal):**

- Is data collecting correctly?
- Any obvious errors to fix?
- Early directional trends?

**Week 4-6 Analysis:**

- Compare aggregate metrics
- Calculate percentage change
- Note any external factors

**Final Analysis:**

- Statistical significance assessment
- Confounding factor review
- Business impact calculation

### Significance Determination

**Simple Approach (Small Tests):**

- 20%+ improvement = Likely significant
- 10-20% improvement = Promising, extend test
- <10% improvement = Inconclusive
- Decline = Investigate immediately

**Statistical Approach (Large Tests):**

- Use Causal Impact model (Google's methodology)
- Tools: SearchPilot, SplitSignal, or manual calculation
- Target: 95% confidence interval

---

## Step 7: Decision & Documentation

### Decision Framework

**Results Positive + Significant:**
→ Roll out to remaining pages
→ Document as proven tactic
→ Add to standard playbook

**Results Positive + Not Significant:**
→ Extend test duration
→ Or expand to more pages
→ Re-evaluate in 4 weeks

**Results Neutral:**
→ Analyze for implementation issues
→ Check for conflicting signals
→ Consider test design problems

**Results Negative:**
→ Roll back changes immediately
→ Document what didn't work
→ Analyze why

### Post-Experiment Documentation

## Experiment Results: [Name]

### Summary

**Hypothesis:** [original hypothesis]
**Result:** [CONFIRMED / REJECTED / INCONCLUSIVE]
**Recommendation:** [action to take]

### Data Summary

| Metric   | Control | Variant | Change | Significant? |
| -------- | ------- | ------- | ------ | ------------ |
| Clicks   |         |         | %      | Y/N          |
| CTR      |         |         | %      | Y/N          |
| Position |         |         | Δ      | Y/N          |

### Timeline

- Start date: [date]
- End date: [date]
- Duration: [X weeks]

### External Factors

- Algorithm updates during test: [Y/N, which]
- Competitor changes noted: [Y/N, what]
- Seasonal impact: [Y/N, how]

### Key Learnings

1. [learning 1]
2. [learning 2]
3. [learning 3]

### Next Steps

- [ ] [action 1]
- [ ] [action 2]

### Attachments

- Dashboard link: [url]
- Raw data: [link]
- Screenshots: [link]

---

## Experiment Log Template

### Master Tracking Spreadsheet

| ID   | Name                        | Hypothesis                                | Start Date | End Date   | Status   | Result    | Impact   |
| ---- | --------------------------- | ----------------------------------------- | ---------- | ---------- | -------- | --------- | -------- |
| E001 | Title keyword front-loading | IF keyword in first 3 words THEN CTR +15% | 2025-01-15 | 2025-02-15 | Complete | Confirmed | +18% CTR |
| E002 | FAQ schema on guides        | IF FAQ schema THEN rich results +50%      | 2025-02-01 | 2025-03-01 | Running  | Pending   | TBD      |
| E003 | Content expansion 500→1000  | IF longer content THEN position +3        | 2025-02-15 | TBD        | Planning | N/A       | N/A      |

### Status Values

- **Planning:** Designing experiment
- **Ready:** Approved, awaiting implementation
- **Running:** Active experiment
- **Analyzing:** Data collection complete, reviewing
- **Complete:** Decision made

### Result Values

- **Confirmed:** Hypothesis validated
- **Rejected:** Hypothesis disproven
- **Inconclusive:** No clear signal
- **Abandoned:** Test stopped early

---

## Common SEO Experiments

### Title Tag Experiments

| Test             | Variable                    | Expected Outcome |
| ---------------- | --------------------------- | ---------------- |
| Keyword position | First 3 words vs. end       | CTR improvement  |
| Power words      | With vs. without            | CTR improvement  |
| Numbers          | Include year/number vs. not | CTR improvement  |
| Length           | 50 chars vs. 60 chars       | CTR/truncation   |
| Question format  | Question vs. statement      | CTR change       |

### Content Experiments

| Test              | Variable                 | Expected Outcome        |
| ----------------- | ------------------------ | ----------------------- |
| Word count        | 500 vs. 1000 vs. 2000    | Ranking improvement     |
| FAQ section       | With vs. without         | PAA capture             |
| Table of contents | With vs. without         | Engagement/time on page |
| Statistics/data   | With sources vs. without | E-E-A-T signals         |
| Expert quotes     | With vs. without         | Authority signals       |

### Technical Experiments

| Test               | Variable               | Expected Outcome            |
| ------------------ | ---------------------- | --------------------------- |
| Schema markup      | With vs. without       | Rich result appearance      |
| Internal links     | +5 links vs. baseline  | Ranking improvement         |
| Image optimization | Alt text + compression | Image search traffic        |
| Page speed         | Optimized vs. baseline | Core Web Vitals improvement |

---

## Requirements for Valid Testing

### Minimum Requirements

1. **Traffic threshold:** 100k+ organic visits/month site-wide

   - OR 500+ sessions/month on test pages
   - Lower traffic = longer test duration needed

2. **Page homogeneity:** Similar structure/template

   - Product pages to product pages
   - Blog posts to blog posts
   - Not mixed page types

3. **Clean measurement:** Single variable changed

   - Don't change title AND content simultaneously
   - Isolate variables for clear attribution

4. **Sufficient duration:** See timeline guidelines

   - Minimum 2 weeks for any test
   - Longer for ranking-focused tests

5. **Documentation:** Everything recorded
   - Can't analyze what isn't tracked

### When Testing Isn't Possible

**If traffic too low:**

- Implement best practices directly
- Monitor results over longer periods
- Use industry benchmarks instead

**If pages too different:**

- Test one page at a time (case study approach)
- Look for directional signals
- Document qualitatively

---

## Tools for SEO Testing

### Free Tools

- Google Search Console (primary data source)
- Google Analytics 4 (traffic + conversions)
- Google Looker Studio (dashboards)
- Spreadsheets (tracking + analysis)

### Paid Tools (Enterprise)

- **SearchPilot:** Server-side SEO testing
- **SplitSignal (Semrush):** Causal Impact model
- **Optimizely:** A/B testing platform

### Monitoring Tools

- **Semrush Sensor:** Algorithm volatility
- **Mozcast:** Google "weather"
- **Advanced Web Ranking:** SERP volatility

---

## Integration with Workflow System

### Experiments from Workflow R

Rank enhancement recommendations become testable hypotheses:

**Workflow R Recommendation:** Update title to include keyword in first 3 words
**Experiment:** Test across 10 similar pages before rolling out site-wide

### Experiment-Validated Playbook

Over time, build a playbook of validated tactics:

```markdown
## Validated SEO Tactics

### Title Tags

✅ Keyword in first 3 words: +18% CTR (E001)
✅ Include year for evergreen: +12% CTR (E007)
❌ Questions in titles: No significant impact (E004)

### Content

✅ 1000+ words for informational: +2.3 positions avg (E002)
✅ FAQ section with schema: +35% PAA capture (E003)
? Expert quotes: Test inconclusive (E005)

### Technical

✅ FAQ schema: +40% rich result appearance (E006)
✅ +5 internal links: +1.5 positions avg (E008)
```

---

## Summary

The SEO Experiments Tracker provides:

✅ 7-step testing framework (Ideate → Determine)  
✅ Hypothesis formation template (IF/THEN/BECAUSE)  
✅ Control/variant group methodology  
✅ Timeline guidelines by test type  
✅ Monitoring setup instructions  
✅ Analysis and decision framework  
✅ Documentation templates  
✅ Master experiment log structure  
✅ Common experiment ideas  
✅ Tool recommendations

**Key Quote:**

> "Document everything: It is important to keep a detailed log of the experiments performed and their changes and results. Future experiments and analyses will benefit from this." — Ralf van Veen

**Critical Reminder:** Testing can help justify further investment or help prevent potentially negative impact. The investment in proper tracking pays dividends in confidence and organizational learning.
