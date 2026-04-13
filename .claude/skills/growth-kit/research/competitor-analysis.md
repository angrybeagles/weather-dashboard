# Competitive Intelligence Framework

A structured approach for analyzing competitors and maintaining ongoing competitive awareness.

---

## Quick Reference: Competitor Tracking

### Primary Competitors (Direct)

| Competitor     | Product Type | Price | Target Audience | Threat Level      |
| -------------- | ------------ | ----- | --------------- | ----------------- |
| [Competitor A] | [Type]       | $[X]  | [Audience]      | [HIGH/MEDIUM/LOW] |
| [Competitor B] | [Type]       | $[X]  | [Audience]      | [Level]           |
| [Competitor C] | [Type]       | $[X]  | [Audience]      | [Level]           |

### Adjacent Competitors (Indirect)

| Competitor   | Product Type       | Overlap Area       | Notes                  |
| ------------ | ------------------ | ------------------ | ---------------------- |
| [Company]    | Free documentation | Tool education     | Not direct competition |
| [Platform]   | Free content       | Beginner education | Funnel, not competitor |
| [Competitor] | [Type]             | [Overlap]          | [Notes]                |

---

## Competitor Brief Template

Use this structure when analyzing a new competitor:

### [Competitor Name] Brief

**Last Updated:** [Date]
**URL:** [Website]
**Threat Level:** HIGH | MEDIUM | LOW

#### Overview

| Attribute       | Value                                   |
| --------------- | --------------------------------------- |
| Product Type    | [Course / Framework / SaaS / Community] |
| Price           | $[X]                                    |
| Target Audience | [Description]                           |
| Platform/Tools  | [What they teach/use]                   |
| Customers       | [X]                                     |
| Launch Date     | [Date]                                  |

#### Product Structure

- **Format:** [Video / Text / Code / Mixed]
- **Length:** [X hours / X modules]
- **Delivery:** [Self-paced / Cohort / Live]
- **Support:** [Community / 1:1 / AI assistant / None]
- **Guarantee:** [X days / Outcome-based / None]

#### Pricing Model

| Tier         | Price | Includes   |
| ------------ | ----- | ---------- |
| [Base]       | $[X]  | [Features] |
| [Premium]    | $[X]  | [Features] |
| [Enterprise] | $[X]  | [Features] |

#### Target Audience (Verbatim from their marketing)

**Who it's for:**

- [Bullet 1]
- [Bullet 2]

**Who it's NOT for:**

- [Bullet 1]
- [Bullet 2]

#### Key Messaging

**Headline:** "[Their main headline]"

**Pain Points Addressed:**

- [Pain 1]
- [Pain 2]

**Promise/Outcome:**

- [Promise 1]
- [Promise 2]

#### Strengths (What they do well)

1. [Strength 1]
2. [Strength 2]
3. [Strength 3]

#### Weaknesses (Gaps we can exploit)

1. [Weakness 1] - Our opportunity: [How we win]
2. [Weakness 2] - Our opportunity: [How we win]
3. [Weakness 3] - Our opportunity: [How we win]

#### Your Positioning Against Them

**They are:** [One-line description]
**We are:** [One-line differentiation]

**Win on:**

- [Advantage 1]
- [Advantage 2]

**Don't compete on:**

- [Their strength 1]
- [Their strength 2]

---

## Competitor Monitoring Checklist

### Monthly Review Tasks

- [ ] Check competitor pricing changes
- [ ] Review new content/features launched
- [ ] Monitor social media for positioning changes
- [ ] Search for new reviews/testimonials
- [ ] Check job postings (indicates growth areas)

### Quarterly Deep Dive

- [ ] Full landing page analysis
- [ ] Customer sentiment review (Reddit, Twitter, reviews)
- [ ] Feature comparison update
- [ ] Market positioning reassessment

---

## Battle Card Template

Quick reference for positioning conversations:

### vs [Competitor Name]

**30-Second Pitch:**

> "Unlike [Competitor] which [weakness], [Your Product] [strength]. This means you [concrete benefit]."

**Key Differentiators:**

| They                | We                |
| ------------------- | ----------------- |
| [Their approach]    | [Our approach]    |
| [Their limitation]  | [Our advantage]   |
| [Their price/model] | [Our price/model] |

**Handle Common Objections:**

"But [Competitor] teaches a process I can reuse"

- [Your response about instant capability vs learning curve]

"But [Competitor] has more content/hours"

- [Your response about results vs content volume]

"But [Competitor] is more established"

- [Your response about innovation/specialization]

---

## Competitive Response Triggers

### When to Update Analysis

| Trigger                                  | Action                     |
| ---------------------------------------- | -------------------------- |
| Competitor launches new product          | Full brief update          |
| Competitor changes pricing               | Pricing section update     |
| Competitor gets major press/visibility   | Assess threat level change |
| Customer mentions competitor in feedback | Note positioning gap       |
| Competitor targets your keywords         | SEO response planning      |

### Response Playbook

**If competitor copies your feature:**

- Document the copy (screenshots, dates)
- Emphasize your innovation and depth of implementation
- Continue improving faster than they can copy

**If competitor undercuts your price:**

- Don't race to the bottom
- Emphasize value and results over cost
- Consider targeted offers for specific segments

**If competitor targets your customers directly:**

- Strengthen customer relationships
- Highlight switching costs and unique value
- Create comparison content

---

## Integration with Other Skills

**After completing competitive intelligence:**

1. **hook-finder** - Use gaps identified for differentiation
2. **direct-response-copy** - Update messaging based on competitive landscape
3. **seo-content** - Create comparison content for organic traffic
4. **gtm-playbook** - Adjust launch positioning based on competitors

---

## Competitor Analysis Prompt Template

Use this prompt to analyze a new competitor:

```
Analyze [COMPETITOR URL] as a competitor. Research thoroughly and provide:

1. **Product Overview:** What they sell, pricing, target audience
2. **Messaging Analysis:** Key headlines, pain points addressed, promises made
3. **Strengths:** What they do well (be objective)
4. **Weaknesses:** Gaps and limitations
5. **Positioning Comparison:** How we differ and win
6. **Threat Assessment:** High/Medium/Low and why

Use web search to gather current information from their website, social media, and reviews.

Format as a competitor brief following the standard template.
```

---

## Market Landscape Mapping

### Category Mapping

Plot competitors on these dimensions:

**Price vs Depth:**

```
        HIGH PRICE
            |
            |    [Enterprise Tools]
            |
            |
LOW DEPTH --+------------------ HIGH DEPTH
            |
            |    [Your Position?]
            |
        LOW PRICE
```

**Audience Sophistication:**

```
        BEGINNERS
            |
            |
            |
BROAD ------+------------- NARROW
TARGET      |               NICHE
            |
            |
        EXPERTS
```

### Positioning Gaps

Identify white space in the market:

| Gap Identified        | Competitor Coverage             | Our Opportunity     |
| --------------------- | ------------------------------- | ------------------- |
| [Underserved segment] | [How competitors miss it]       | [How we can own it] |
| [Unmet need]          | [Why competitors don't address] | [How we can solve]  |

---

## Ongoing Intelligence Gathering

### Web Crawling with crawl-cli

Use the `crawl-cli` skill to systematically extract and archive competitor content:

```bash
# Full competitor site crawl
crawl-cli https://competitor.com --discover -m 100 -O "./research/competitors/competitor-name/"

# Targeted extraction
crawl-cli https://competitor.com/pricing -o "./research/competitors/competitor-pricing.md"
crawl-cli https://competitor.com/blog -d 2 -m 50 -O "./research/competitors/competitor-blog/"
```

**Recommended crawl schedule:**

- **Monthly:** Pricing pages, feature pages, homepage messaging
- **Quarterly:** Full site crawl, blog content, documentation

### Information Sources

**Primary:**

- Competitor websites and pricing pages
- Customer reviews (G2, Capterra, Reddit, Twitter)
- Social media accounts
- Email newsletters (sign up for all competitors)
- Product changelogs

**Secondary:**

- Industry publications and blogs
- Conference presentations
- Podcast appearances
- Job postings
- Press releases

### Weekly Scan Checklist

- [ ] Check competitor social accounts for announcements
- [ ] Review relevant subreddits for competitor mentions
- [ ] Scan industry Twitter for trends
- [ ] Check competitor email inbox for updates

---

## The Test

Good competitive intelligence answers:

1. **Who are we actually competing against?** (Named competitors with specifics)
2. **Where do we win?** (Concrete advantages we can claim)
3. **Where do we lose?** (Honest assessment of competitor strengths)
4. **What gaps exist?** (White space we can own)
5. **How should we respond?** (Actionable positioning decisions)

If you can't answer these clearly, dig deeper.

---

## Competitor & Alternative Pages

Transform competitive intelligence into SEO content that captures high-intent traffic.

### Page Format 1: [Competitor] Alternative (Singular)

**Search intent:** User actively looking to switch from a specific competitor

**URL pattern:** `/alternatives/[competitor]` or `/[competitor]-alternative`

**Target keywords:**
- "[Competitor] alternative"
- "alternative to [Competitor]"
- "switch from [Competitor]"
- "[Competitor] replacement"

**Page structure:**
1. Why people look for alternatives (validate their pain)
2. Summary: You as the alternative (quick positioning)
3. Detailed comparison (features, service, pricing)
4. Who should switch (and who shouldn't)
5. Migration path
6. Social proof from switchers
7. CTA

---

### Page Format 2: [Competitor] Alternatives (Plural)

**Search intent:** User researching options, earlier in journey

**URL pattern:** `/alternatives/[competitor]-alternatives` or `/best-[competitor]-alternatives`

**Target keywords:**
- "[Competitor] alternatives"
- "best [Competitor] alternatives"
- "tools like [Competitor]"
- "[Competitor] competitors"

**Page structure:**
1. Why people look for alternatives (common pain points)
2. What to look for in an alternative (criteria framework)
3. List of alternatives (you first, but include real options)
4. Comparison table (summary)
5. Detailed breakdown of each alternative
6. Recommendation by use case
7. CTA

**Important:** Include 4-7 real alternatives. Being genuinely helpful builds trust and ranks better.

---

### Page Format 3: You vs [Competitor]

**Search intent:** User directly comparing you to a specific competitor

**URL pattern:** `/vs/[competitor]` or `/compare/[you]-vs-[competitor]`

**Target keywords:**
- "[You] vs [Competitor]"
- "[Competitor] vs [You]"
- "[You] compared to [Competitor]"

**Page structure:**
1. TL;DR summary (key differences in 2-3 sentences)
2. At-a-glance comparison table
3. Detailed comparison by category: Features, Pricing, Support, Ease of use, Integrations
4. Who [You] is best for
5. Who [Competitor] is best for (be honest)
6. Testimonials from switchers
7. Migration support
8. CTA

---

### Page Format 4: [Competitor A] vs [Competitor B]

**Search intent:** User comparing two competitors (not you directly)

**URL pattern:** `/compare/[competitor-a]-vs-[competitor-b]`

**Target keywords:**
- "[Competitor A] vs [Competitor B]"
- "[Competitor A] or [Competitor B]"

**Page structure:**
1. Overview of both products
2. Comparison by category
3. Who each is best for
4. The third option (introduce yourself)
5. Comparison table (all three)
6. CTA

**Why this works:** Captures search traffic for competitor terms, positions you as knowledgeable, introduces you to qualified audience.

---

### Index Pages

Each format needs an index page listing all pages of that type.

**Alternatives Index** (`/alternatives`):
```markdown
## Explore [Your Product] as an Alternative

Looking to switch? See how [Your Product] compares:

- **[Notion Alternative](/alternatives/notion)** - Better for teams who need [X]
- **[Airtable Alternative](/alternatives/airtable)** - Better for teams who need [Y]
```

**Vs Comparisons Index** (`/vs` or `/compare`):
```markdown
## Compare [Your Product]

### [Your Product] vs. the Competition
- **[[Your Product] vs Notion](/vs/notion)** - Best for [differentiator]
- **[[Your Product] vs Airtable](/vs/airtable)** - Best for [differentiator]

### Other Comparisons
- **[Notion vs Airtable](/compare/notion-vs-airtable)**
```

---

### Comparison Page Section Templates

**TL;DR Summary:**
```markdown
**TL;DR**: [Competitor] excels at [strength] but struggles with [weakness].
[Your product] is built for [your focus], offering [key differentiator].
Choose [Competitor] if [their ideal use case]. Choose [You] if [your ideal use case].
```

**Paragraph Comparison (Not Just Tables):**
```markdown
## Features

[Competitor] offers [description of their approach]. Their strength is [specific strength],
which works well for [use case]. However, [limitation] can be challenging for [user type].

[Your product] takes a different approach with [your approach]. This means [benefit],
though [honest tradeoff]. Teams who [specific need] often find this more effective.
```

**Pricing Comparison:**
```markdown
## Pricing

| | [Competitor] | [Your Product] |
|---|---|---|
| Free tier | [Details] | [Details] |
| Starting price | $X/user/mo | $X/user/mo |
| Business tier | $X/user/mo | $X/user/mo |

**Total cost consideration**: Beyond per-seat pricing, consider [hidden costs, add-ons].
```

**Who It's For Section:**
```markdown
## Who Should Choose [Competitor]

[Competitor] is the right choice if:
- [Specific use case or need]
- [Team type or size]
- [Workflow or requirement]

**Ideal [Competitor] customer**: [Persona description in 1-2 sentences]

## Who Should Choose [Your Product]

[Your product] is built for teams who:
- [Specific use case or need]
- [Team type or size]
- [Workflow or requirement]

**Ideal [Your product] customer**: [Persona description in 1-2 sentences]
```

**Migration Section:**
```markdown
## Switching from [Competitor]

### What transfers
- [Data type]: [How easily, any caveats]

### What needs reconfiguration
- [Thing]: [Why and effort level]

### Migration support
We offer [migration support details]:
- [Free data import tool / white-glove migration]
- [Documentation / migration guide]
- [Timeline expectation]

### What customers say about switching
> "[Quote from customer who switched]"
> - [Name], [Role] at [Company]
```

---

### Centralized Competitor Data Architecture

Create a single source of truth for each competitor:

```yaml
# competitor_data/notion.yaml
name: Notion
website: notion.so
tagline: "The all-in-one workspace"

# Positioning
primary_use_case: "docs + light databases"
target_audience: "teams wanting flexible workspace"
market_position: "premium, feature-rich"

# Pricing
pricing_model: per-seat
free_tier: true
starter_price: $8/user/month

# Features (rate 1-5)
features:
  documents: 5
  databases: 4
  project_management: 3

# Strengths
strengths:
  - Extremely flexible and customizable
  - Beautiful, modern interface
  - Strong template ecosystem

# Weaknesses
weaknesses:
  - Can be slow with large databases
  - Learning curve for advanced features

# Best for
best_for:
  - Teams wanting all-in-one workspace
  - Documentation-first teams

# Not ideal for
not_ideal_for:
  - Complex project management needs
  - Large databases (1000s of rows)

# Migration notes
migration_from:
  difficulty: medium
  data_export: "Markdown, CSV, HTML"
  time_estimate: "1-3 days for small team"
```

**Benefits of centralized data:**
- Update competitor pricing once, updates everywhere
- Add new feature comparison once, appears on all pages
- Consistent accuracy across pages
- Easier to maintain at scale

---

### Comparison Page SEO

**Keyword targeting by format:**

| Format | Primary Keywords | Secondary Keywords |
|--------|-----------------|-------------------|
| Alternative (singular) | [Competitor] alternative | alternative to [Competitor], switch from [Competitor] |
| Alternatives (plural) | [Competitor] alternatives | best [Competitor] alternatives, tools like [Competitor] |
| You vs Competitor | [You] vs [Competitor] | [Competitor] vs [You], [You] compared to [Competitor] |

**Internal linking:**
- Link between related competitor pages
- Link from feature pages to relevant comparisons
- Hub page linking to all competitor content

---

### Comparison Page Principles

1. **Honesty Builds Trust:** Acknowledge competitor strengths, be accurate about your limitations
2. **Depth Over Surface:** Go beyond feature checklists, explain why differences matter
3. **Help Them Decide:** Different tools fit different needs, be clear about who you're best for
4. **Modular Architecture:** Centralized competitor data, updates propagate to all pages

---

## Related Skills

- **seo/seo-master-reference** - Programmatic SEO for competitor pages at scale
- **seo/workflow-t-technical** - Schema markup for comparison pages
- **marketing/references/copywriting-frameworks** - Writing compelling comparison copy
- **cro/page-cro** - Optimizing competitor pages for conversion
