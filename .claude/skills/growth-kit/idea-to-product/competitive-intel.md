# Competitive Intelligence

Deep research on direct competitors to inform positioning and strategy. This skill produces competitor briefs and synthesis documents with "How to Win" playbooks.

**Duration:** 30-60 minutes
**Output:** 2-5 Competitor Briefs + Competitive Intelligence Synthesis

---

## When to Use

- Have completed idea extraction and need to understand the competitive landscape
- Need battle cards for sales conversations
- Want to identify positioning gaps and opportunities
- Preparing launch messaging that differentiates

---

## Competitor Brief Template

For each direct competitor, research and document:

### Basic Information

```markdown
## Competitor: [Name]

**Website:** [URL]
**Founded:** [Year]
**Funding:** [Amount/Stage or Bootstrapped]
**Team Size:** [Estimate]
**Pricing:** [Range]
```

### Product Analysis

```markdown
### Core Offering

- **Main product:** [Description]
- **Target customer:** [Who they serve]
- **Primary use case:** [What problem they solve]

### Feature Set

| Feature     | Available      | Notes     |
| ----------- | -------------- | --------- |
| [Feature 1] | Yes/No/Partial | [Details] |
| [Feature 2] | Yes/No/Partial | [Details] |
| [Feature 3] | Yes/No/Partial | [Details] |

### Pricing Structure

| Tier       | Price  | Key Limits |
| ---------- | ------ | ---------- |
| Free       | $0     | [Limits]   |
| Starter    | $X/mo  | [Limits]   |
| Pro        | $X/mo  | [Limits]   |
| Enterprise | Custom | [Limits]   |
```

### Positioning Analysis

```markdown
### Their Positioning

- **Tagline:** [Their headline]
- **Core promise:** [What they claim to deliver]
- **Differentiation:** [How they position against others]

### Messaging Themes

- [Theme 1 they emphasize]
- [Theme 2 they emphasize]
- [Theme 3 they emphasize]
```

### SWOT Analysis

```markdown
### Strengths

- [What they do well]
- [Advantages they have]

### Weaknesses

- [What they do poorly]
- [Gaps in their offering]

### Opportunities (for you)

- [Where you can beat them]
- [Underserved segments]

### Threats (from them)

- [What makes them dangerous]
- [Barriers they create]
```

---

## Competitive Intelligence Synthesis

After analyzing 2-5 competitors, synthesize findings:

### Market Overview

```markdown
## Market Landscape

**Total Competitors Identified:** [Number]
**Direct Competitors Analyzed:** [Number]

### Market Segmentation

| Segment    | Competitors | Positioning         |
| ---------- | ----------- | ------------------- |
| Enterprise | [Names]     | [How they position] |
| SMB        | [Names]     | [How they position] |
| Individual | [Names]     | [How they position] |

### Pricing Landscape

| Tier Level | Market Range | Sweet Spot |
| ---------- | ------------ | ---------- |
| Entry      | $X-Y/mo      | $X/mo      |
| Mid        | $X-Y/mo      | $X/mo      |
| Premium    | $X-Y/mo      | $X/mo      |
```

### Feature Gap Analysis

```markdown
## Feature Comparison Matrix

| Feature     | [Comp A] | [Comp B] | [Comp C] | Your Product   |
| ----------- | -------- | -------- | -------- | -------------- |
| [Feature 1] | Yes      | No       | Partial  | Planned        |
| [Feature 2] | Yes      | Yes      | Yes      | MVP            |
| [Feature 3] | No       | Yes      | No       | Differentiator |
```

### How to Win Playbooks

For each major competitor, document specific strategies:

```markdown
## How to Win Against [Competitor Name]

### Their Achilles Heel

[The fundamental weakness you can exploit]

### Target Customer Profile

[Customers who are underserved by this competitor]

### Key Messages

1. "[Message that highlights their weakness]"
2. "[Message that positions your strength]"
3. "[Message that creates urgency to switch]"

### Objection Handling

| Objection                     | Response        |
| ----------------------------- | --------------- |
| "But they have more features" | [Your response] |
| "They're more established"    | [Your response] |
| "My team already uses them"   | [Your response] |

### Competitive Triggers

- [Situation 1 where prospect considers switching]
- [Situation 2 where prospect considers switching]
```

---

## Positioning Gap Analysis

After synthesis, identify your positioning opportunity:

```markdown
## Positioning Opportunity

### Gaps Identified

1. **[Gap 1]:** [No competitor addresses X]
2. **[Gap 2]:** [All competitors weak at Y]
3. **[Gap 3]:** [Underserved segment Z]

### Recommended Positioning

**For:** [Target customer]
**Who:** [Has this need/problem]
**Our product is:** [Category]
**That:** [Key benefit]
**Unlike:** [Competitors]
**We:** [Key differentiator]

### Differentiation Strategy

| Dimension     | Competitors      | You             | Why This Matters |
| ------------- | ---------------- | --------------- | ---------------- |
| [Dimension 1] | [Their approach] | [Your approach] | [Customer value] |
| [Dimension 2] | [Their approach] | [Your approach] | [Customer value] |
```

---

## Integration with Other Skills

**Feeds into:**

- **hook-finder** (marketing/01-foundations) - Use gaps for differentiation angles
- **direct-response-copy** (marketing/02-content-creation) - Competitive messaging
- **target-segments** (research) - Refine segment definitions
- **gtm-playbook** (marketing/05-gtm) - Launch positioning

**Cross-reference with Marketing Skill:**

The competitive intelligence directly informs:

- Your positioning angles and hooks
- Landing page differentiation claims
- Content strategy topics (comparison posts)
- Sales objection handling

---

## Research Methods

### Web Crawling with crawl-cli

Use the `crawl-cli` skill to systematically extract competitor content:

```bash
# Crawl competitor's entire docs/marketing site
crawl-cli https://competitor.com --discover -m 100 -O "./competitor-research/competitor-name/"

# Extract specific sections
crawl-cli https://competitor.com/pricing -o "./competitor-research/competitor-pricing.md"
crawl-cli https://competitor.com/features -o "./competitor-research/competitor-features.md"
```

**What to crawl:**

- Pricing pages (for pricing structure analysis)
- Features/product pages (for feature comparison)
- About/team pages (for company intel)
- Blog (for positioning and messaging themes)
- Documentation (for product depth assessment)

### Primary Sources

- Competitor websites and pricing pages
- Their blog and content marketing
- G2, Capterra, TrustRadius reviews
- Social media presence (Twitter, LinkedIn)
- ProductHunt launches and comments

### Secondary Sources

- News articles and press releases
- Crunchbase for funding/team data
- SimilarWeb for traffic estimates
- BuiltWith for technology stack

### Direct Intelligence

- Sign up for free trials
- Watch demo videos
- Join their community/Discord
- Read their documentation

---

## Output Checklist

Before moving forward:

- [ ] 2-5 competitor briefs completed
- [ ] Feature comparison matrix filled
- [ ] Pricing landscape mapped
- [ ] How to Win playbooks for top 2-3 competitors
- [ ] Positioning gaps identified
- [ ] Recommended positioning documented

---
