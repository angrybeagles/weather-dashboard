---
name: growth-kit-research
description: "Market research and competitive intelligence. Use when user needs competitor analysis, target audience segments, positioning strategy, or keyword research. Outputs battle cards, segment profiles, positioning frameworks, and prioritized keyword clusters."
---

# Research Skill Pack

A focused research skill pack for Claude Code. Build the strategic foundation for marketing and product decisions through competitor analysis, audience segmentation, positioning, and keyword research.

---

## What This Is

A modular system for strategic research:

- **Competitor Analysis** - Map the competitive landscape
- **Target Segments** - Define and understand your audience
- **Positioning Strategy** - Find where you fit in the market
- **Keyword Research** - Discover what to write about

**Think of it as:** Your market research department in Claude Code. Understand before you execute.

**Related Skills:**

- `idea-to-product` - For product discovery and validation before research
- `marketing` - For content creation and distribution after research
- `media-creation` - For visual assets, brand direction, and image generation
- `crawl-cli` - For web crawling and content extraction (see crawl-cli/ folder)

---

## Skill Directory

| Skill                    | What It Does               | Inputs                           | Outputs                          |
| ------------------------ | -------------------------- | -------------------------------- | -------------------------------- |
| **competitor-analysis**  | Maps competitive landscape | Competitor list, research method | Battle cards, positioning gaps   |
| **target-segments**      | Defines audience segments  | Business context, product        | Segment profiles with messaging  |
| **positioning-strategy** | Creates market positioning | Research outputs                 | Positioning statement, framework |

**Supporting Tools:**
- Use the `crawl-cli` skill for web crawling and content extraction
- For keyword research, use **SEO Boost API** + `seo/workflow-s-content-strategy.md`

---

## Skill Dependencies

```
RESEARCH FLOW
├── competitor-analysis (understand the market)
├── target-segments (understand the audience)
├── positioning-strategy (find your place)
└── (crawl-cli skill for web content extraction)

KEYWORD RESEARCH (separate flow)
└── SEO Boost API + seo/workflow-s-content-strategy.md
```

**Entry Points:**

- From `idea-to-product`: competitive-intel feeds into competitor-analysis
- From scratch: Start with competitor-analysis or target-segments
- For content: Start with keyword-research

---

## When to Use Each Skill

### competitor-analysis

**Use when:**

- Entering a market with existing players
- Need battle cards for sales team
- Want to identify positioning gaps

**Output:** Competitive landscape + battle cards

### target-segments

**Use when:**

- Don't know who you're talking to
- Need messaging for different audiences
- Want to prioritize customer types

**Output:** Segment profiles with tailored messaging

### positioning-strategy

**Use when:**

- Have competitor and segment data
- Need a positioning statement
- Want to clarify differentiation

**Output:** Positioning framework + messaging hierarchy

### Keyword Research (via SEO skills)

**Use when:**

- Planning content strategy
- Need to know what to write about
- Want SEO-driven topic priorities

**How to run:**
1. Use **SEO Boost API** for keyword discovery
2. Run `seo/workflow-s-content-strategy.md` to categorize and prioritize

**Output:** Prioritized keyword clusters

---

## Research Workflows

### Full Market Research

```
1. (crawl-cli) Extract competitor content
   Crawl competitor sites

2. competitor-analysis
   Analyze competitive landscape

3. target-segments
   Define audience segments

4. positioning-strategy
   Create positioning framework

5. keyword-research
   Identify content opportunities
```

### Quick Competitive Scan

```
1. (crawl-cli) competitor sites
2. competitor-analysis
```

### Content Planning Only

```
1. keyword-research
2. (proceed to marketing skill)
```

---

## Integration with Other Skills

### From idea-to-product

| From idea-to-product | To research          | What transfers                |
| -------------------- | -------------------- | ----------------------------- |
| competitive-intel    | competitor-analysis  | Initial competitor list, gaps |
| idea-extraction      | target-segments      | Target customer definition    |
| idea-extraction      | positioning-strategy | Core value proposition        |

### To marketing

| From research        | To marketing         | What transfers      |
| -------------------- | -------------------- | ------------------- |
| competitor-analysis  | direct-response-copy | Competitive claims  |
| target-segments      | lead-magnet-builder  | Segment pain points |
| positioning-strategy | hook-finder          | Positioning angles  |
| keyword-research     | seo-content-planning | Priority keywords   |

---

## Quick Reference

### By Situation

| Situation               | Start With          | Then                 |
| ----------------------- | ------------------- | -------------------- |
| New to market           | competitor-analysis | Full workflow        |
| Know competitors        | target-segments     | positioning-strategy |
| Need content topics     | keyword-research    | To marketing skill   |
| Need competitor content | (crawl-cli)         | competitor-analysis  |

### By Output Needed

| Need                  | Run This             |
| --------------------- | -------------------- |
| Competitive landscape | competitor-analysis  |
| Audience profiles     | target-segments      |
| Positioning statement | positioning-strategy |
| Content topics        | keyword-research     |
| Extracted web content | (crawl-cli)          |

---

## Context Passing

### What Flows Forward

```
(crawl-cli)
└── → competitor-analysis: Extracted competitor content

competitor-analysis
├── → target-segments: Competitor audience insights
├── → positioning-strategy: Competitive gaps
└── → marketing/direct-response-copy: Battle cards

target-segments
├── → positioning-strategy: Audience priorities
└── → marketing/lead-magnet-builder: Segment pain points

positioning-strategy
├── → marketing/hook-finder: Positioning angles
└── → marketing/direct-response-copy: Messaging framework

keyword-research
└── → marketing/seo-content-planning: Priority clusters
```

### Compression Principle

| From                 | Pass This                | Not This            |
| -------------------- | ------------------------ | ------------------- |
| competitor-analysis  | Key gaps (3-5 bullets)   | Full battle cards   |
| target-segments      | Primary segment + pains  | All segment details |
| positioning-strategy | Positioning statement    | Full framework      |
| keyword-research     | Top cluster + 5 keywords | Full keyword list   |

---

## Anti-Patterns

### Don't:

- Skip competitor analysis for "unique" ideas
- Define segments without research
- Create positioning without competitive context
- Run keyword research without knowing the audience

### Do:

- Research before execution
- Validate assumptions with data
- Document findings for other skills
- Update research as market changes

---

## Success Criteria

Your research is complete when:

- [ ] Competitive landscape mapped with 3-5 competitors
- [ ] At least 2-3 target segments defined
- [ ] Positioning statement created
- [ ] Keyword clusters prioritized (if content needed)
- [ ] Ready to hand off to marketing skill
