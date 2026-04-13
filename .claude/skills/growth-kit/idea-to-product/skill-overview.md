---
name: idea-to-product
description: "Product discovery and validation workflow. Use when user has a new business idea to validate, needs competitive intelligence, wants to select a starter kit, or needs implementation planning. Outputs strategy documents, competitor briefs, and technical plans."
---

# Idea to Product Discovery

A complete product discovery and validation skill pack for Claude Code. Take any business idea from concept through competitive analysis, technical planning, and implementation-ready specifications.

---

## What This Is

A structured workflow for validating and planning new products:

- **Idea Extraction** - Deep-dive interviews to surface viable concepts
- **Competitive Intel** - Understand the market landscape
- **Starter Kit Selection** - Match requirements to technical foundations
- **Transformation Analysis** - Plan implementation across 7 domains
- **Expert Co-founder** - Create an AI advisor with full context

**Think of it as:** A product discovery consultant in your Claude Code. Validates before you build, plans before you code.

**Related Skills:**

- `research` - For detailed competitor analysis, segments, and keywords
- `marketing` - For content creation and go-to-market execution
- `media-creation` - For visual assets, brand direction, and image generation
- `crawl-cli` - For web crawling and content extraction

---

## Skill Directory

| Skill                       | What It Does                   | Inputs                     | Outputs                        |
| --------------------------- | ------------------------------ | -------------------------- | ------------------------------ |
| **idea-extraction**         | Validates product ideas        | Business concept           | Strategy doc + viability score |
| **competitive-intel**       | Deep competitor research       | Competitor list            | Briefs + How to Win playbooks  |
| **starter-kit-selection**   | Matches requirements to kits   | Tech requirements          | Selected kit + gap analysis    |
| **transformation-analysis** | Plans implementation by domain | Selected kit, requirements | 7-domain transformation report |
| **expert-cofounder**        | Creates Claude project advisor | All strategy docs          | Configured expert project      |

---

## The 5-Phase Workflow

```
PHASE 1: IDEA EXTRACTION
├── Deep-dive interview (20 questions)
├── Viability scoring (market, competition, execution, monetization)
└── Output: Master strategy document

PHASE 2: COMPETITIVE INTELLIGENCE
├── Identify 3-5 key competitors
├── Create detailed competitor briefs
├── Synthesize "How to Win" playbook
└── Output: Competitive landscape + positioning gaps

PHASE 3: STARTER KIT SELECTION
├── Map technical requirements from strategy
├── Evaluate available starter kits
├── Score fit and identify gaps
└── Output: Selected foundation + gap analysis

PHASE 4: TRANSFORMATION ANALYSIS
├── Analyze 7 implementation domains
├── Create atomic task lists
├── Estimate effort and dependencies
└── Output: Implementation-ready technical plan

PHASE 5: EXPERT CO-FOUNDER
├── Compile all documentation
├── Configure Claude project
├── Set up knowledge base
└── Output: AI advisor with full product context
```

---

## When to Use Each Skill

### idea-extraction

**Use when:**

- Starting a new product from scratch
- Pivoting an existing product
- Validating a business concept before building

**Output:** Master strategy document with viability score (0-100)

### competitive-intel

**Use when:**

- Entering a market with existing players
- Need to understand competitive positioning
- Want to identify gaps and opportunities

**Output:** Competitor briefs + synthesis playbook

### starter-kit-selection

**Use when:**

- Ready to choose technical foundation
- Have clear product requirements
- Need to evaluate build vs buy decisions

**Output:** Selected kit + gap analysis

### transformation-analysis

**Use when:**

- Have selected a starter kit
- Ready to plan implementation
- Need detailed technical specifications

**Output:** 7-domain transformation report + task lists

### expert-cofounder

**Use when:**

- Completed previous phases
- Want ongoing AI assistance
- Ready to start implementation

**Output:** Configured Claude project with full context

---

## Skill Dependencies

```
idea-extraction (start here)
    │
    ▼
competitive-intel (understand market)
    │
    ▼
starter-kit-selection (choose foundation)
    │
    ▼
transformation-analysis (plan implementation)
    │
    ▼
expert-cofounder (ongoing guidance)
```

**Rule:** Complete each phase before moving to the next. Earlier phases inform later ones.

---

## Integration with Marketing Skill

The idea-to-product and marketing skills are designed to work together:

### Handoff Points

| From idea-to-product    | To marketing        | What transfers                   |
| ----------------------- | ------------------- | -------------------------------- |
| idea-extraction         | hook-finder         | Core value prop, target customer |
| competitive-intel       | competitor-analysis | Competitive landscape, gaps      |
| idea-extraction         | voice-profiler      | Brand personality, tone          |
| transformation-analysis | gtm-playbook        | Timeline, launch readiness       |

### Recommended Sequence

**For new products:**

1. Complete idea-to-product workflow (all 5 phases)
2. Start marketing with hook-finder (uses strategy from phase 1)
3. Run competitor-analysis (extends competitive-intel from phase 2)
4. Continue with marketing execution skills

**For existing products:**

- Skip to marketing skill directly
- Use idea-extraction only if pivoting or repositioning

---

## Quick Reference

### By Situation

| Situation         | Start With              | Then                  |
| ----------------- | ----------------------- | --------------------- |
| Brand new idea    | idea-extraction         | Full workflow         |
| Know the market   | competitive-intel       | Continue from phase 2 |
| Have requirements | starter-kit-selection   | Continue from phase 3 |
| Kit selected      | transformation-analysis | Continue from phase 4 |
| Ready to build    | expert-cofounder        | Then marketing skill  |

### By Output Needed

| Need                  | Run This                |
| --------------------- | ----------------------- |
| Viability assessment  | idea-extraction         |
| Competitive landscape | competitive-intel       |
| Technical foundation  | starter-kit-selection   |
| Implementation plan   | transformation-analysis |
| AI advisor            | expert-cofounder        |

---

## Context Passing

### What Flows Forward

```
idea-extraction
├── → competitive-intel: Target market, key differentiators
├── → starter-kit-selection: Technical requirements, constraints
└── → expert-cofounder: Full strategy document

competitive-intel
├── → starter-kit-selection: Integration requirements
├── → transformation-analysis: Feature comparison needs
└── → expert-cofounder: All competitor briefs

starter-kit-selection
├── → transformation-analysis: Selected kit, gap analysis
└── → expert-cofounder: Technical decisions

transformation-analysis
└── → expert-cofounder: Implementation plan, task lists
```

### Compression Principle

When passing between phases, compress:

| From                    | Pass This              | Not This                  |
| ----------------------- | ---------------------- | ------------------------- |
| idea-extraction         | Core strategy (1 page) | Full interview transcript |
| competitive-intel       | Key differentiators    | All competitor details    |
| starter-kit-selection   | Selected kit + gaps    | Evaluation matrix         |
| transformation-analysis | Priority tasks         | Full domain analysis      |

---

## Anti-Patterns

### Don't:

- Skip idea extraction for "obvious" ideas (validate anyway)
- Rush through competitive intel (your positioning depends on it)
- Choose a kit without requirements (leads to rework)
- Start coding before transformation analysis (hidden complexity)
- Forget to create expert co-founder (loses all context)

### Do:

- Spend adequate time on idea extraction
- Document all competitor insights
- Match kit capabilities to actual requirements
- Create atomic, testable tasks
- Maintain expert co-founder as source of truth

---

## Success Criteria

Your idea-to-product workflow is complete when:

- [ ] Viability score calculated and acceptable (>60)
- [ ] 3-5 competitors analyzed with positioning gaps identified
- [ ] Starter kit selected with clear gap analysis
- [ ] All 7 domains analyzed with task lists
- [ ] Expert co-founder project configured and tested
- [ ] Ready to begin marketing skill workflow

---

## Handoff to Marketing Skill

When idea-to-product is complete, transition to the marketing skill with these outputs:

### What Marketing Needs From You

| Output                       | Source                  | Marketing Destination                |
| ---------------------------- | ----------------------- | ------------------------------------ |
| Core value proposition       | idea-extraction         | voice-profiler, hook-finder          |
| Target customer profile      | idea-extraction         | target-segments, lead-magnet-builder |
| Competitive positioning gaps | competitive-intel       | hook-finder, direct-response-copy    |
| Key differentiators          | competitive-intel       | direct-response-copy                 |
| Launch timeline              | transformation-analysis | gtm-playbook                         |

### Handoff Checklist

Before starting marketing, confirm:

- [ ] **Positioning statement drafted** - "For [audience] who [need], [product] is [category] that [benefit]. Unlike [competitors], we [differentiator]."
- [ ] **Target customer defined** - Demographics, psychographics, pain points documented
- [ ] **Competitive landscape mapped** - Know who you're positioning against
- [ ] **Value proposition clear** - Can articulate in one sentence why someone should buy
- [ ] **Technical foundation set** - Know what you're building (for accurate marketing claims)

### Starting Marketing Skill

**Recommended entry points:**

1. **If brand voice undefined:** Start with `marketing/01-foundations/voice-profiler.md`
2. **If positioning unclear:** Start with `marketing/01-foundations/hook-finder.md`
3. **If ready for content:** Start with `research/keyword-research.md` then SEO content
4. **If launching soon:** Start with `marketing/05-gtm/gtm-playbook.md`

### Context to Pass

When invoking marketing skills, provide:

```markdown
## Product Context (from idea-to-product)

**Product:** [One-line description]
**Target Customer:** [Who it's for]
**Core Problem Solved:** [The pain point]
**Key Differentiator:** [What makes it unique]
**Competitive Position:** [How it compares to alternatives]
**Launch Status:** [Timeline/readiness]
```

---

## The 7 Domains (Transformation Analysis)

Quick reference for what transformation-analysis covers:

1. **Authentication & Multi-tenancy** - User model, roles, workspace isolation
2. **Payments** - Pricing tiers, billing logic, webhooks
3. **Database Architecture** - Schema changes, migrations, indexes
4. **Background Jobs** - Scheduled tasks, queue processing
5. **UI Components** - Pages, components, design system
6. **Integrations** - Third-party APIs, environment variables
7. **Gap Synthesis** - Priority ordering, effort estimates, risks

See `transformation-analysis.md` for detailed templates.
