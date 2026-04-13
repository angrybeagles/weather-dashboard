# Expert Co-founder Setup

Create a Claude project loaded with all your product documentation, acting as an expert co-founder for ongoing strategic and technical guidance.

**Duration:** 15-30 minutes
**Output:** Configured Claude Project with Knowledge Base

---

## When to Use

- Have completed all previous phases (strategy, competitors, transformation)
- Want ongoing AI assistance that understands your full context
- Need a strategic advisor with complete product knowledge
- Ready to start implementation with contextual guidance

---

## Project Structure

Organize your documents into a Claude project:

```
[Product Name] - Expert Co-founder/
├── strategy/
│   ├── master-strategy.md
│   ├── viability-scorecard.md
│   └── gtm-playbook.md
├── research/
│   ├── competitor-[name]-brief.md
│   ├── competitive-intelligence.md
│   └── target-segments.md
├── technical/
│   ├── starter-kit-selection.md
│   ├── transformation-report.md
│   └── implementation-checklist.md
└── decisions/
    ├── architecture-decisions.md
    └── feature-priorities.md
```

---

## Project Setup Steps

### Step 1: Create New Project

In Claude Desktop or claude.ai:

1. Create new Project
2. Name it: "[Product Name] - Expert Co-founder"
3. Set custom instructions (see below)

### Step 2: Upload Knowledge Base

Upload all documents from your product development process:

**Required Documents:**

- Master Strategy Document
- Viability Scorecard
- Competitor Briefs (all)
- Competitive Intelligence Synthesis
- Target Segments
- Transformation Report
- Implementation Checklist

**Optional Documents:**

- GTM Playbook
- Architecture Decision Records
- API Integration Guides
- Meeting Notes
- User Research

### Step 3: Configure Custom Instructions

```markdown
You are an expert co-founder for [Product Name], a [brief description].

You have access to comprehensive documentation including:

- Master strategy and viability scoring
- Competitive intelligence and positioning
- Technical transformation analysis
- Implementation plans and checklists

Your role:

1. **Strategic Advisor:** Help make product, marketing, and business decisions aligned with the strategy
2. **Technical Guide:** Provide implementation guidance based on the transformation analysis
3. **Quality Check:** Ensure decisions align with documented strategy and competitive positioning
4. **Memory:** Remember all context from uploaded documents

When answering:

- Reference specific documents when relevant
- Flag when a question contradicts documented strategy
- Suggest when documents need updating
- Think long-term about product-market fit

Key context:

- Target customer: [Primary segment]
- Core differentiator: [Key positioning]
- Current phase: [Development/Launch/Growth]
- Tech stack: [Stack details]
```

---

## Use Cases

### Strategic Questions

```
"Based on our positioning, should we add [feature X]?"
"How does this pricing change affect our competitive position?"
"What messaging would resonate with [segment]?"
```

### Technical Decisions

```
"Given our transformation plan, how should we implement [feature]?"
"What's the priority order for the next sprint?"
"Does this architecture decision align with our scalability needs?"
```

### Marketing Guidance

```
"Write landing page copy based on our positioning"
"Create a comparison table vs [competitor]"
"Draft an email sequence for our target segment"
```

### Implementation Support

```
"Walk me through implementing [domain] from the transformation report"
"What are the dependencies for [task]?"
"Review this code against our documented requirements"
```

---

## Keeping the Project Updated

### When to Update

- After major feature decisions
- When strategy pivots
- After significant competitor changes
- When entering new market segments
- After launch and real user feedback

### Update Checklist

- [ ] Update strategy document with learnings
- [ ] Add new competitor briefs as needed
- [ ] Document architecture decisions
- [ ] Track feature priority changes
- [ ] Record user feedback themes

---

## Integration with Other Skills

**Uses outputs from:**

- **idea-extraction** (idea-to-product) - Master strategy
- **competitive-intel** (idea-to-product) - Competitor briefs
- **transformation-analysis** (idea-to-product) - Technical plans
- **target-segments** (research) - Customer personas
- **gtm-playbook** (marketing/05-gtm) - Launch strategy

**Provides ongoing support for:**

- **direct-response-copy** (marketing/02-content-creation) - Contextual copywriting
- **email-sequences** (marketing/03-email-marketing) - Aligned messaging
- **content-repurposer** (marketing/04-distribution) - On-brand content

---

## Project Maintenance Tips

1. **Keep Documents Current:** Update when strategy changes
2. **Add Decision Records:** Document why decisions were made
3. **Include User Feedback:** Add real customer insights
4. **Track Competitors:** Update briefs when they change
5. **Version Important Docs:** Keep history of major changes

---

## Success Indicators

Your expert co-founder project is working well when:

- [ ] Quick answers to strategic questions
- [ ] Consistent advice aligned with strategy
- [ ] Catches contradictions and misalignments
- [ ] Remembers context across conversations
- [ ] Provides actionable implementation guidance

---
