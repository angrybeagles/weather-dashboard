# Starter Kit Selection

Match your product requirements to available starter kits and frameworks. This skill helps you choose the right foundation and identify gaps that need custom development.

**Duration:** 15-30 minutes
**Output:** Selected Starter Kit + Requirements Mapping + Gap Analysis

---

## When to Use

- Have completed idea extraction and competitive analysis
- Ready to choose technical foundation for development
- Need to estimate development scope based on starter kit coverage
- Want to identify what needs custom building vs. what's provided

---

## Requirements Mapping Framework

### Step 1: Document Core Requirements

From your strategy document, extract technical requirements:

```markdown
## Technical Requirements Summary

### Authentication & Users

- [ ] User registration/login
- [ ] OAuth providers needed: [Google, GitHub, etc.]
- [ ] Multi-tenancy (workspaces/teams)
- [ ] Role-based access control
- [ ] Invite system

### Payments & Billing

- [ ] Subscription billing
- [ ] One-time payments
- [ ] Usage-based billing
- [ ] Free tier / trial
- [ ] Multiple pricing tiers

### Data & Storage

- [ ] Database type: [PostgreSQL, etc.]
- [ ] File uploads
- [ ] Media handling
- [ ] Data export
- [ ] Backup requirements

### Integrations

- [ ] Email (transactional)
- [ ] Email (marketing)
- [ ] Analytics
- [ ] AI/ML APIs
- [ ] Third-party APIs: [List]

### Background Processing

- [ ] Scheduled jobs
- [ ] Queue processing
- [ ] Webhooks
- [ ] Real-time features

### Frontend

- [ ] Dashboard/admin
- [ ] Public marketing pages
- [ ] Mobile responsive
- [ ] Component library needed
```

### Step 2: Evaluate Starter Kits

For each potential starter kit:

```markdown
## Starter Kit Evaluation: [Kit Name]

**URL:** [Link]
**Tech Stack:** [Next.js, Supabase, etc.]
**Price:** [One-time/subscription]
**Last Updated:** [Date]

### Feature Coverage

| Requirement           | Coverage | Notes                 |
| --------------------- | -------- | --------------------- |
| Auth (email/password) | Full     | Built-in              |
| OAuth (Google)        | Full     | Configured            |
| Multi-tenancy         | Partial  | Needs workspace logic |
| Subscription billing  | Full     | Stripe integration    |
| Database (PostgreSQL) | Full     | Supabase              |
| File uploads          | None     | Need to add           |

### Coverage Score

- **Full Coverage:** X items
- **Partial Coverage:** X items
- **Not Covered:** X items
- **Coverage Percentage:** X%

### Estimated Gap Work

- [Gap 1]: ~X hours
- [Gap 2]: ~X hours
- [Gap 3]: ~X hours
- **Total Gap Work:** ~X hours
```

### Step 3: Compare and Select

```markdown
## Starter Kit Comparison

| Criteria         | [Kit A]        | [Kit B] | [Kit C] |
| ---------------- | -------------- | ------- | ------- |
| Coverage %       | X%             | X%      | X%      |
| Tech stack fit   | Good/Fair/Poor | -       | -       |
| Documentation    | Good/Fair/Poor | -       | -       |
| Community        | Active/Limited | -       | -       |
| Price            | $X             | $X      | $X      |
| Gap work (hours) | X              | X       | X       |

### Selection Decision

**Selected Kit:** [Name]

**Reasoning:**

1. [Reason 1]
2. [Reason 2]
3. [Reason 3]

**Key Trade-offs Accepted:**

1. [Trade-off 1]
2. [Trade-off 2]
```

---

## Gap Analysis

After selecting a starter kit, document what needs to be built:

```markdown
## Development Gap Analysis

### Critical Gaps (MVP Blockers)

| Gap     | Description      | Estimated Effort | Priority |
| ------- | ---------------- | ---------------- | -------- |
| [Gap 1] | [What's missing] | X hours          | P0       |
| [Gap 2] | [What's missing] | X hours          | P0       |

### Important Gaps (Launch Blockers)

| Gap     | Description      | Estimated Effort | Priority |
| ------- | ---------------- | ---------------- | -------- |
| [Gap 1] | [What's missing] | X hours          | P1       |

### Nice-to-Have Gaps (Post-Launch)

| Gap     | Description      | Estimated Effort | Priority |
| ------- | ---------------- | ---------------- | -------- |
| [Gap 1] | [What's missing] | X hours          | P2       |

### Total Effort Estimate

- **MVP (P0):** X hours
- **Launch (P0+P1):** X hours
- **Full Feature (All):** X hours
```

---

## Evaluation Criteria Rubric

Use this to score starter kits objectively:

### Technical Fit (40%)

| Score | Criteria                                                |
| ----- | ------------------------------------------------------- |
| 10    | Perfect tech stack match, all core features covered     |
| 7-9   | Good match, minor gaps or different but compatible tech |
| 4-6   | Moderate match, significant gaps or learning curve      |
| 1-3   | Poor match, major gaps or incompatible technology       |

### Documentation Quality (20%)

| Score | Criteria                                           |
| ----- | -------------------------------------------------- |
| 10    | Excellent docs, tutorials, examples, API reference |
| 7-9   | Good docs, covers most scenarios                   |
| 4-6   | Basic docs, gaps in coverage                       |
| 1-3   | Poor or missing documentation                      |

### Community & Support (15%)

| Score | Criteria                                       |
| ----- | ---------------------------------------------- |
| 10    | Active Discord, fast response, regular updates |
| 7-9   | Good community, reasonable support             |
| 4-6   | Limited community, slow support                |
| 1-3   | No community, no support                       |

### Code Quality (15%)

| Score | Criteria                                        |
| ----- | ----------------------------------------------- |
| 10    | Clean architecture, well-tested, easy to extend |
| 7-9   | Good quality, mostly follows best practices     |
| 4-6   | Acceptable quality, some technical debt         |
| 1-3   | Poor quality, hard to understand or modify      |

### Value (10%)

| Score | Criteria                                  |
| ----- | ----------------------------------------- |
| 10    | Excellent value vs. building from scratch |
| 7-9   | Good value, saves significant time        |
| 4-6   | Moderate value, some savings              |
| 1-3   | Poor value, not worth the price           |

---

## Integration with Other Skills

**Follows from:**

- **idea-extraction** (idea-to-product) - Technical requirements
- **competitive-intel** (idea-to-product) - Feature requirements

**Feeds into:**

- **transformation-analysis** (idea-to-product) - Implementation planning
- **gtm-playbook** (marketing/05-gtm) - Launch timeline based on development scope

---

## Output Checklist

Before moving to implementation:

- [ ] Requirements extracted from strategy document
- [ ] 2-3 starter kits evaluated
- [ ] Coverage percentages calculated
- [ ] Starter kit selected with reasoning
- [ ] Gap analysis completed with effort estimates
- [ ] Priority levels assigned to gaps
- [ ] Total development effort estimated

---
