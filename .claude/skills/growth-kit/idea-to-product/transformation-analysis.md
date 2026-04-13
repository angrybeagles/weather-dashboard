# Transformation Analysis

Deep analysis of how to transform a starter kit into your specific product. This skill produces implementation-ready technical plans across 7 key domains.

**Duration:** 60-90 minutes
**Output:** Comprehensive Transformation Report + Implementation Checklist

---

## When to Use

- Have selected a starter kit and documented gaps
- Ready to plan detailed implementation
- Need domain-by-domain analysis of changes required
- Want atomic task lists for AI-assisted development

---

## The 7 Domains

Transform your starter kit by analyzing each domain:

### Domain 1: Authentication & Multi-tenancy

```markdown
## Authentication Analysis

### Current Starter Kit State

- Auth provider: [Supabase Auth, NextAuth, etc.]
- User model: [Fields provided]
- Session handling: [How it works]

### Required Changes

**User Model Modifications:**
| Field | Add/Modify/Remove | Purpose |
|-------|-------------------|---------|
| [field] | Add | [Why needed] |

**Multi-tenancy Implementation:**

- Workspace/Team model needed: Yes/No
- Isolation strategy: [RLS, schema, application]
- User-workspace relationship: [1:1, 1:many, many:many]

**Role-Based Access:**
| Role | Permissions | Implementation |
|------|-------------|----------------|
| Owner | Full access | [How to implement] |
| Admin | Manage team | [How to implement] |
| Member | Basic access | [How to implement] |

### Implementation Tasks

- [ ] Task 1: [Specific change]
- [ ] Task 2: [Specific change]
```

### Domain 2: Payments

```markdown
## Payment Integration Analysis

### Current Starter Kit State

- Payment provider: [Stripe, Polar, etc.]
- Billing models supported: [Subscription, one-time]
- Webhook handling: [Yes/No, how]

### Required Changes

**Pricing Tiers:**
| Tier | Price | Limits | Features |
|------|-------|--------|----------|
| Free | $0 | [Limits] | [Features] |
| Pro | $X/mo | [Limits] | [Features] |

**Billing Logic:**

- Upgrade/downgrade handling
- Proration rules
- Trial period: [Duration]
- Grace period: [Duration]

**Webhook Events to Handle:**

- [ ] checkout.completed
- [ ] subscription.updated
- [ ] subscription.cancelled
- [ ] payment.failed

### Implementation Tasks

- [ ] Task 1: [Specific change]
- [ ] Task 2: [Specific change]
```

### Domain 3: Database Architecture

```markdown
## Database Analysis

### Current Starter Kit Schema

[List existing tables/models]

### Required Schema Changes

**New Tables:**
| Table | Purpose | Key Fields |
|-------|---------|------------|
| [table] | [Why needed] | [Fields] |

**Modified Tables:**
| Table | Changes | Reason |
|-------|---------|--------|
| [table] | [What changes] | [Why] |

**Relationships:**
[Describe entity relationships]

**Indexes Needed:**
| Table | Column(s) | Type | Purpose |
|-------|-----------|------|---------|
| [table] | [columns] | [B-tree, etc.] | [Why] |

### Migration Strategy

- [ ] Create migrations for new tables
- [ ] Modify existing tables
- [ ] Seed required data
- [ ] Test rollback

### Implementation Tasks

- [ ] Task 1: [Specific change]
- [ ] Task 2: [Specific change]
```

### Domain 4: Background Jobs

```markdown
## Background Processing Analysis

### Current Starter Kit State

- Job framework: [node-cron, BullMQ, etc.]
- Existing jobs: [List]

### Required Jobs

**Scheduled Jobs:**
| Job | Schedule | Purpose |
|-----|----------|---------|
| [job name] | [cron] | [What it does] |

**Queue Jobs:**
| Job | Trigger | Purpose |
|-----|---------|---------|
| [job name] | [Event] | [What it does] |

**Implementation Approach:**

- Job runner location: [Same server, separate worker]
- Error handling: [Retry strategy]
- Monitoring: [How to track]

### Implementation Tasks

- [ ] Task 1: [Specific job]
- [ ] Task 2: [Specific job]
```

### Domain 5: UI Components

```markdown
## UI/UX Analysis

### Current Starter Kit Components

- Component library: [Radix, shadcn, etc.]
- Styling: [Tailwind, CSS modules, etc.]
- Existing pages: [List]

### Required UI Changes

**New Pages:**
| Page | Purpose | Components Needed |
|------|---------|-------------------|
| [page] | [Why] | [What components] |

**Modified Pages:**
| Page | Changes | Reason |
|------|---------|--------|
| [page] | [What changes] | [Why] |

**New Components:**
| Component | Purpose | Complexity |
|-----------|---------|------------|
| [component] | [Why] | Low/Medium/High |

### Design System Changes

- Colors/theme modifications
- Typography changes
- Spacing/layout patterns

### Implementation Tasks

- [ ] Task 1: [Specific component]
- [ ] Task 2: [Specific page]
```

### Domain 6: Integrations

```markdown
## Third-Party Integration Analysis

### Required Integrations

| Service   | Purpose      | API Type     | Complexity   |
| --------- | ------------ | ------------ | ------------ |
| [Service] | [Why needed] | REST/GraphQL | Low/Med/High |

### Per-Integration Details

**[Integration Name]:**

- API documentation: [URL]
- Authentication: [API key, OAuth, etc.]
- Rate limits: [Limits]
- Key endpoints: [List]
- Error handling: [Strategy]

**Environment Variables:**
| Variable | Purpose | Required |
|----------|---------|----------|
| [VAR_NAME] | [What for] | Yes/No |

### Implementation Tasks

- [ ] Task 1: [Specific integration]
- [ ] Task 2: [Specific integration]
```

### Domain 7: Gap Synthesis

```markdown
## Gap Synthesis & Prioritization

### Critical Path Items (MVP)

| Item   | Domain   | Effort  | Dependencies |
| ------ | -------- | ------- | ------------ |
| [Item] | [Domain] | X hours | [What first] |

### Implementation Order

1. **Week 1:** [Items to complete]
2. **Week 2:** [Items to complete]
3. **Week 3:** [Items to complete]

### Risk Areas

| Risk   | Impact       | Mitigation      |
| ------ | ------------ | --------------- |
| [Risk] | High/Med/Low | [How to handle] |

### Total Effort Summary

| Domain       | Hours | Complexity   |
| ------------ | ----- | ------------ |
| Auth         | X     | Low/Med/High |
| Payments     | X     | Low/Med/High |
| Database     | X     | Low/Med/High |
| Jobs         | X     | Low/Med/High |
| UI           | X     | Low/Med/High |
| Integrations | X     | Low/Med/High |
| **Total**    | **X** | -            |
```

---

## Implementation Checklist Template

```markdown
# Implementation Checklist: [Product Name]

## Phase 1: Foundation (Week 1)

- [ ] Set up development environment
- [ ] Configure starter kit
- [ ] Implement database migrations
- [ ] Set up authentication changes

## Phase 2: Core Features (Week 2)

- [ ] Implement [Core Feature 1]
- [ ] Implement [Core Feature 2]
- [ ] Set up payment integration

## Phase 3: Polish (Week 3)

- [ ] Build remaining UI components
- [ ] Implement background jobs
- [ ] Add third-party integrations
- [ ] Testing and bug fixes

## Phase 4: Launch Prep (Week 4)

- [ ] Production deployment
- [ ] Monitoring setup
- [ ] Documentation
- [ ] Launch checklist items
```

---

## Integration with Other Skills

**Follows from:**

- **starter-kit-selection** (idea-to-product) - Which kit to transform

**Feeds into:**

- **gtm-playbook** (marketing/05-gtm) - Timeline for launch planning
- **launch-checklists** (marketing/05-gtm) - Technical readiness items

---

## Output Checklist

Before starting development:

- [ ] All 7 domains analyzed
- [ ] Implementation tasks extracted
- [ ] Priority order determined
- [ ] Effort estimates validated
- [ ] Dependencies mapped
- [ ] Risk areas identified
- [ ] Implementation checklist created

---
