# Form CRO

## Overview

Optimize any form that is NOT signup/registration - including lead capture, contact forms, demo requests, applications, surveys, and checkout forms.

## When to Use

- "Form optimization", "lead form conversions"
- "Form friction", "form completion rate"
- Contact form, demo request form, quote form optimization
- For signup/registration forms, use `signup-flow-cro.md`

## Inputs Required

- Current form design/fields
- Form type and purpose
- Current completion rate (if known)
- Mobile vs. desktop traffic split

---

## Core Principles

### 1. Every Field Has a Cost

Each field reduces completion rate:
- 3 fields: Baseline
- 4-6 fields: 10-25% reduction
- 7+ fields: 25-50%+ reduction

For each field, ask:
- Is this absolutely necessary before we can help them?
- Can we get this information another way?
- Can we ask this later?

### 2. Value Must Exceed Effort

- Clear value proposition above form
- Make what they get obvious
- Reduce perceived effort

### 3. Reduce Cognitive Load

- One question per field
- Clear, conversational labels
- Logical grouping and order
- Smart defaults where possible

---

## Field-by-Field Optimization

### Email Field
- Single field, no confirmation
- Inline validation
- Typo detection (gmial.com > gmail.com?)
- Proper mobile keyboard

### Name Fields
- Test single "Name" vs. First/Last
- Single field reduces friction
- Split only if personalization requires it

### Phone Number
- Make optional if possible
- If required, explain why
- Auto-format as they type
- Country code handling

### Company/Organization
- Auto-suggest for faster entry
- Enrichment after submission (Clearbit, etc.)
- Consider inferring from email domain

### Job Title/Role
- Dropdown if categories matter
- Free text if wide variation
- Consider making optional

### Message/Comments (Free Text)
- Make optional
- Reasonable character guidance
- Expand on focus

### Dropdown Selects
- "Select one..." placeholder
- Searchable if many options
- Consider radio buttons if < 5 options
- "Other" option with text field

### Checkboxes (Multi-select)
- Clear, parallel labels
- Reasonable number of options
- "Select all that apply" instruction

---

## Form Layout Optimization

### Field Order
1. Start with easiest fields (name, email)
2. Build commitment before asking more
3. Sensitive fields last (phone, company size)
4. Logical grouping if many fields

### Labels and Placeholders

**Good:**
```
Email
[name@company.com]  <- Example as placeholder
```

**Bad:**
```
[Enter your email address]  <- Disappears on focus
```

- Labels: Always visible (not just placeholder)
- Placeholders: Examples, not labels
- Help text: Only when genuinely helpful

### Single vs. Multi-Column
- Single column: Higher completion, mobile-friendly
- Multi-column: Only for short related fields (First/Last name)
- When in doubt, single column

---

## Multi-Step Forms

### When to Use
- More than 5-6 fields
- Logically distinct sections
- Conditional paths based on answers
- Complex forms (applications, quotes)

### Best Practices
- Progress indicator (step X of Y)
- Start with easy, end with sensitive
- One topic per step
- Allow back navigation
- Save progress (don't lose on refresh)
- Clear required vs. optional indication

### Progressive Commitment Pattern
1. Low-friction start (just email)
2. More detail (name, company)
3. Qualifying questions
4. Contact preferences

---

## Error Handling

### Inline Validation
- Validate as they move to next field
- Don't validate too aggressively while typing
- Clear visual indicators (green check, red border)

### Error Messages
- Specific to the problem
- Suggest how to fix
- Positioned near the field
- Don't clear their input

**Good:** "Please enter a valid email address (e.g., name@company.com)"
**Bad:** "Invalid input"

---

## Submit Button Optimization

### Button Copy

**Weak:** "Submit" | "Send"
**Strong:** "[Action] + [What they get]"

Examples:
- "Get My Free Quote"
- "Download the Guide"
- "Request Demo"
- "Send Message"

### Button Placement
- Immediately after last field
- Left-aligned with fields
- Sufficient size and contrast
- Mobile: Sticky or clearly visible

---

## Trust & Friction Reduction

### Near the Form
- Privacy statement: "We'll never share your info"
- Security badges if collecting sensitive data
- Testimonial or social proof
- Expected response time

### Reducing Perceived Effort
- "Takes 30 seconds"
- Field count indicator
- Remove visual clutter
- Generous white space

### Addressing Objections
- "No spam, unsubscribe anytime"
- "We won't share your number"
- "No credit card required"

---

## Form Types: Specific Guidance

### Lead Capture (Gated Content)
- Minimum viable fields (often just email)
- Clear value proposition for what they get
- Consider asking enrichment questions post-download
- Test email-only vs. email + name

### Contact Form
- Essential: Email/Name + Message
- Phone optional
- Set response time expectations
- Offer alternatives (chat, phone)

### Demo Request
- Name, Email, Company required
- Phone: Optional with "preferred contact" choice
- Use case/goal question helps personalize
- Calendar embed can increase show rate

### Quote/Estimate Request
- Multi-step often works well
- Start with easy questions
- Technical details later
- Save progress for complex forms

---

## Mobile Optimization

- Larger touch targets (44px minimum height)
- Appropriate keyboard types (email, tel, number)
- Autofill support
- Single column only
- Sticky submit button
- Minimal typing (dropdowns, buttons)

---

## Measurement

### Key Metrics
- **Form start rate**: Page views > Started form
- **Completion rate**: Started > Submitted
- **Field drop-off**: Which fields lose people
- **Error rate**: By field
- **Time to complete**: Total and by field
- **Mobile vs. desktop**: Completion by device

---

## Experiment Ideas

### Form Structure
- Single-step vs. multi-step with progress bar
- 1-column vs. 2-column layout
- Form embedded vs. separate page
- Form above fold vs. after content

### Field Optimization
- Reduce to minimum viable fields
- Add/remove phone number field
- Add/remove company field
- Required vs. optional balance
- Field enrichment to auto-fill known data

### Copy & Design
- Field label clarity and length
- Placeholder text optimization
- Button text variations
- Trust elements near form
- Error message tone

### Mobile & UX
- Larger touch targets
- Appropriate keyboard types
- Sticky submit button
- Form container styling

---

## Output Format

### Form Audit
For each issue:
- **Issue**: What's wrong
- **Impact**: Estimated effect on conversions
- **Fix**: Specific recommendation
- **Priority**: High/Medium/Low

### Recommended Form Design
- **Required fields**: Justified list
- **Optional fields**: With rationale
- **Field order**: Recommended sequence
- **Copy**: Labels, placeholders, button
- **Error messages**: For each field

## Related Skills

- `cro/signup-flow-cro.md` - For account creation forms
- `cro/popup-cro.md` - For forms inside popups
- `cro/page-cro.md` - For the page containing the form
- `cro/ab-testing.md` - For testing form changes
