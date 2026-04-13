# Signup Flow CRO

## Overview

Optimize signup, registration, account creation, and trial activation flows to reduce friction, increase completion rates, and set users up for successful activation.

## When to Use

- "Signup conversions", "registration friction"
- "Signup form optimization", "free trial signup"
- "Reduce signup dropoff", "account creation flow"
- For post-signup onboarding, use `onboarding-cro.md`
- For lead capture forms (not account creation), use `form-cro.md`

## Inputs Required

- Current signup flow (steps/screens)
- Current completion rate
- Field-level drop-off data (if available)
- What happens immediately after signup

---

## Core Principles

### 1. Minimize Required Fields

Every field reduces conversion. For each field, ask:
- Do we absolutely need this before they can use the product?
- Can we collect this later through progressive profiling?
- Can we infer this from other data?

**Field priority:**
- Essential: Email (or phone), Password
- Often needed: Name
- Usually deferrable: Company, Role, Team size, Phone, Address

### 2. Show Value Before Asking for Commitment

- What can you show/give before requiring signup?
- Can they experience the product before creating an account?
- Reverse the order: value first, signup second

### 3. Reduce Perceived Effort

- Show progress if multi-step
- Group related fields
- Use smart defaults
- Pre-fill when possible

### 4. Remove Uncertainty

- Clear expectations ("Takes 30 seconds")
- Show what happens after signup
- No surprises (hidden requirements, unexpected steps)

---

## Field-by-Field Optimization

### Email Field
- Single field (no email confirmation)
- Inline validation for format
- Check for common typos (gmial.com > gmail.com)
- Clear error messages

### Password Field
- Show password toggle (eye icon)
- Show requirements upfront, not after failure
- Consider passphrase hints for strength
- Update requirement indicators in real-time
- Allow paste (don't disable)
- Show strength meter instead of rigid rules
- Consider passwordless options

### Name Field
- Single "Full name" vs. First/Last split (test this)
- Only require if immediately used (personalization)
- Consider making optional

### Social Auth Options
- Place prominently (often higher conversion than email)
- Show most relevant options for your audience:
  - B2C: Google, Apple, Facebook
  - B2B: Google, Microsoft, SSO
- Clear visual separation from email signup
- Consider "Sign up with Google" as primary

### Phone Number
- Defer unless essential (SMS verification, calling leads)
- If required, explain why
- Use proper input type with country code handling

### Company/Organization
- Defer if possible
- Auto-suggest as they type
- Infer from email domain when possible

---

## Single-Step vs. Multi-Step

### Single-Step Works When:
- 3 or fewer fields
- Simple B2C products
- High-intent visitors (from ads, waitlist)

### Multi-Step Works When:
- More than 3-4 fields needed
- Complex B2B products needing segmentation
- You need different types of info

### Multi-Step Best Practices
- Show progress indicator
- Lead with easy questions (name, email)
- Put harder questions later (after psychological commitment)
- Each step should feel completable in seconds
- Allow back navigation
- Save progress (don't lose on refresh)

### Progressive Commitment Pattern
1. Email only (lowest barrier)
2. Password + name
3. Customization questions (optional)

---

## Trust and Friction Reduction

### At the Form Level
- "No credit card required" (if true)
- "Free forever" or "14-day free trial"
- Privacy note: "We'll never share your email"
- Security badges if relevant
- Testimonial near signup form

### Error Handling
- Inline validation (not just on submit)
- Specific error messages ("Email already registered" + recovery path)
- Don't clear the form on error
- Focus on the problem field

### Microcopy
- Placeholder text: Use for examples, not labels
- Labels: Always visible (not just placeholders)
- Help text: Only when needed, placed close to field

---

## Mobile Signup Optimization

- Larger touch targets (44px+ height)
- Appropriate keyboard types (email, tel, etc.)
- Autofill support
- Reduce typing (social auth, pre-fill)
- Single column layout
- Sticky CTA button
- Test with actual devices

---

## Post-Submit Experience

### Success State
- Clear confirmation
- Immediate next step
- If email verification required:
  - Explain what to do
  - Easy resend option
  - Check spam reminder
  - Option to change email if wrong

### Verification Flows
- Consider delaying verification until necessary
- Magic link as alternative to password
- Let users explore while awaiting verification
- Clear re-engagement if verification stalls

---

## Common Signup Flow Patterns

### B2B SaaS Trial
1. Email + Password (or Google auth)
2. Name + Company (optional: role)
3. > Onboarding flow

### B2C App
1. Google/Apple auth OR Email
2. > Product experience
3. Profile completion later

### Waitlist/Early Access
1. Email only
2. Optional: Role/use case question
3. > Waitlist confirmation

### E-commerce Account
1. Guest checkout as default
2. Account creation optional post-purchase
3. OR Social auth with single click

---

## Experiment Ideas

### Form Design
- Single-step vs. multi-step signup flow
- Multi-step with progress bar vs. without
- 1-column vs. 2-column layout
- Form embedded vs. separate signup page

### Field Optimization
- Reduce to minimum fields (email + password only)
- Add/remove phone number field
- Single "Name" field vs. "First/Last" split
- Add/remove company field
- Required vs. optional balance

### Authentication Options
- Add SSO options (Google, Microsoft, GitHub)
- SSO prominent vs. email form prominent
- Test which SSO options resonate
- SSO-only vs. SSO + email option

### Copy & Messaging
- Headline variations above signup form
- CTA text: "Create Account" vs. "Start Free Trial" vs. "Get Started"
- Trial length clarity in CTA
- Value proposition emphasis in header

### Trial & Commitment
- Credit card required vs. not required
- Trial length impact (7 vs. 14 vs. 30 days)
- Freemium vs. free trial model
- Email verification timing

---

## Measurement

### Key Metrics
- Form start rate (landed > started filling)
- Form completion rate (started > submitted)
- Field-level drop-off
- Time to complete
- Error rate by field
- Mobile vs. desktop completion

### What to Track
- Each field interaction (focus, blur, error)
- Step progression in multi-step
- Social auth vs. email signup ratio
- Time between steps

---

## Output Format

### Audit Findings
For each issue found:
- **Issue**: What's wrong
- **Impact**: Why it matters
- **Fix**: Specific recommendation
- **Priority**: High/Medium/Low

### Recommended Changes
1. Quick wins (same-day fixes)
2. High-impact changes (week-level effort)
3. Test hypotheses (things to A/B test)

### Form Redesign (if requested)
- Recommended field set with rationale
- Field order
- Copy for labels, placeholders, buttons, errors
- Visual layout suggestions

## Related Skills

- `cro/onboarding-cro.md` - For post-signup optimization
- `cro/form-cro.md` - For non-signup forms
- `cro/page-cro.md` - For landing page leading to signup
- `cro/ab-testing.md` - For testing signup flow changes
