# Workflow T: Technical & On-Page SEO Implementation Guide

- **Purpose:** Enable agentic coding AI to programmatically audit, implement, and validate technical and on-page SEO across websites, blogs, or digital content
- **Platform:** Any agentic coding environment (Claude Code, Cursor, Windsurf, etc.)
- **Input:** Website/project codebase access
- **Output:** Implemented SEO optimizations with validation

---

## Quick Start: What Do You Need?

| Task                         | Jump To Section                                                   |
| ---------------------------- | ----------------------------------------------------------------- |
| **Full site audit**          | [Section 1: Audit Checklist](#section-1-seo-audit-checklist)      |
| **Add schema markup**        | [Section 2: Structured Data](#section-2-structured-data--json-ld) |
| **Fix meta tags**            | [Section 3: Meta Tags](#section-3-meta-tags--social)              |
| **Configure robots.txt**     | [Section 4: Crawl Control](#section-4-crawl-control)              |
| **Set up redirects**         | [Section 5: Redirects](#section-5-redirects)                      |
| **Generate sitemap**         | [Section 6: Sitemaps](#section-6-xml-sitemaps)                    |
| **Optimize images**          | [Section 7: Images](#section-7-image-optimization)                |
| **Improve Core Web Vitals**  | [Section 8: Performance](#section-8-core-web-vitals)              |
| **Framework-specific setup** | [Section 9: Frameworks](#section-9-framework-patterns)            |
| **Validate before deploy**   | [Section 10: Validation](#section-10-validation--testing)         |

---

## Section 1: SEO Audit Checklist

### Pre-Implementation Audit

Run this checklist before making changes:

```markdown
## Technical SEO Audit

### Crawlability

- [ ] robots.txt exists and is valid
- [ ] robots.txt allows important pages
- [ ] XML sitemap exists and is submitted
- [ ] No critical pages blocked by noindex
- [ ] No redirect chains (A→B→C)
- [ ] No redirect loops

### Indexability

- [ ] Canonical tags on all pages (self-referencing)
- [ ] No duplicate content without canonicals
- [ ] Hreflang set up (if multilingual)
- [ ] HTTP → HTTPS redirect in place
- [ ] www/non-www consolidated

### On-Page SEO

- [ ] Unique title tags (50-60 chars)
- [ ] Unique meta descriptions (150-160 chars)
- [ ] One H1 per page
- [ ] Logical heading hierarchy (H1→H2→H3)
- [ ] Alt text on all images
- [ ] Internal links with descriptive anchors

### Structured Data

- [ ] Organization schema on homepage
- [ ] Article/BlogPosting schema on posts
- [ ] BreadcrumbList schema site-wide
- [ ] FAQ schema where applicable
- [ ] Product schema on product pages

### Performance

- [ ] LCP < 2.5s
- [ ] INP < 200ms
- [ ] CLS < 0.1
- [ ] Images optimized (WebP/AVIF)
- [ ] Critical CSS inlined
- [ ] JS deferred/async

### Social

- [ ] OpenGraph tags on all pages
- [ ] Twitter Card tags configured
- [ ] OG images at 1200×630px
```

---

## Section 2: Structured Data / JSON-LD

### 2.1 Schema Type Decision Tree

```
What type of page?
│
├── Homepage
│   └── Organization + WebSite + BreadcrumbList
│
├── Blog Post / Article
│   └── Article (or BlogPosting/NewsArticle) + BreadcrumbList + Author
│
├── Product Page
│   └── Product + Offer + AggregateRating + BreadcrumbList
│
├── Local Business
│   └── LocalBusiness (specific subtype) + BreadcrumbList
│
├── FAQ Section
│   └── FAQPage (add to existing page schema)
│
├── How-To Guide
│   └── HowTo + BreadcrumbList
│
├── About Page
│   └── Organization or Person + BreadcrumbList
│
└── Any Page
    └── WebPage + BreadcrumbList (minimum)
```

### 2.2 Template: Article / BlogPosting

```json
{
  "@context": "https://schema.org",
  "@type": "BlogPosting",
  "@id": "{{page_url}}#article",
  "mainEntityOfPage": "{{page_url}}",
  "headline": "{{title_max_110_chars}}",
  "description": "{{meta_description}}",
  "image": ["{{image_1x1_url}}", "{{image_4x3_url}}", "{{image_16x9_url}}"],
  "datePublished": "{{iso_date_published}}",
  "dateModified": "{{iso_date_modified}}",
  "author": [
    {
      "@type": "Person",
      "@id": "{{site_url}}/#author-{{author_slug}}",
      "name": "{{author_name}}",
      "url": "{{author_page_url}}"
    }
  ],
  "publisher": {
    "@type": "Organization",
    "@id": "{{site_url}}/#organization",
    "name": "{{site_name}}",
    "logo": {
      "@type": "ImageObject",
      "url": "{{logo_url}}",
      "width": 600,
      "height": 60
    }
  }
}
```

### 2.3 Template: FAQPage

```json
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "{{question_1}}",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "{{answer_1_can_include_basic_html}}"
      }
    },
    {
      "@type": "Question",
      "name": "{{question_2}}",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "{{answer_2}}"
      }
    }
  ]
}
```

**Allowed HTML in FAQ answers:** `<h2>-<h6>`, `<br>`, `<ol>`, `<ul>`, `<li>`, `<a>`, `<p>`, `<b>`, `<strong>`, `<i>`, `<em>`

### 2.4 Template: Product

```json
{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "{{product_name}}",
  "image": ["{{product_image_url}}"],
  "description": "{{product_description}}",
  "sku": "{{sku}}",
  "brand": {
    "@type": "Brand",
    "name": "{{brand_name}}"
  },
  "aggregateRating": {
    "@type": "AggregateRating",
    "ratingValue": "{{rating}}",
    "reviewCount": "{{review_count}}"
  },
  "offers": {
    "@type": "Offer",
    "url": "{{product_url}}",
    "priceCurrency": "{{currency_code}}",
    "price": "{{price}}",
    "availability": "https://schema.org/{{InStock|OutOfStock|PreOrder}}",
    "itemCondition": "https://schema.org/NewCondition"
  }
}
```

### 2.5 Template: LocalBusiness

```json
{
  "@context": "https://schema.org",
  "@type": "{{specific_type: Restaurant|Dentist|Store|etc}}",
  "@id": "{{site_url}}/#localbusiness",
  "name": "{{business_name}}",
  "image": "{{storefront_image}}",
  "url": "{{site_url}}",
  "telephone": "{{phone}}",
  "address": {
    "@type": "PostalAddress",
    "streetAddress": "{{street}}",
    "addressLocality": "{{city}}",
    "addressRegion": "{{state}}",
    "postalCode": "{{zip}}",
    "addressCountry": "{{country_code}}"
  },
  "geo": {
    "@type": "GeoCoordinates",
    "latitude": "{{lat}}",
    "longitude": "{{lng}}"
  },
  "openingHoursSpecification": [
    {
      "@type": "OpeningHoursSpecification",
      "dayOfWeek": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
      "opens": "09:00",
      "closes": "17:00"
    }
  ],
  "priceRange": "$$"
}
```

### 2.6 Template: BreadcrumbList

```json
{
  "@context": "https://schema.org",
  "@type": "BreadcrumbList",
  "itemListElement": [
    {
      "@type": "ListItem",
      "position": 1,
      "name": "Home",
      "item": "{{site_url}}"
    },
    {
      "@type": "ListItem",
      "position": 2,
      "name": "{{category_name}}",
      "item": "{{category_url}}"
    },
    {
      "@type": "ListItem",
      "position": 3,
      "name": "{{page_title}}"
    }
  ]
}
```

### 2.7 Template: Organization (Homepage)

```json
{
  "@context": "https://schema.org",
  "@type": "Organization",
  "@id": "{{site_url}}/#organization",
  "name": "{{company_name}}",
  "url": "{{site_url}}",
  "logo": {
    "@type": "ImageObject",
    "url": "{{logo_url}}",
    "width": 600,
    "height": 60
  },
  "sameAs": ["{{twitter_url}}", "{{linkedin_url}}", "{{github_url}}"],
  "contactPoint": {
    "@type": "ContactPoint",
    "telephone": "{{phone}}",
    "contactType": "customer service"
  }
}
```

### 2.8 Template: WebSite with Search

```json
{
  "@context": "https://schema.org",
  "@type": "WebSite",
  "@id": "{{site_url}}/#website",
  "name": "{{site_name}}",
  "url": "{{site_url}}",
  "potentialAction": {
    "@type": "SearchAction",
    "target": {
      "@type": "EntryPoint",
      "urlTemplate": "{{site_url}}/search?q={search_term_string}"
    },
    "query-input": "required name=search_term_string"
  }
}
```

### 2.9 Template: SoftwareApplication

```json
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "{{app_name}}",
  "applicationCategory": "{{BusinessApplication|DeveloperApplication|DesignApplication}}",
  "operatingSystem": "Web, iOS, Android",
  "description": "{{app_description}}",
  "offers": {
    "@type": "Offer",
    "price": "{{price_or_0_for_free}}",
    "priceCurrency": "{{currency_code}}"
  },
  "aggregateRating": {
    "@type": "AggregateRating",
    "ratingValue": "{{rating}}",
    "ratingCount": "{{rating_count}}"
  },
  "screenshot": "{{screenshot_url}}",
  "featureList": "{{comma_separated_features}}"
}
```

**Use for:** SaaS product pages, app landing pages

### 2.10 Template: HowTo

```json
{
  "@context": "https://schema.org",
  "@type": "HowTo",
  "name": "{{how_to_title}}",
  "description": "{{how_to_description}}",
  "totalTime": "{{iso_duration_PT15M}}",
  "estimatedCost": {
    "@type": "MonetaryAmount",
    "currency": "{{currency}}",
    "value": "{{cost_or_0}}"
  },
  "supply": [
    {
      "@type": "HowToSupply",
      "name": "{{supply_1}}"
    }
  ],
  "tool": [
    {
      "@type": "HowToTool",
      "name": "{{tool_1}}"
    }
  ],
  "step": [
    {
      "@type": "HowToStep",
      "name": "{{step_1_title}}",
      "text": "{{step_1_instructions}}",
      "url": "{{page_url}}#step1",
      "image": "{{step_1_image_url}}"
    },
    {
      "@type": "HowToStep",
      "name": "{{step_2_title}}",
      "text": "{{step_2_instructions}}",
      "url": "{{page_url}}#step2",
      "image": "{{step_2_image_url}}"
    }
  ]
}
```

**Use for:** Tutorials, instructional content, guides with step-by-step instructions

### 2.11 Template: Event

```json
{
  "@context": "https://schema.org",
  "@type": "Event",
  "name": "{{event_name}}",
  "startDate": "{{iso_start_datetime}}",
  "endDate": "{{iso_end_datetime}}",
  "eventAttendanceMode": "https://schema.org/{{OnlineEventAttendanceMode|OfflineEventAttendanceMode|MixedEventAttendanceMode}}",
  "eventStatus": "https://schema.org/{{EventScheduled|EventPostponed|EventCancelled}}",
  "location": {
    "@type": "{{VirtualLocation|Place}}",
    "url": "{{event_url_or_venue_url}}",
    "name": "{{venue_name_if_physical}}",
    "address": {
      "@type": "PostalAddress",
      "streetAddress": "{{street}}",
      "addressLocality": "{{city}}",
      "addressRegion": "{{state}}",
      "postalCode": "{{zip}}",
      "addressCountry": "{{country}}"
    }
  },
  "image": "{{event_image_url}}",
  "description": "{{event_description}}",
  "offers": {
    "@type": "Offer",
    "url": "{{ticket_url}}",
    "price": "{{price}}",
    "priceCurrency": "{{currency}}",
    "availability": "https://schema.org/{{InStock|SoldOut|PreOrder}}",
    "validFrom": "{{iso_date_tickets_available}}"
  },
  "performer": {
    "@type": "{{Person|Organization}}",
    "name": "{{performer_name}}"
  },
  "organizer": {
    "@type": "Organization",
    "name": "{{organizer_name}}",
    "url": "{{organizer_url}}"
  }
}
```

**Use for:** Webinars, conferences, meetups, workshops

### 2.12 Template: Review / AggregateRating

```json
{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "{{product_name}}",
  "aggregateRating": {
    "@type": "AggregateRating",
    "ratingValue": "{{average_rating}}",
    "bestRating": "5",
    "worstRating": "1",
    "ratingCount": "{{total_ratings}}",
    "reviewCount": "{{total_reviews}}"
  },
  "review": [
    {
      "@type": "Review",
      "author": {
        "@type": "Person",
        "name": "{{reviewer_name}}"
      },
      "datePublished": "{{iso_review_date}}",
      "reviewRating": {
        "@type": "Rating",
        "ratingValue": "{{rating_1_to_5}}",
        "bestRating": "5"
      },
      "reviewBody": "{{review_text}}"
    }
  ]
}
```

**Important:** Self-serving reviews (reviewing your own product) are against Google guidelines. Reviews must be from real customers.

**Use for:** Product pages with customer reviews, service pages with testimonials

### 2.13 Implementation Pattern

```html
<!-- Place in <head> section -->
<script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@graph": [
      {
        /* Organization schema */
      },
      {
        /* WebSite schema */
      },
      {
        /* BreadcrumbList schema */
      },
      {
        /* Page-specific schema (Article, Product, etc.) */
      }
    ]
  }
</script>
```

### 2.14 Schema Type Selection Guide

| Page Type | Primary Schema | Secondary Schema |
|-----------|----------------|------------------|
| Homepage | Organization + WebSite | BreadcrumbList |
| Blog Post | Article/BlogPosting | BreadcrumbList, FAQPage (if FAQs) |
| Product Page | Product + Offer | BreadcrumbList, AggregateRating |
| SaaS Landing | SoftwareApplication | Organization, FAQPage |
| Tutorial | HowTo | BreadcrumbList, Article |
| Event | Event | Organization, BreadcrumbList |
| Local Business | LocalBusiness | BreadcrumbList, AggregateRating |
| About Page | Organization/Person | BreadcrumbList |
| FAQ Page | FAQPage | BreadcrumbList |

### 2.15 Schema Validation & Testing

**Validation Tools:**
- **Rich Results Test:** https://search.google.com/test/rich-results
- **Schema.org Validator:** https://validator.schema.org/
- **Search Console:** Admin → Enhancements (monitor live data)

**Common Validation Errors:**

| Error | Cause | Fix |
|-------|-------|-----|
| Missing required property | Required field not included | Add the required property |
| Invalid date format | Date not in ISO 8601 | Use format: YYYY-MM-DDTHH:MM:SS+TZ |
| Invalid URL | URL malformed or relative | Use fully qualified URLs |
| Schema doesn't match content | Schema claims don't match page | Ensure schema reflects actual content |
| Ratings without reviews shown | AggregateRating but no visible reviews | Show reviews on page or remove schema |

**Testing Checklist:**
- [ ] Validates without errors in Rich Results Test
- [ ] No warnings in validator
- [ ] Schema content matches visible page content
- [ ] All required properties included
- [ ] URLs are absolute (not relative)
- [ ] Dates in ISO 8601 format
- [ ] Images accessible and correct dimensions

---

## Section 3: Meta Tags & Social

### 3.1 Character Limits Reference

| Element             | Limit         | Notes                          |
| ------------------- | ------------- | ------------------------------ |
| Title tag           | 50-60 chars   | 580px desktop display          |
| Meta description    | 150-160 chars | 920px desktop, 680px mobile    |
| URL slug            | 25-30 chars   | 3-5 words, hyphens, lowercase  |
| H1                  | 60-70 chars   | Should align with title intent |
| Alt text            | 125 chars     | Screen readers may truncate    |
| OG title            | 60-70 chars   | Platform display limits        |
| OG description      | 155-200 chars |                                |
| Twitter description | 200 chars max |                                |

### 3.2 URL Structure Best Practices

**Optimal URL Format:**

```
https://example.com/category/keyword-rich-slug
```

**Guidelines:**

| Rule       | Good                    | Bad              |
| ---------- | ----------------------- | ---------------- |
| Length     | 3-5 words (25-30 chars) | 10+ words        |
| Separator  | hyphens (-)             | underscores (\_) |
| Case       | lowercase               | Mixed Case       |
| Stop words | Remove (the, and, of)   | Include all      |
| Parameters | Avoid (?id=123)         | Rely on them     |

**Examples:**

```
✅ /blog/seo-checklist-2025
✅ /tools/keyword-research
✅ /pricing

❌ /blog/the-ultimate-guide-to-seo-for-beginners-in-2025
❌ /blog/post_123_final_v2
❌ /Blog/SEO-Tools
❌ /p?id=456&category=seo
```

**URL Change Protocol:**

1. Create 301 redirect from old URL to new
2. Update all internal links
3. Update XML sitemap
4. Monitor Search Console for crawl errors

---

### 3.3 Complete Head Template

```html
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />

  <!-- Primary SEO -->
  <title>{{primary_keyword}} - {{benefit}} | {{brand}}</title>
  <meta
    name="description"
    content="{{compelling_description_with_keyword_and_cta}}"
  />
  <meta name="robots" content="index, follow" />
  <link rel="canonical" href="{{canonical_url}}" />

  <!-- Open Graph -->
  <meta property="og:type" content="{{website|article|product}}" />
  <meta property="og:url" content="{{page_url}}" />
  <meta property="og:title" content="{{og_title}}" />
  <meta property="og:description" content="{{og_description}}" />
  <meta property="og:image" content="{{og_image_1200x630}}" />
  <meta property="og:image:width" content="1200" />
  <meta property="og:image:height" content="630" />
  <meta property="og:site_name" content="{{site_name}}" />

  <!-- Twitter Card -->
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:site" content="@{{twitter_handle}}" />
  <meta name="twitter:title" content="{{twitter_title}}" />
  <meta name="twitter:description" content="{{twitter_description}}" />
  <meta name="twitter:image" content="{{twitter_image_1200x675}}" />

  <!-- Article-specific (for blog posts) -->
  <meta property="article:published_time" content="{{iso_date}}" />
  <meta property="article:modified_time" content="{{iso_date}}" />
  <meta property="article:author" content="{{author_url}}" />

  <!-- Performance hints -->
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="dns-prefetch" href="//cdn.example.com" />

  <!-- Structured Data -->
  <script type="application/ld+json">
    {
      /* Schema JSON-LD */
    }
  </script>
</head>
```

### 3.3 Meta Robots Directives

```html
<!-- Standard indexable page -->
<meta name="robots" content="index, follow" />

<!-- No index, no follow (private pages) -->
<meta name="robots" content="noindex, nofollow" />

<!-- Index but don't show cached version -->
<meta name="robots" content="index, follow, noarchive" />

<!-- Control snippet length -->
<meta name="robots" content="max-snippet:160, max-image-preview:large" />

<!-- Remove after date -->
<meta name="robots" content="unavailable_after: 2025-12-31" />

<!-- Bot-specific -->
<meta name="googlebot" content="noindex" />
<meta name="bingbot" content="noindex" />
```

### 3.4 Image Requirements for Social

| Platform        | Dimensions           | Aspect Ratio | Max Size |
| --------------- | -------------------- | ------------ | -------- |
| OpenGraph       | 1200×630             | 1.91:1       | 8MB      |
| Twitter Large   | 1200×675             | 1.91:1       | 5MB      |
| Twitter Summary | 144×144 to 4096×4096 | 1:1          | 5MB      |
| LinkedIn        | 1200×627             | 1.91:1       | 5MB      |

### 3.5 Validation Tools

- Facebook: https://developers.facebook.com/tools/debug/
- Twitter: https://cards-dev.twitter.com/validator
- LinkedIn: https://www.linkedin.com/post-inspector/

---

## Section 4: Crawl Control

### 4.1 robots.txt Standard Template

```
# robots.txt for {{domain}}
# Last updated: {{date}}

# Default rules
User-agent: *
Allow: /
Disallow: /admin/
Disallow: /login/
Disallow: /private/
Disallow: /api/
Disallow: /search?
Disallow: /*?utm_
Disallow: /*?ref=
Disallow: /*?session=

# Sitemap
Sitemap: https://{{domain}}/sitemap.xml
```

### 4.2 E-commerce robots.txt

```
User-agent: *
Allow: /
Disallow: /cart/
Disallow: /checkout/
Disallow: /account/
Disallow: /wishlist/
Disallow: /compare/
Disallow: /search?
Disallow: /*?sort=
Disallow: /*?filter=
Disallow: /*?page=
Disallow: /admin/
Disallow: /api/

Crawl-delay: 1

Sitemap: https://{{domain}}/sitemap.xml
Sitemap: https://{{domain}}/sitemap-products.xml
```

### 4.3 AI Crawler Directives (2025)

```
# Block AI training crawlers
User-agent: GPTBot
Disallow: /

User-agent: ChatGPT-User
Disallow: /

User-agent: CCBot
Disallow: /

User-agent: ClaudeBot
Disallow: /

User-agent: anthropic-ai
Disallow: /

User-agent: Google-Extended
Disallow: /

User-agent: PerplexityBot
Disallow: /

User-agent: Bytespider
Disallow: /

User-agent: Meta-ExternalAgent
Disallow: /
```

### 4.4 Allow Search, Block Training

```
# Allow search engines
User-agent: Googlebot
Allow: /

User-agent: Bingbot
Allow: /

# Block AI training only
User-agent: GPTBot
Disallow: /

User-agent: CCBot
Disallow: /

User-agent: Google-Extended
Disallow: /

# Allow AI answer retrieval
User-agent: ChatGPT-User
Allow: /

User-agent: PerplexityBot
Allow: /
```

---

## Section 5: Redirects

### 5.1 Decision Tree

```
Is content permanently moved?
├── YES → Use 301 (permanent)
│         └── Passes ~90-99% link equity
└── NO → Is it temporary?
         ├── YES → Use 302 or 307 (temporary)
         └── MAINTENANCE → Use 503 + Retry-After header

Content deleted?
├── Relevant replacement exists → 301 to replacement
└── No replacement → 410 Gone (faster deindexing than 404)
```

### 5.2 Apache .htaccess

```apache
RewriteEngine On

# Single page redirect
Redirect 301 /old-page https://example.com/new-page/

# Pattern redirect
RewriteRule ^old-directory/(.*)$ /new-directory/$1 [R=301,L]

# HTTP to HTTPS
RewriteCond %{HTTPS} off
RewriteRule ^(.*)$ https://%{HTTP_HOST}%{REQUEST_URI} [R=301,L]

# non-www to www
RewriteCond %{HTTP_HOST} !^www\. [NC]
RewriteRule ^(.*)$ https://www.%{HTTP_HOST}/$1 [R=301,L]

# www to non-www
RewriteCond %{HTTP_HOST} ^www\.(.*)$ [NC]
RewriteRule ^(.*)$ https://%1/$1 [R=301,L]

# Remove trailing slashes
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule ^(.*)/$ /$1 [R=301,L]
```

### 5.3 Nginx

```nginx
server {
    listen 80;
    server_name example.com www.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name example.com www.example.com;

    # www to non-www
    if ($host = www.example.com) {
        return 301 https://example.com$request_uri;
    }

    # Single page redirect
    location = /old-page {
        return 301 /new-page;
    }

    # Directory redirect
    location /old-directory/ {
        rewrite ^/old-directory/(.*)$ /new-directory/$1 permanent;
    }
}
```

### 5.4 Next.js (next.config.js)

```javascript
module.exports = {
  async redirects() {
    return [
      {
        source: "/old-page",
        destination: "/new-page",
        permanent: true,
      },
      {
        source: "/blog/:slug",
        destination: "/posts/:slug",
        permanent: true,
      },
      {
        source: "/old-blog/:slug(\\d{1,})",
        destination: "/news/:slug",
        permanent: true,
      },
    ];
  },
};
```

### 5.5 Vercel (vercel.json)

```json
{
  "redirects": [
    {
      "source": "/old-page",
      "destination": "/new-page",
      "permanent": true
    },
    {
      "source": "/blog/:path*",
      "destination": "/posts/:path*",
      "permanent": true
    }
  ]
}
```

### 5.6 Netlify (\_redirects)

```
/old-page    /new-page    301
/blog/*      /posts/:splat    301
/twitter     https://twitter.com/handle    302
```

---

## Section 6: XML Sitemaps

### 6.1 Format Specification

```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/</loc>
    <lastmod>2025-12-20</lastmod>
  </url>
  <url>
    <loc>https://example.com/about/</loc>
    <lastmod>2025-11-15</lastmod>
  </url>
</urlset>
```

**Note:** Google ignores `<changefreq>` and `<priority>`. Only use `<loc>` and `<lastmod>`.

### 6.2 Sitemap Index (for large sites)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap>
    <loc>https://example.com/sitemap-pages.xml</loc>
    <lastmod>2025-12-20</lastmod>
  </sitemap>
  <sitemap>
    <loc>https://example.com/sitemap-posts.xml</loc>
    <lastmod>2025-12-19</lastmod>
  </sitemap>
  <sitemap>
    <loc>https://example.com/sitemap-products.xml</loc>
    <lastmod>2025-12-20</lastmod>
  </sitemap>
</sitemapindex>
```

### 6.3 Next.js Dynamic Sitemap

```javascript
// app/sitemap.js (App Router)
export default async function sitemap() {
  const baseUrl = "https://example.com";

  // Static pages
  const staticPages = [
    { url: baseUrl, lastModified: new Date() },
    { url: `${baseUrl}/about`, lastModified: new Date() },
  ];

  // Dynamic pages from database/CMS
  const posts = await getPosts();
  const postUrls = posts.map((post) => ({
    url: `${baseUrl}/blog/${post.slug}`,
    lastModified: new Date(post.updatedAt),
  }));

  return [...staticPages, ...postUrls];
}
```

### 6.4 Python Sitemap Generator

```python
import xml.etree.ElementTree as ET
from datetime import datetime

def generate_sitemap(urls, output_path='sitemap.xml'):
    """Generate XML sitemap from list of URL dicts."""

    urlset = ET.Element('urlset')
    urlset.set('xmlns', 'http://www.sitemaps.org/schemas/sitemap/0.9')

    for url_data in urls:
        url_elem = ET.SubElement(urlset, 'url')

        loc = ET.SubElement(url_elem, 'loc')
        loc.text = url_data['url']

        if 'lastmod' in url_data:
            lastmod = ET.SubElement(url_elem, 'lastmod')
            lastmod.text = url_data['lastmod']

    tree = ET.ElementTree(urlset)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding='UTF-8', xml_declaration=True)

# Usage
urls = [
    {'url': 'https://example.com/', 'lastmod': '2025-12-20'},
    {'url': 'https://example.com/about/', 'lastmod': '2025-11-15'},
]
generate_sitemap(urls)
```

---

## Section 7: Image Optimization

### 7.1 File Naming Convention

```
[primary-keyword]-[descriptor]-[variant].[extension]

Examples:
- running-shoes-mens-black.webp
- seo-checklist-infographic-2025.png
- product-hero-front-view.avif
```

**Rules:**

- Lowercase only
- Hyphens between words (not underscores)
- Descriptive, keyword-rich
- No generic names (IMG_001, photo, image)

### 7.2 Format Conversion Commands

**ImageMagick:**

```bash
# Convert to WebP
convert input.jpg -quality 80 output.webp

# Resize and convert
convert input.jpg -resize 1200x800 -quality 80 output.webp

# Batch convert
for f in *.jpg; do convert "$f" -quality 80 "${f%.jpg}.webp"; done
```

**cwebp (Google):**

```bash
# Basic conversion
cwebp -q 80 input.png -o output.webp

# With resize
cwebp -resize 1200 0 -q 80 input.jpg -o output.webp
```

**Sharp (Node.js):**

```javascript
const sharp = require("sharp");

// Convert to WebP
await sharp("input.jpg").webp({ quality: 80 }).toFile("output.webp");

// Generate multiple sizes
const sizes = [400, 800, 1200, 1600];
for (const width of sizes) {
  await sharp("input.jpg")
    .resize(width)
    .webp({ quality: 80 })
    .toFile(`output-${width}w.webp`);
}
```

### 7.3 Responsive Images Pattern

```html
<picture>
  <!-- AVIF for modern browsers -->
  <source
    type="image/avif"
    srcset="image-400w.avif 400w, image-800w.avif 800w, image-1200w.avif 1200w"
    sizes="(max-width: 600px) 100vw, 50vw"
  />

  <!-- WebP fallback -->
  <source
    type="image/webp"
    srcset="image-400w.webp 400w, image-800w.webp 800w, image-1200w.webp 1200w"
    sizes="(max-width: 600px) 100vw, 50vw"
  />

  <!-- JPEG fallback -->
  <img
    src="image-800w.jpg"
    srcset="image-400w.jpg 400w, image-800w.jpg 800w, image-1200w.jpg 1200w"
    sizes="(max-width: 600px) 100vw, 50vw"
    alt="Descriptive alt text"
    width="1200"
    height="800"
    loading="lazy"
    decoding="async"
  />
</picture>
```

### 7.4 Lazy Loading

```html
<!-- Native lazy loading (recommended) -->
<img
  src="image.jpg"
  alt="Description"
  loading="lazy"
  width="800"
  height="600"
/>

<!-- Eager for above-the-fold -->
<img src="hero.jpg" alt="Hero" loading="eager" fetchpriority="high" />
```

### 7.5 Alt Text Rules

**Good:**

```html
<img alt="Golden retriever puppy playing fetch in park" />
<img alt="2025 Tesla Model S dashboard with touchscreen" />
<img alt="Step 3: Add flour to mixing bowl" />
```

**Bad:**

```html
<img alt="dog" />
<!-- Too vague -->
<img alt="Image of cute puppy photo" />
<!-- Redundant -->
<img alt="" />
<!-- Missing for meaningful image -->
```

**Decorative (no alt needed):**

```html
<img alt="" role="presentation" />
```

### 7.6 Size Budgets

| Image Type    | Target Size |
| ------------- | ----------- |
| Hero image    | <200KB      |
| Content image | <100KB      |
| Thumbnail     | <30KB       |
| Icon/Logo     | <10KB       |

---

## Section 8: Core Web Vitals

### 8.1 Targets

| Metric | Good   | Needs Work | Poor   |
| ------ | ------ | ---------- | ------ |
| LCP    | <2.5s  | 2.5-4s     | >4s    |
| INP    | <200ms | 200-500ms  | >500ms |
| CLS    | <0.1   | 0.1-0.25   | >0.25  |

### 8.2 LCP Optimization

**Preload hero image:**

```html
<link rel="preload" as="image" href="/hero.webp" fetchpriority="high" />
```

**Preconnect to origins:**

```html
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://cdn.example.com" crossorigin />
```

**Inline critical CSS:**

```html
<style>
  /* Critical above-the-fold styles */
  body {
    margin: 0;
    font-family: system-ui;
  }
  .hero {
    height: 100vh;
  }
</style>
<link
  rel="stylesheet"
  href="/styles.css"
  media="print"
  onload="this.media='all'"
/>
```

### 8.3 CLS Prevention

**Always set dimensions:**

```html
<img src="image.jpg" width="800" height="600" alt="..." />
<video width="1280" height="720"></video>
<iframe width="560" height="315"></iframe>
```

**Reserve space for dynamic content:**

```css
.ad-container {
  min-height: 250px;
}
.embed-container {
  aspect-ratio: 16/9;
}
```

**Font loading strategy:**

```css
@font-face {
  font-family: "CustomFont";
  src: url("/fonts/custom.woff2") format("woff2");
  font-display: swap;
}
```

### 8.4 INP Optimization

**Defer non-critical JS:**

```html
<script src="/analytics.js" defer></script>
<script src="/non-critical.js" defer></script>
```

**Use passive event listeners:**

```javascript
document.addEventListener("scroll", handleScroll, { passive: true });
```

**Break long tasks:**

```javascript
async function processItems(items) {
  for (const item of items) {
    processItem(item);
    // Yield to main thread
    await new Promise((r) => setTimeout(r, 0));
  }
}
```

### 8.5 Performance Budget

| Resource          | Budget                   |
| ----------------- | ------------------------ |
| Total page weight | <3MB (aim <1.5MB mobile) |
| HTML              | <100KB                   |
| Critical CSS      | <14KB                    |
| Total CSS         | <100KB                   |
| Initial JS        | <200KB                   |
| Total JS          | <500KB                   |
| TTFB              | <200ms                   |

---

## Section 9: Framework Patterns

### 9.1 Next.js SEO Setup

**app/layout.js (App Router):**

```javascript
export const metadata = {
  metadataBase: new URL("https://example.com"),
  title: {
    default: "Site Name",
    template: "%s | Site Name",
  },
  description: "Site description",
  openGraph: {
    type: "website",
    siteName: "Site Name",
    images: ["/og-image.jpg"],
  },
  twitter: {
    card: "summary_large_image",
    site: "@handle",
  },
  robots: {
    index: true,
    follow: true,
  },
};
```

**Dynamic page metadata:**

```javascript
// app/blog/[slug]/page.js
export async function generateMetadata({ params }) {
  const post = await getPost(params.slug);

  return {
    title: post.title,
    description: post.excerpt,
    openGraph: {
      title: post.title,
      description: post.excerpt,
      images: [post.image],
      type: "article",
      publishedTime: post.publishedAt,
    },
  };
}
```

### 9.2 Astro SEO Setup

**src/layouts/BaseLayout.astro:**

```astro
---
import { SEO } from 'astro-seo';

const { title, description, image } = Astro.props;
---
<html>
<head>
  <SEO
    title={title}
    description={description}
    openGraph={{
      basic: {
        title: title,
        type: 'website',
        image: image,
      }
    }}
    twitter={{
      card: 'summary_large_image',
    }}
  />
</head>
```

### 9.3 Rendering Strategy Decision

```
Content type?
├── Static content (blogs, docs)
│   └── Use SSG (Static Site Generation)
│
├── Changes periodically (products, listings)
│   └── Use ISR (Incremental Static Regeneration)
│
├── Real-time/personalized
│   ├── Public + needs SEO → Use SSR
│   └── Behind login → Use CSR
│
└── Mix of static/dynamic
    └── Use Hybrid approach
```

---

## Section 10: Validation & Testing

### 10.1 Pre-Deployment Checklist

```bash
# 1. Validate robots.txt
curl https://example.com/robots.txt

# 2. Check sitemap accessibility
curl -I https://example.com/sitemap.xml

# 3. Validate HTML
npx html-validate dist/**/*.html

# 4. Check redirects
curl -I https://example.com/old-page

# 5. Test Core Web Vitals
npx lighthouse https://example.com --only-categories=performance
```

### 10.2 Tool URLs

| Tool                 | URL                                            | Purpose           |
| -------------------- | ---------------------------------------------- | ----------------- |
| Rich Results Test    | https://search.google.com/test/rich-results    | Schema validation |
| Schema Validator     | https://validator.schema.org/                  | Schema syntax     |
| PageSpeed Insights   | https://pagespeed.web.dev/                     | Performance       |
| Mobile-Friendly Test | https://search.google.com/test/mobile-friendly | Mobile UX         |
| Facebook Debugger    | https://developers.facebook.com/tools/debug/   | OG validation     |
| Twitter Validator    | https://cards-dev.twitter.com/validator        | Twitter cards     |

### 10.3 Orphan Page Detection (Python)

```python
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def find_orphans(site_url, sitemap_url, max_pages=500):
    """Find pages in sitemap but not linked internally."""

    # Get sitemap URLs
    sitemap_response = requests.get(sitemap_url)
    sitemap_soup = BeautifulSoup(sitemap_response.text, 'xml')
    sitemap_urls = {loc.text.strip() for loc in sitemap_soup.find_all('loc')}

    # Crawl site for internal links
    domain = urlparse(site_url).netloc
    visited = set()
    to_visit = {site_url}
    linked = set()

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited:
            continue

        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            visited.add(url)

            for a in soup.find_all('a', href=True):
                link = urljoin(url, a['href'])
                parsed = urlparse(link)
                if parsed.netloc == domain:
                    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    linked.add(clean)
                    if clean not in visited:
                        to_visit.add(clean)
        except:
            continue

    # Orphans = in sitemap but not linked
    orphans = sitemap_urls - linked
    return orphans

# Usage
orphans = find_orphans('https://example.com', 'https://example.com/sitemap.xml')
print(f"Found {len(orphans)} orphan pages")
```

### 10.4 Bulk Meta Tag Audit

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

def audit_meta_tags(urls):
    """Audit meta tags for list of URLs."""
    results = []

    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            title = soup.find('title')
            meta_desc = soup.find('meta', {'name': 'description'})
            canonical = soup.find('link', {'rel': 'canonical'})
            h1 = soup.find('h1')

            results.append({
                'url': url,
                'title': title.text if title else '',
                'title_length': len(title.text) if title else 0,
                'description': meta_desc['content'] if meta_desc else '',
                'desc_length': len(meta_desc['content']) if meta_desc else 0,
                'canonical': canonical['href'] if canonical else '',
                'h1': h1.text.strip() if h1 else '',
                'status': response.status_code,
            })
        except Exception as e:
            results.append({'url': url, 'error': str(e)})

    return pd.DataFrame(results)

# Usage
df = audit_meta_tags(['https://example.com/', 'https://example.com/about/'])
df.to_csv('meta_audit.csv', index=False)
```

---

## Section 11: Canonical Tags

### 11.1 Decision Tree

```
Duplicate/similar content on multiple URLs?
├── NO → Add self-referencing canonical
└── YES →
    ├── Both should be accessible → Use canonical tag
    ├── Only one should exist → Use 301 redirect
    └── Cross-domain duplicate → Cross-domain canonical OK
```

### 11.2 Patterns

**Self-referencing (default for all pages):**

```html
<link rel="canonical" href="https://example.com/current-page/" />
```

**Parameter handling:**

```html
<!-- On /products/item?color=red -->
<link rel="canonical" href="https://example.com/products/item/" />
```

**Cross-domain:**

```html
<!-- On syndicated content at partner-site.com -->
<link rel="canonical" href="https://original-site.com/article/" />
```

### 11.3 Common Mistakes

| Mistake                      | Fix                 |
| ---------------------------- | ------------------- |
| Canonical to redirecting URL | Point to final URL  |
| Canonical to 404 page        | Point to valid page |
| Multiple canonical tags      | Use only one        |
| Relative URL                 | Use absolute URL    |
| HTTP canonical on HTTPS page | Match protocol      |

---

## Section 12: Hreflang (Multilingual)

### 12.1 HTML Implementation

```html
<head>
  <link rel="alternate" hreflang="en-US" href="https://example.com/page" />
  <link rel="alternate" hreflang="es-ES" href="https://example.com/es/pagina" />
  <link rel="alternate" hreflang="fr-FR" href="https://example.com/fr/page" />
  <link rel="alternate" hreflang="x-default" href="https://example.com/page" />
</head>
```

### 12.2 Sitemap Implementation

```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:xhtml="http://www.w3.org/1999/xhtml">
  <url>
    <loc>https://example.com/page</loc>
    <xhtml:link rel="alternate" hreflang="en-US" href="https://example.com/page"/>
    <xhtml:link rel="alternate" hreflang="es-ES" href="https://example.com/es/pagina"/>
    <xhtml:link rel="alternate" hreflang="x-default" href="https://example.com/page"/>
  </url>
</urlset>
```

### 12.3 Common Language Codes

| Code      | Language/Region     |
| --------- | ------------------- |
| en        | English (all)       |
| en-US     | English (US)        |
| en-GB     | English (UK)        |
| es        | Spanish (all)       |
| es-ES     | Spanish (Spain)     |
| es-MX     | Spanish (Mexico)    |
| fr        | French (all)        |
| de        | German (all)        |
| zh-CN     | Chinese Simplified  |
| zh-TW     | Chinese Traditional |
| x-default | Fallback            |

### 12.4 Critical Rule

Every page MUST link to all versions including itself:

```
Page A (en) → links to A, B, C
Page B (es) → links to A, B, C
Page C (fr) → links to A, B, C
```

---

## Section 13: HTTP Status Codes

### 13.1 Quick Reference

| Code | Name               | SEO Impact               | Use Case               |
| ---- | ------------------ | ------------------------ | ---------------------- |
| 200  | OK                 | Indexable                | Normal pages           |
| 301  | Moved Permanently  | Passes 90-99% equity     | Permanent moves        |
| 302  | Found              | No equity transfer       | Temporary redirects    |
| 304  | Not Modified       | Saves crawl budget       | Cached unchanged       |
| 307  | Temporary Redirect | Preserves method         | Temporary, method-safe |
| 308  | Permanent Redirect | Preserves method         | Permanent, method-safe |
| 404  | Not Found          | Deindexed over time      | Missing content        |
| 410  | Gone               | Faster deindexing        | Permanently removed    |
| 500  | Server Error       | Crawl issues             | Fix immediately        |
| 503  | Unavailable        | Expected for maintenance | Use Retry-After        |

### 13.2 Soft 404 Detection

A soft 404 occurs when a page returns 200 but displays "not found" content.

**Indicators:**

- Page contains "not found", "404", "doesn't exist"
- Very little content (<500 characters)
- Title contains "404"

**Fix:** Return proper 404 status code for missing content.

---

## Section 14: Semantic HTML

### 14.1 Element Reference

| Element     | Purpose              | Use For              |
| ----------- | -------------------- | -------------------- |
| `<main>`    | Primary content      | ONE per page         |
| `<article>` | Standalone content   | Blog posts, news     |
| `<section>` | Thematic grouping    | Requires heading     |
| `<nav>`     | Navigation           | Menus, breadcrumbs   |
| `<aside>`   | Related content      | Sidebars             |
| `<header>`  | Intro content        | Site/article header  |
| `<footer>`  | Closing content      | Copyright, links     |
| `<figure>`  | Self-contained media | Images with captions |
| `<time>`    | Date/time            | Use datetime attr    |

### 14.2 Article Template

```html
<main>
  <article>
    <header>
      <h1>Article Title</h1>
      <p>By <address><a rel="author" href="/author">Name</a></address>
         on <time datetime="2025-12-20">Dec 20, 2025</time></p>
    </header>

    <figure>
      <img src="/image.jpg" alt="Description" width="1200" height="630">
      <figcaption>Image caption</figcaption>
    </figure>

    <section>
      <h2>Section Heading</h2>
      <p>Content...</p>
    </section>

    <footer>
      <p>Tags: <a href="/tag/seo">SEO</a></p>
    </footer>
  </article>

  <aside>
    <h2>Related Articles</h2>
    <ul>...</ul>
  </aside>
</main>
```

---

## Section 15: Anchor Text Distribution

### 15.1 Recommended Ratios (External Links)

**Homepage:**

```
Branded/URL:      80-95%
Generic:           5-15%
Partial match:     1-5%
Exact match:       0-2%
```

**Inner Pages:**

```
Branded:          35-50%
Naked URL:        10-20%
Generic:          10-15%
Partial match:    15-25%
Exact match:       5-10%
```

### 15.2 Internal Links (More Flexible)

```
Descriptive/contextual:  60-70%
Exact/partial match:     20-30%
Generic:                 10-15%
```

### 15.3 Warning Signs

| Signal                          | Risk                         |
| ------------------------------- | ---------------------------- |
| >40% exact match                | HIGH - diversify immediately |
| Same anchor from multiple sites | HIGH - vary anchors          |
| No branded anchors              | MEDIUM - add brand links     |
| No generic anchors              | LOW - add for naturalness    |

---

## Appendix A: File Size Quick Reference

| Resource       | Target |
| -------------- | ------ |
| Total page     | <3MB   |
| HTML           | <100KB |
| CSS (critical) | <14KB  |
| CSS (total)    | <100KB |
| JS (initial)   | <200KB |
| JS (total)     | <500KB |
| Hero image     | <200KB |
| Content image  | <100KB |
| Thumbnail      | <30KB  |

---

## Appendix B: Integration with SEO Workflows

### From Workflow A (Audit)

Workflow A identifies technical issues → Use this guide to implement fixes.

### From Workflow R (Rank)

Workflow R recommends on-page improvements → Use Section 3 (Meta Tags) and Section 2 (Schema).

### For New Sites

1. Run Section 1 Audit Checklist
2. Implement Section 4 (robots.txt) + Section 6 (Sitemap)
3. Set up Section 9 (Framework patterns)
4. Add Section 2 (Schema) site-wide
5. Validate with Section 10

---

## Appendix C: Validation Commands

```bash
# Lighthouse full audit
npx lighthouse https://example.com --output=html --output-path=./report.html

# HTML validation
npx html-validate dist/**/*.html

# Check robots.txt
curl -s https://example.com/robots.txt

# Check sitemap
curl -s https://example.com/sitemap.xml | head -50

# Check redirects
curl -I -L https://example.com/old-page

# Check canonical
curl -s https://example.com/page | grep canonical

# Check meta tags
curl -s https://example.com | grep -E "(title>|description|og:|twitter:)"
```
