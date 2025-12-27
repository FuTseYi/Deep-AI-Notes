# Deep-AI-Notes

<div align="center">

<img src="docs/public/favicon/emoji-idea.png" alt="Deep-AI-Notes" width="25%" />

**🚀 Deeply Customized AI Learning Notes Documentation Framework Based on VitePress**

[![VitePress](https://img.shields.io/badge/VitePress-1.6.4-brightgreen.svg)](https://vitepress.dev/)
[![Vue](https://img.shields.io/badge/Vue-3.5.25-42b883.svg)](https://vuejs.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/FuTseYi/Deep-AI-Notes/pulls)

[Live Demo](https://edu.xieyi.org) | [Quick Start](#-quick-start) | [Features](#-features) | [Deployment](#-deployment-guide)

English | [简体中文](README_zh-CN.md)

</div>

---

## 📖 Introduction

**Deep-AI-Notes** is a VitePress documentation framework built and deeply optimized from scratch, specifically designed for showcasing AI learning notes. This project not only implements complete configuration logic but also provides a robust and perfect UI rendering solution, with custom Vue components that achieve practical features like reading progress circles similar to VuePress.

### 🎯 Use Cases

- 📚 **Technical Documentation**: Perfect for building tech blogs, learning notes, and course documentation
- 🎓 **Education & Training**: Ideal for online tutorials, training materials, and knowledge bases
- 📊 **Data Science**: Especially suitable for displaying AI/ML content with mathematical formulas and charts
- 🌐 **Multilingual Sites**: Supports internationalization configuration for easy multilingual documentation

---

## ✨ Features

### 🎨 Core Features

- ✅ **Mermaid Diagram Support**: Built-in flowcharts, sequence diagrams, Gantt charts, and more
- ✅ **LaTeX Math Formulas**: Full support for inline and block formulas with automatic horizontal scrolling
- ✅ **Reading Progress Circle**: VuePress-style reading progress indicator with one-click back-to-top
- ✅ **Image Viewer**: Integrated viewerjs with zoom, rotate, and navigation features
- ✅ **Timeline Component**: Elegant display of project progress and update history
- ✅ **Auto Sidebar**: Automatically generates sidebar based on file structure with manual sorting support
- ✅ **Internationalization**: Complete multilingual configuration (Chinese & English)
- ✅ **SEO Optimization**: Auto-generates sitemap, robots.txt, and Open Graph tags
- ✅ **Responsive Design**: Perfect adaptation for desktop, tablet, and mobile devices
- ✅ **Dark Mode**: Built-in dark theme with automatic system preference detection

### 🎨 UI/UX Enhancements

- 🌈 **Custom Theme Colors**: Sky blue (#87CEFA) brand color with dark mode support
- 🎭 **Smooth Animations**: Carefully designed transitions for page switches, card hovers, and button interactions
- 📱 **Mobile Optimization**: Horizontal table scrolling, automatic sidebar title wrapping
- 🎯 **Custom Cards**: Homepage feature cards (About, Outcomes, Comparison, etc.)
- 🔍 **Local Search**: Built-in full-text search functionality
- 📊 **Adaptive Tables**: Auto-adjusts table width with mobile horizontal scrolling

### 🛠️ Developer Experience

- ⚡ **Fast Hot Reload**: Based on Vite with second-level response in development
- 📦 **Automated Build**: One-click production build with automatic SEO file generation
- 🔧 **Flexible Configuration**: Unified site configuration management, easy to maintain
- 🌐 **LAN Access**: Development server supports LAN access for multi-device testing
- 📝 **TypeScript Support**: Configuration files use TypeScript for type safety

---

## 🚀 Quick Start

### Requirements

- **Node.js**: >= 18.0.0
- **Package Manager**: pnpm recommended (npm and yarn also supported)

### Installation Steps

#### 1. Fork This Repository

Click the **Fork** button in the top right corner to fork the project to your GitHub account.

#### 2. Clone to Local

```bash
# Using HTTPS
git clone https://github.com/YOUR_USERNAME/Deep-AI-Notes.git

# Or using SSH
git clone git@github.com:YOUR_USERNAME/Deep-AI-Notes.git

# Enter project directory
cd Deep-AI-Notes
```

#### 3. Install Dependencies

```bash
# Using pnpm (recommended)
pnpm install

# Or using npm
npm install

# Or using yarn
yarn install
```

#### 4. Start Development Server

```bash
# Using pnpm
pnpm dev

# Or using npm
npm run dev

# Or using yarn
yarn dev
```

After successful startup, visit `http://localhost:5173` to preview.

#### 5. Build for Production

```bash
# Using pnpm
pnpm build

# Or using npm
npm run build

# Or using yarn
yarn build
```

After building, static files will be generated in the `docs/.vitepress/dist` directory.

#### 6. Preview Production Build Locally

```bash
# Using pnpm
pnpm preview

# Or using npm
npm run preview

# Or using yarn
yarn preview
```

---

## 📁 Project Structure

```
Deep-AI-Notes/
├── docs/                          # Documentation root directory
│   ├── .vitepress/                # VitePress configuration directory
│   │   ├── config.ts              # Core configuration file (★ Important)
│   │   ├── theme/                 # Theme customization
│   │   │   ├── index.ts           # Theme entry (image viewer, reading progress)
│   │   │   ├── custom.css         # Custom styles (★ Important)
│   │   │   └── components/        # Custom components
│   │   │       └── ReadingProgress.vue  # Reading progress circle component
│   │   └── cache/                 # Build cache (auto-generated)
│   ├── en/                        # English content directory
│   │   ├── index.md               # English homepage
│   │   ├── guide/                 # Guide
│   │   └── 80-MachineLearning/    # Machine learning notes
│   ├── public/                    # Static assets
│   │   └── favicon/               # Site icons
│   └── zh/                        # Chinese content directory (optional)
├── package.json                   # Project dependencies
├── pnpm-lock.yaml                 # Dependency lock file
├── .gitignore                     # Git ignore configuration
├── LICENSE                        # Open source license
└── README.md                      # Project documentation
```

---

## ⚙️ Configuration Guide

### Core Configuration File: `docs/.vitepress/config.ts`

All configurations are centralized in the `config.ts` file with a modular design for easy understanding and modification.

#### 1. Site Basic Configuration

```typescript
const SITE_CONFIG = {
  // Site title configuration
  title: "Your Site Title",           // Browser tab title
  siteTitle: "Navbar Title",          // Top-left navbar title
  description: "Site Description",    // SEO description
  
  // SEO configuration
  url: 'https://yourdomain.com',     // Website domain (for sitemap generation)
  keywords: 'AI,Machine Learning,Deep Learning', // SEO keywords
  author: 'Your Name',                // Author information
  
  // Resource configuration
  logo: '/favicon/logo.png',         // Website logo
  favicon: {
    href: '/favicon/favicon.png',    // Site icon
    type: 'image/png'                // Icon type
  },
}
```

#### 2. Auto-Generated Sidebar

Uses `vitepress-sidebar` plugin to automatically generate sidebar:

```typescript
const commonSidebarConfig = {
  useTitleFromFileHeading: true,        // Use first heading as sidebar title
  useFolderTitleFromIndexFile: true,    // Use index.md title as folder title
  useFolderLinkFromIndexFile: true,     // Folder link points to index.md
  hyphenToSpace: true,                  // Convert hyphens to spaces
  collapsed: true,                      // Collapse submenus by default
  excludePattern: ['public', 'assets'], // Excluded directories
  manualSortFileNameByPriority: [       // Manual sort priority
    'guide',
    'MachineLearning'
  ],
}
```

#### 3. Internationalization Configuration

Supports multilingual sites, English configured by default:

```typescript
locales: {
  root: {
    label: 'English',
    lang: 'en',
    title: SITE_CONFIG.title,
    description: SITE_CONFIG.description,
    themeConfig: {
      nav: [
        { text: 'Home🏠️', link: '/' },
      ],
      sidebar: createSidebar('docs/en'),
    }
  },
  // Chinese configuration (uncomment to enable)
  // zh: { ... }
}
```

#### 4. Markdown Enhancements

```typescript
markdown: {
  math: true,              // Enable LaTeX math formulas
  lineNumbers: true,       // Show code line numbers
  languageAlias: {         // Language aliases
    'gitignore': 'ini',
    'env': 'properties'
  },
  config: (md) => {
    md.use(timeline);      // Register timeline plugin
  },
}
```

#### 5. SEO Optimization

Auto-generates sitemap, robots.txt, and Open Graph tags:

```typescript
// Auto-generate sitemap
sitemap: { hostname: SITE_CONFIG.url }

// Dynamically generate SEO meta tags for each page
transformHead: ({ pageData }) => {
  // Auto-generate canonical, og:url, og:title, etc.
}

// Auto-generate robots.txt after build
buildEnd: async (siteConfig) => {
  // Auto-write robots.txt
}
```

### Custom Styles: `docs/.vitepress/theme/custom.css`

#### 1. Theme Color Customization

```css
:root {
  /* Brand primary color - Sky blue */
  --vp-c-brand-1: #6CB8E8;
  --vp-c-brand-2: #87CEFA;
  --vp-c-brand-3: #A3D9F5;
}
```

Modify these variables to change the site-wide theme color.

#### 2. Custom Card Styles

The project includes various custom card styles:

- `.info-card`: Information card
- `.outcome-card`: Outcome card
- `.audience-card`: Audience card
- `.compare-card`: Comparison card

#### 3. Animation Effects

All page transitions, card hovers, and button interactions have carefully designed transition effects:

```css
/* Page transition animation */
.vp-doc {
  animation: slideInUp 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

/* Button hover effect */
.VPButton:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
```

### Custom Components

#### Reading Progress Circle: `ReadingProgress.vue`

VuePress-style reading progress indicator with features:

- Real-time reading progress percentage display
- Shows back-to-top arrow after 1.5 seconds of scroll stop
- One-click back to top
- Dark mode support
- Mobile responsive

---

## 📝 Content Writing Guide

### 1. Create New Page

Create Markdown files in the `docs/en/` directory:

```markdown
---
title: Page Title (for SEO)
description: Page description (for SEO)
lastUpdated: true  # Show last updated time
---

# Page Title

Page content...
```

### 2. Using Math Formulas

```markdown
Inline formula: $E = mc^2$

Block formula:
$$
\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$
```

### 3. Using Mermaid Diagrams

````markdown
```mermaid
graph LR
    A[Start] --> B[Process]
    B --> C[End]
```
````

### 4. Using Timeline

```markdown
::: timeline 2025-12-24
#### Version 1
Completed transition from docsify to vitepress
:::

::: timeline 2025-12-25
#### Version 2
Added more features
:::
```

### 5. Inserting Images

```markdown
![Image Description](./assets/image.png)
```

Images automatically support click-to-zoom functionality.

### 6. Using Custom Cards

```html
<div class="info-card">
  <div class="info-card-content">
    <div class="audience-title">Card Title</div>
    <div class="info-card-desc">
      Card content description
    </div>
  </div>
</div>
```

---

## 🌐 Deployment Guide

For deployment methods, please refer to the [VitePress Official Deployment Guide](https://vitepress.dev/guide/deploy).

---

## 🔧 Troubleshooting

### 1. Dependency Installation Failed

**Issue**: `pnpm install` errors

**Solution**:
```bash
# Clear cache
pnpm store prune

# Remove node_modules and lock file
rm -rf node_modules pnpm-lock.yaml

# Reinstall
pnpm install
```

### 2. Mermaid Diagrams Not Displaying

**Issue**: Mermaid diagrams don't show in production

**Solution**: Ensure SSR options are configured in `config.ts`:
```typescript
vite: {
  ssr: {
    noExternal: ['vitepress-plugin-mermaid', 'mermaid'],
  },
}
```

### 3. Math Formulas Display Issues

**Issue**: LaTeX formulas render incorrectly or don't display

**Solution**:
1. Ensure `markdown.math: true` is enabled
2. Check formula syntax is correct
3. Use `$$` for block formulas, `$` for inline formulas

### 4. Image Viewer Not Working

**Issue**: Clicking images doesn't trigger viewer

**Solution**: Ensure `imageViewer` is correctly configured in `theme/index.ts`:
```typescript
import imageViewer from 'vitepress-plugin-image-viewer'
import 'viewerjs/dist/viewer.min.css'

setup() {
  const route = useRoute()
  imageViewer(route, '.vp-doc', {
    toolbar: {
      zoomIn: 4,
      zoomOut: 4,
      oneToOne: 4,
      reset: 4,
      prev: 4,
      next: 4,
    },
  })
}
```

### 5. Sidebar Not Displaying

**Issue**: Sidebar is empty or displays incorrectly

**Solution**:
1. Check file structure is correct
2. Ensure Markdown files have a level-1 heading `# Title`
3. Check if `excludePattern` excludes target directory

### 6. Development Server Inaccessible

**Issue**: Cannot access `localhost:5173` after `pnpm dev`

**Solution**:
1. Check if port is occupied
2. Try specifying a different port:
   ```typescript
   vite: {
     server: {
       port: 3000,
     }
   }
   ```

---

## 🤝 Contributing

All forms of contribution are welcome! Whether reporting bugs, suggesting new features, or submitting code improvements.

### Contribution Process

1. **Fork this repository**
2. **Create feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to branch**: `git push origin feature/AmazingFeature`
5. **Submit Pull Request**

### Code Standards

- Use TypeScript for configuration files
- Follow Vue 3 Composition API conventions
- Use BEM naming convention for CSS
- Follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

This project is built upon the following excellent open-source projects:

- [VitePress](https://vitepress.dev/) - Vite & Vue powered static site generator
- [Vue.js](https://vuejs.org/) - Progressive JavaScript framework
- [Mermaid](https://mermaid.js.org/) - Text-based diagram generation tool
- [MathJax](https://www.mathjax.org/) - Math formula rendering engine
- [viewerjs](https://github.com/fengyuanchen/viewerjs) - Powerful image viewer
- [vitepress-plugin-image-viewer](https://github.com/T-miracle/vitepress-plugin-image-viewer) - VitePress image viewer plugin
- [vitepress-sidebar](https://github.com/jooy2/vitepress-sidebar) - Auto sidebar generation
- [vitepress-plugin-mermaid](https://github.com/emersonbottero/vitepress-plugin-mermaid) - Mermaid integration plugin
- [vitepress-markdown-timeline](https://github.com/HanochMa/vitepress-markdown-timeline) - Timeline component

Special thanks to [Datawhale](https://www.datawhale.cn/) open-source learning platform for providing learning resources.

---

## 📧 Contact

- **Author**: 謝懿Shine
- **Email**: tseyi.wk@icloud.com
- **GitHub**: [@FuTseYi](https://github.com/FuTseYi)
- **Blog**: [https://xieyi.org](https://xieyi.org)

---

## ⭐ Star History

If this project helps you, please give it a Star ⭐️!

![Star History Chart](https://api.star-history.com/svg?repos=FuTseYi/Deep-AI-Notes&type=Date)

---

<div align="center">

**[⬆ Back to Top](#deep-ai-notes)**

Made with ❤️ by [謝懿Shine](https://github.com/FuTseYi)

</div>

