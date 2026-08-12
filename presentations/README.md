# Presentations

This directory contains Beamer presentations for portfolio, talks, and interviews.

## Structure

```
presentations/
├── panel_interview/          # Individual presentation projects
│   └── src/
│       ├── panel_interview.tex        # Main presentation file
│       ├── beamercolorthemeaggie.sty  # TAMU Aggie color theme
│       └── panel_interview.pdf        # Built output (not committed)
├── templates/               # Reusable Beamer templates
│   └── tamu_metropolis/
│       ├── demo.tex         # Template example/reference
│       ├── beamercolorthemeaggie.sty  # Aggie color theme
│       └── demo.pdf         # Built example
└── README.md               # This file
```

## Creating a New Presentation

1. **Create a new folder** under `presentations/`:
   ```bash
   mkdir -p presentations/{name}/src
   ```

2. **Copy the color theme** from the template:
   ```bash
   cp presentations/templates/tamu_metropolis/beamercolorthemeaggie.sty presentations/{name}/src/
   ```

3. **Create your presentation file** `presentations/{name}/src/{name}.tex`:
   ```latex
   \documentclass[10pt]{beamer}
   
   \usetheme[progressbar=frametitle]{metropolis}
   \usecolortheme{aggie}
   
   % Your presentation content here...
   ```

4. **Build with latexmk**:
   ```bash
   cd presentations/{name}/src
   latexmk -pdf -interaction=nonstopmode {name}.tex
   ```

## Theme Details

- **Beamer Theme**: Metropolis (modern, minimal design)
- **Color Theme**: Aggie (Texas A&M branded colors)
- **Font Requirement**: Fira Sans (part of Metropolis theme)
- **Build Tool**: latexmk (configured via VS Code LaTeX Workshop extension)

## Notes

- PDF files are ignored by Git (see `.gitignore`)
- Build artifacts (`.aux`, `.log`, `.fdb_latexmk`, etc.) are ignored by Git
- `.tex` files and `.sty` files are tracked
- Each presentation maintains its own copy of `beamercolorthemeaggie.sty` for portability

## Extending

To customize the Aggie colors, edit `beamercolorthemeaggie.sty` in your presentation's `src/` folder. See the template version for available colors and usage.
