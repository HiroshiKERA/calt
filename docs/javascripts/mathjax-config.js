window.MathJax = {
  // 1. Specify loader modules and use CDN for dynamic loading
  loader: {
    load: ['input/tex', 'output/chtml'],
    // Load MathJax core and submodules from this CDN path
    path: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/'
  },
  // 2. TeX configuration
  tex: {
    // Inline math delimiters: $…$ and \(...\)
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    // Display math delimiters: $$…$$ and \[…\]
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    // Allow escaping of characters (e.g. \$)
    processEscapes: true,
    // Enable processing of environments (\begin…\end…)
    processEnvironments: true
  },
  // 3. Set absolute CDN path for CHTML output fonts
  chtml: {
    // Use this URL for loading font files (woff)
    fontURL: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/output/chtml/fonts/woff-v2'
  },
  // 4. Process only elements with the arithmatex class
  options: {
    processHtmlClass: 'arithmatex'
  }
};

// Re-render math whenever the page content changes (e.g., navigation or live reload)
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});

