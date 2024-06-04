# Convert SVG figures to pdf_latex before submitting to arxiv

Problem: arxiv does not accept SVG figures.
So I have to update code from 
```latex
\includesvg[test.svg]
```

to 
```latex
\includegraphics[width=0.5\textwidth]{test.png}
```


or I can update code like this 
```latex
\includeinkscape[width=0.5\textwidth]{test.pdf_latex}
```

This svg to pdf_latex conversion is done by inkscape package 
if you run this compilation 
```
pdflatex -shell-escape test.tex
```

pdf_latex figures are stored in directories `svg_inkscape`

And you can use arxiv-latex-cleaner to convert
```latex
\includesvg[test.svg]
```
to 
```latex
\includeinkscape[width=0.5\textwidth]{test_svg.pdf_latex}
```
for you.

This can help you save your time manually updating source code which is very handy.

[https://github.com/google-research/arxiv-latex-cleaner](https://github.com/google-research/arxiv-latex-cleaner)






