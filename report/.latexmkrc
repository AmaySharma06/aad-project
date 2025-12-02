# LaTeX compilation configuration
$pdf_mode = 1;  # Use pdflatex
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
$clean_ext = "aux log out toc idx ind ilg synctex.gz fdb_latexmk fls";
