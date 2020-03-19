#enable pdf mode
$pdf_mode = 1;
$recorder = 1;
$pdflatex = 'pdflatex --shell-escape -interaction nonstopmode -synctex=1 %O %S';

add_cus_dep('py', 'bpl_tex', 0, 'py2bpl_tex');
sub py2bpl_tex {
   return system("python \"$_[0].py\" \"latexmk\"");
}

show_cus_dep();

#add generated extensions so they are cleaned correctly
push @generated_exts, 'synctex', 'synctex.gz';
push @generated_exts, 'acn', 'acr', 'alg';
push @generated_exts, 'sbn', 'sbl', 'syml';
push @generated_exts, 'idn', 'idl', 'indx';
push @generated_exts, 'run.xml';
push @generated_exts, 'bpl_tex', 'pdf';
$clean_ext .= ' %R.ist %R.xdy';
