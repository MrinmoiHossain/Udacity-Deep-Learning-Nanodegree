### Jupyter notebooks
- a web application that allows you to combine explanatory text, math equations, code, and visualizations all in one easily sharable document

### Installation
* Using Conda: ```conda install jupyter notebook```
* Using PIP: ```pip install jupyter notebook```


### Resources
* Literate programming: http://www.literateprogramming.com/
* LIGO: https://www.ligo.caltech.edu/news/ligo20160211
* Markdown Cheatsheet: https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
* Built-in-magic-Command: https://ipython.readthedocs.io/en/stable/interactive/magics.html

### Magic Keywords
- If you want to time how long it takes for a whole cell to run, you’d use %%timeit
- On higher resolution screens such as Retina displays, the default images in notebooks can look blurry. Use %config InlineBackend.figure_format = 'retina' after %matplotlib inline to render higher resolution images.
- turn on the interactive debugger using the magic command %pdb

### Converting notebooks
```bash
jupyter nbconvert --to html notebook.ipynb
```