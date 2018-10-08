Emukit uses Sphinx to build its documentation.

### Dependecies

To install dependencies required to build docs, run this from the package root folder:

```
pip install -r requirements/doc_requirements.txt
```

### Generating API doc source files

Sphinx-apidoc is used to generate API reference source .rst files. If you are changing the structure of the modules or introducing new module, you need to re-generate these files. To do so, from inside the "doc" folder, run:

```
rm api/*
sphinx-apidoc -d 1 -E -o ./api ../emukit
```

### Generating docs locally

If you'd like to generate the docs locally, from inside the "doc" directory, run:

```
make html
```