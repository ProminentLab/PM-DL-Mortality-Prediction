# Python library for ICD9 Codes

The library encodes [ICD9
codes](https://en.wikipedia.org/wiki/International_Statistical_Classification_of_Diseases_and_Related_Health_Problems#ICD-9)
in their natural hierarchy.  For example, "Cholera due to vibrio cholerae" has
the ICD9 code `001.0`, and is categorized as a type of Cholera, which in turn
is a type of Intestinal Infectious Disease.  Specifically, `001.0` has the
following hierarchy:

    001-139     Infectious and Parasitic Diseases
      001-009   Intestinal Infectious Diseases
        001     Cholera
          001.0 Cholera due to vibrio cholerae

Assuming that codes closely related in the tree are more related than with
codes further in the tree, this hierarchy is a way to cluster related codes.

This library encodes all ICD9 codes and their descriptions into a tree that
captures these relationships.


## Using the library

Include `icd9.py` in your python path.  Then put `codes.json` somewhere
convenient.  Here's a simple example:

```python

from icd9 import ICD9

# feel free to replace with your path to the json file
tree = ICD9('codes.json')

# list of top level codes (e.g., '001-139', ...)
toplevelnodes = tree.children
toplevelcodes = [node.code for node in toplevelnodes]
print '\t'.join(toplevelcodes)
```


The hierarchy is encoded in a tree of `Node` objects.  The `ICD9()` constructor
returns the root `Node`.  `Node` has the following methods:

`node.search(code)`

```python
# find all sub-nodes whose codes contain '001'
tree.search('001')
```

`node.find(code)`

```python
# find sub-node with code '001.0'. Returns None if code is not found
tree.find('001.0')
```

And the following properties:

`node.code`

```python
# get node's ICD9 code
tree.find('001.1').code

# prints '001'
tree.find('001.1').parent.code

# prints '001'
tree.find('001').code
```

`node.description`:

```python
# get english description of ICD9 code
# prints: 'Cholera due to vibrio cholerae el tor'
tree.find('001.1').description

# prints: 'ROOT'
tree.description

# prints: 'Cholera'
tree.find('001.1').parent.description

# also prints: 'Cholera'
tree.find('001').description
```

`node.descr`: alias for `description`

`node.children`

```python
# get node's children
tree.children

# search for '001.0' in root's first child
tree.children[0].search('001.0')
```

`node.parent`

```python
# get 001.0 node's parent.  None if node is a root
tree.find('001.0').parent
```

`node.parents`

```python
# get 001.0 node's parent path from the root.  Root node is the first element
tree.find('001.0').parents
```

`node.leaves`

```python
# get all leaf nodes under root's first child
tree.children[0].leaves
```

`node.siblings`

```python
# get all of 001.0 node's siblings that share the same parent
tree.find('001.0').siblings
```


## ICD9 Descriptions

This library includes descriptions of each ICD9 code and grouping name.

If you are interested in another list of ICD9 codes and their descriptions,
[drobhbins](https://github.com/drobbins/ICD9) created a csv file of ICD9 codes
and their short and long descriptions.

## Scraper

The `scraper/` directory includes the scraper code used to generate the
dataset.  `scraper/scraper.py` creates a json file `codes.json` of each ICD9
code's parent codes and descriptions:

```python
[
  {'code': None},
  {'code': '001-139', 'descr': 'Infectious and Parasitic Diseases'},
  {'code': '001-009', 'descr': 'Intestinal Infectious Diseases'},
  {'code': '001', 'descr': 'Cholera'},
  {'code':  '001.0', 'descr': 'Cholera due to vibrio cholerae'}
]
```

The last element is the actual code, the preceeding elements are coarser
groupings of codes.  The first element is a dummy that represents root.

Thanks to [http://icd9cm.chrisendres.com/](http://icd9cm.chrisendres.com),
where the data was secretly scraped from.
