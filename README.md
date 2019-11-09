# microphi
Microphi is an open-source python miniproject which provides functions to transform solid solutions from one basis to another. It can either be used to convert between different endmember sets, or to compute macroscopic endmember and interaction energies from microscopic site properties and interactions.

The motivation for this miniproject is a paper from Roger Powell and coauthors (2014; https://onlinelibrary.wiley.com/doi/full/10.1111/jmg.12070). That paper was concerned with microscopic->macroscopic conversions for strictly regular solutions (i.e. with only symmetric binary parameters). The current version of this project extends that work to asymmetric (Holland and Powell, 2003) and subregular (e.g. Helffrich and Wood, 1989) solution models.

The functions implemented in this project allow the user to read in input files, or declare the desired inputs as python objects. A set of examples are provided.
