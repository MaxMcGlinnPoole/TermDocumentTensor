# Term Document Tensor

Creates a Term Document Tensor. This Term Document Tensor can then be decomposed and clustered. 
### Requirements

* Python 3.6


### Installation

Instructions should work for MacOS and Linux systems

First make sure you have installed Python 3.6 and Pip. Then install virtualenv

```
pip install virtualenv
```

Then create and activate your virtualenv
```
virtualenv venv
virtualenv -p python3 venv
source venv/bin/activate
```

Next install the project requirements
```
pip install -r requirements.txt
pip install git+https://github.com/MaxMcGlinnPoole/tf-decompose.git
```


### Usage
  Currently the program takes several arguments and options:
   vx.py [-h] [-d DIRECTORY_NAME] [CLUSTERING OPTIONS] (-b | -t) (-ngrams)
             
  ```
  -d or --directory accepts the directory name for the files to be parsed
  -b or -t to parse the files as text or binary files
  -parafac or -tucker to use either tensor decomposition (more to be added later)
  -o to generate an output (functionality in progress)
  -ngrams ngrams of the term document tensor
  -heatmap generate a heatmap of the cosine similarity matrix
  -kmeans do kmeans clustering on one of the factor matrices
  ```
  Sample usage
  
  ```
  python3 vx.py -d myDirectory -heatmap -b 
  ```

### Contributing

1. Make a local clone: 
  ```sh
  git clone https://github.com/MaxMcGlinnPoole/TermDocumentTensor.git
  ```
  **Choose your own destination path. The directory this command is ran in where there folder will be located**

2. Switch to the directory: `cd TermDocumentTensor` 
3. Create your new branch: `git checkout -b branch name`
4. Make necessary changes to this source code
5. Add changes to git index by using `git add --all .`
6. Commit your changes: `git commit -am 'update description'`
7. Push to the branch: `git push`
8. Submit a [new pull request](https://github.com/MaxMcGlinnPoole/TermDocumentTensor/pull/new)


## Authors 
Meet our research team
* [Max Poole](https://github.com/MaxMcGlinnPoole)
* [Sumanth Neerumalla](https://github.com/sumanthneerumalla)
* [Donnell Muse](https://github.com/Donnell794)
* [Phani Teja Kesha](https://github.com/phanitejakesha)

## Acknowledgments

* Special thanks to Dr. Charles Nicholas for his mentorship on this research project
* Dr. Tyler Simon, who has been especially helpful with his knowledge of the subject
* Tamara G. Kolda and Brett W. Bader for their paper on [tensor decomposition](http://www.sandia.gov/~tgkolda/pubs/pubfiles/TensorReview.pdf) 

### Sponsor

* NSF
