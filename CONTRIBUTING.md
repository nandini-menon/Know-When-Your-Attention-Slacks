# Contributing

When contributing to this repository, please first discuss the change you wish to 
make via issue, slack or any other method with the owners 
of this repository before making a change. 

## Pull Request Process

### 1. Explore

If there is some issue or bug or enhancements, you are interested in and no one else is working 
on the issue, you may take it up (just leave a comment on the issue).  

**Make sure you create an issue before making major code changes or adding new features**

### 2. Fork & create a branch

If this is something you think you can fix, then create a branch with a 
descriptive name.  

The core code for getting the website up and running should be made against master.

For adding new features/fixing issues, a good branch name would be (where issue #13 is the ticket you're working on):

```sh
git checkout -b 13-add-xyz-feature
```

### 3. Implement your fix or feature

At this point, you're ready to make your changes! Feel free to ask for help;
everyone is a beginner at first :smile_cat:  

### Note 1

**Make sure that you dont make too many code changes at once and later add everything as a single commit. By adding a new function or by fixing a relevant bug or by adding a new feature, make sure that you commit those chunks of code**

### Note 2
**Kindly add descriptive commit messages. Make it elaborate and on point. "Fixes errors" is an example for a bad commit message. Whereas, "fix #331 by removing the call to FuncName() at line 22" is an example of a good commit message.**


### 4. Test for all the checks

Your patch should follow the same conventions & pass the same code quality
checks as the rest of the project.  

Auto-formatting can be done directly via your text editor. [See how auto-formatting can be done](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode).

### 5. Make a Pull Request

At this point, you should switch back to your master branch and make sure it's
up to date with Active Admin's master branch:

```sh
git remote add upstream git@github.com:fisatsdc/fsdc.git
git checkout master
git pull upstream master
```

Getting a `Permission denied (publickey)` error? [Follow this guide to fix it.](https://stackoverflow.com/questions/2643502/how-to-solve-permission-denied-publickey-error-when-using-git)  

Then update your feature branch from your local copy of master, and push it!

```sh
git checkout 13-add-xyz-feature
git rebase master
git push --set-upstream origin 13-add-xyz-feature
```

Finally, go to GitHub and make a Pull Request :D

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable
behavior and are expected to take appropriate and fair corrective action in
response to any instances of unacceptable behavior.  

No maintainer shall make changes that were either not discussed with others or without issues raised.
