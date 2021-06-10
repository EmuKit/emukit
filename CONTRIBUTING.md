## Overview

Hello and thanks for considering contributing to Emukit! It's people like you who make the project great!

Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Following these guidelines helps smooth out the process of contributing for both you as a contributor and those who maintain the project, making better use of everyone's time and energy.

### Reporting a bug
If you find a bug in Emukit, please file an issue, detailing as much about your setup as possible to help future developers in reproducing the issue. Tag it as a bug, with any additional tags as relevant.

Information should contain the following:
1. What version of Emukit are you using?
2. What python version are you using?
3. What did you do?
4. What did you expect to happen?
5. What actually happened?
6. Have you made any modifications relevant to the bug?

### Submitting a feature request
If you're wishing for a feature that doesn't exist yet in Emukit, there are probably others who need it too. Open an issue on GitHub describing what it is you'd like to see added, why it's useful, and how it might work. Add an "Enhancement" tag for bonus points! Better yet, submit a pull request providing the feature you need!

### Contributing code

If you're thinking about adding code to Emukit, here are some guidelines to get you started.

* If the change is a major feature, create an issue detailing your plan, optionally with a prototype implementation of your proposed changes. This is to get community feedback on the changes and document the design reasoning of Emukit for future reference.

* Keep pull requests small, preferably one feature per pull request. This lowers the bar to entry for a reviewer, and keeps feedback focused for each feature.

Some major areas where we appreciate contributions:
* Adding new methods' implementations.
* Model wrappers for new modelling frameworks.
* Example notebooks showing how to use a particular method.

If you're still not sure where to begin, have a look at our [issues](issues) page for open work.

## Contributing via Pull Requests

So you've implemented a new method for uncertainty quantification and you want to give it back to the community. Now it's time to submit a pull request! Before sending us a pull request, please ensure to:

1. Check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
2. Open an issue to discuss any significant work - we would hate for your time to be wasted.
3. Working against the latest source on the *main* branch.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, making sure you've answered the questions in the pull request checklist (see below).
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and [creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

Feel free to ask for help if you're unsure what to do at any point in the process, or on how best to integrate your code with the existing codebase!

If you feel like your pull request is not receiving enough attention, feel free to ping the thread as a reminder.

#### Coding style
We follow to PEP8 standards as a general rule.

The few Emukit specific rules are:
* Interface names start with *I*: *IModel*, *IDifferentiable*
* Use type hints to document input and output types: *def add(a: int, b: int) -> int*

#### Pull Request Checklist
Before submitting the pull request, please go through this checklist to make the process smoother for both yourself and the reviewers.
* Are there unit tests with good code coverage? Please include numerical stability checks against edge cases if applicable.
* Do all public functions have docstrings including examples? If you added a new module, did you add it to the Sphinx ```api.rst``` file in the ```doc``` folder?
* Is the code style correct (PEP8)?
* Is the commit message clear and informative?
* Is there an issue related to the change? If so, please link the issue.
* If this is a large addition, is there a tutorial or more extensive module-level description? Is there an example notebook?

## Setting up a development environment

### Building the code
See installing from source.

### Running tests
Run the full suite of tests by running:
```
python -m pytest
```
from the top level directory. This also does coverage checks.

### Generating docs
Documentation contributions are much appreciated! If you see something incorrect or poorly explained, feel free to fix it and send the update!

If you'd like to generate the docs locally, from inside the "doc" directory, run:

```
make html
```

## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.

## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact opensource-codeofconduct@amazon.com with any additional questions or comments.

## Acknowledgements
CONTRIBUTING contents partially inspired from [scipy's](https://github.com/scipy/scipy/blob/master/HACKING.rst.txt) and [Rust's](https://github.com/rust-lang/rust/blob/master/CONTRIBUTING.md) contribution guides!
