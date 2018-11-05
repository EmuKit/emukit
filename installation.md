---
layout: page
title: Installation
permalink: /installation/
---

Currently only installation from sources is supported.

#### Dependencies / Prerequisites
Emukit's primary dependencies are Numpy, GPy and GPyOpt.
See [requirements](requirements/requirements.txt).

#### Install from sources
To install Emukit from source, create a local folder where you would like to put Emukit source code, and run following commands:

{% highlight ruby %}
git clone https://github.com/amzn/Emukit.git
cd Emukit
python setup.py install
{% endhighlight %}

Alternatively you can run

{% highlight ruby %}
pip install git+https://github.com/amzn/Emukit.git
{% endhighlight %}

#### For developers