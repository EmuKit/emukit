---
layout: default
title: Installation
permalink: /installation/
---

<h1>Installation</h1>

Installation from sources and using pip are both supported.

#### Dependencies / Prerequisites
Emukit requires Python 3.5 or above and NumPy for basic functionality. Some core features also need [GPy](https://sheffieldml.github.io/GPy/) and [GPyOpt](https://sheffieldml.github.io/GPyOpt/). Some advanced elements may have their own dependencies, but their installation is optional.

Required dependecies can be installed from the [requirements](https://github.com/amzn/emukit/blob/master/requirements/requirements.txt) file via 

{% highlight ruby %}
pip install -r requirements/requirements.txt
{% endhighlight %}

#### Install using pip 
Just write:

{% highlight ruby %}
pip install emukit
{% endhighlight %}


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

