---
layout: post
title:  "Elements"
categories: jekyll update
img: image-4.png
---
Below elements can be used just by adding certain classes to regular html elements. Since the theme is based on Bootstrap, you can use any bootstrap class in this theme.

## Buttons

<a href="#" class="btn btn-default">Default</a>
<a href="#" class="btn btn-primary">Primary</a>
<a href="#" class="btn btn-success">Success</a>
<a href="#" class="btn btn-info">Info</a>
<a href="#" class="btn btn-warning">Warning</a>
<a href="#" class="btn btn-danger">Danger</a>
<a href="#" class="btn btn-link">Link</a>
<br /><br />
<a href="#" class="btn btn-primary btn-lg">Large button</a>
<a href="#" class="btn btn-primary">Default button</a>
<a href="#" class="btn btn-primary btn-sm">Small button</a>
<a href="#" class="btn btn-primary btn-xs">Mini button</a>

{% highlight html %}
<a href="#" class="btn btn-default">Default</a>
<a href="#" class="btn btn-primary">Primary</a>
<a href="#" class="btn btn-success">Success</a>
<a href="#" class="btn btn-info">Info</a>
<a href="#" class="btn btn-warning">Warning</a>
<a href="#" class="btn btn-danger">Danger</a>
<a href="#" class="btn btn-link">Link</a>

<a href="#" class="btn btn-primary btn-lg">Large button</a>
<a href="#" class="btn btn-primary">Default button</a>
<a href="#" class="btn btn-primary btn-sm">Small button</a>
<a href="#" class="btn btn-primary btn-xs">Mini button</a>
{% endhighlight %}


## Typo

### Heading

<h1>Heading 1</h1>
<h2>Heading 2</h2>
<h3>Heading 3</h3>
<h4>Heading 4</h4>
<h5>Heading 5</h5>
<h6>Heading 6</h6>

{% highlight html %}
<h1>Heading 1</h1>
<h2>Heading 2</h2>
<h3>Heading 3</h3>
<h4>Heading 4</h4>
<h5>Heading 5</h5>
<h6>Heading 6</h6>
{% endhighlight %}

### Blockquote
<blockquote>
  <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer posuere erat a ante.</p>
  <small>Someone famous in <cite title="Source Title">Source Title</cite></small>
</blockquote>

{% highlight html %}
<blockquote>
  <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer posuere erat a ante.</p>
  <small>Someone famous in <cite title="Source Title">Source Title</cite></small>
</blockquote>
{% endhighlight %}


## Table

| Tables   |      Are      |  Cool |
|----------|:-------------:|------:|
| col 1 is |  left-aligned | $1600 |
| col 2 is |    centered   |   $12 |
| col 3 is | right-aligned |    $1 |
{: .table .table-striped .table-hover}

{% highlight markdown %}
| Tables   |      Are      |  Cool |
|----------|:-------------:|------:|
| col 1 is |  left-aligned | $1600 |
| col 2 is |    centered   |   $12 |
| col 3 is | right-aligned |    $1 |
{: .table .table-striped .table-hover}
{% endhighlight %}

<div class="mt20"></div>

## Indicators


<div class="alert alert-dismissible alert-warning">
  <button type="button" class="close" data-dismiss="alert">&times;</button>
  <h4>Warning!</h4>
  <p>Best check yo self, you're not looking too good. Nulla vitae elit libero, a pharetra augue. Praesent commodo cursus magna, <a href="#" class="alert-link">vel scelerisque nisl consectetur et</a>.</p>
</div>

{% highlight html %}
<div class="alert alert-dismissible alert-warning">
  <button type="button" class="close" data-dismiss="alert">&times;</button>
  <h4>Warning!</h4>
  <p>Best check yo self, you're not looking too good. Nulla vitae elit libero, a pharetra augue. Praesent commodo cursus magna, <a href="#" class="alert-link">vel scelerisque nisl consectetur et</a>.</p>
</div>
{% endhighlight %}


<div class="row">
    <div class="col-md-4">
        <div class="alert alert-dismissible alert-danger">
          <button type="button" class="close" data-dismiss="alert">&times;</button>
          <strong>Oh snap!</strong> <a href="#" class="alert-link">Change a few things up</a> and try submitting again.
          </div>
    </div>



   <div class="col-md-4">
        <div class="alert alert-dismissible alert-success">
          <button type="button" class="close" data-dismiss="alert">&times;</button>
          <strong>Well done!</strong> You successfully read <a href="#" class="alert-link">this important alert message</a>.
        </div>
    </div>



   <div class="col-md-4">
        <div class="alert alert-dismissible alert-info">
          <button type="button" class="close" data-dismiss="alert">&times;</button>
          <strong>Heads up!</strong> This <a href="#" class="alert-link">alert needs your attention</a>, but it's not super important.
        </div>
    </div>
</div>


<div class="row">
    <div class="col-md-4">
       {% highlight html %}
        <div class="alert alert-dismissible alert-danger">
          <button type="button" class="close" data-dismiss="alert">&times;</button>
          <strong>Oh snap!</strong> <a href="#" class="alert-link">Change a few things up</a> and try submitting again.
          </div>
          {% endhighlight %}
    </div>



   <div class="col-md-4">
       {% highlight html %}
        <div class="alert alert-dismissible alert-success">
          <button type="button" class="close" data-dismiss="alert">&times;</button>
          <strong>Well done!</strong> You successfully read <a href="#" class="alert-link">this important alert message</a>.
        </div>
        {% endhighlight %}
    </div>



   <div class="col-md-4">
       {% highlight html %}
        <div class="alert alert-dismissible alert-info">
          <button type="button" class="close" data-dismiss="alert">&times;</button>
          <strong>Heads up!</strong> This <a href="#" class="alert-link">alert needs your attention</a>, but it's not super important.
        </div>
        {% endhighlight %}
    </div>
</div>

## Labels

<span class="label label-default">Default</span>
<span class="label label-primary">Primary</span>
<span class="label label-success">Success</span>
<span class="label label-warning">Warning</span>
<span class="label label-danger">Danger</span>
<span class="label label-info">Info</span>

{% highlight html %}
<span class="label label-default">Default</span>
<span class="label label-primary">Primary</span>
<span class="label label-success">Success</span>
<span class="label label-warning">Warning</span>
<span class="label label-danger">Danger</span>
<span class="label label-info">Info</span>
{% endhighlight %}


## Jumbotron

<div class="jumbotron">
  <h1>Jumbotron</h1>
  <p>This is a simple hero unit, a simple jumbotron-style component for calling extra attention to featured content or information.</p>
  <p><a class="btn btn-primary btn-lg">Learn more</a></p>
</div>

{% highlight html %}
<div class="jumbotron">
  <h1>Jumbotron</h1>
  <p>This is a simple hero unit, a simple jumbotron-style component for calling extra attention to featured content or information.</p>
  <p><a class="btn btn-primary btn-lg">Learn more</a></p>
</div>
{% endhighlight %}

## Panels

<div class="row">
    <div class="col-md-4">
        <div class="panel panel-primary">
          <div class="panel-heading">
            <h3 class="panel-title">Panel primary</h3>
          </div>
          <div class="panel-body">
            Panel content
          </div>
        </div>
    </div>



   <div class="col-md-4">
        <div class="panel panel-success">
          <div class="panel-heading">
            <h3 class="panel-title">Panel success</h3>
          </div>
          <div class="panel-body">
            Panel content
          </div>
        </div>
    </div>



   <div class="col-md-4">
        <div class="panel panel-warning">
          <div class="panel-heading">
            <h3 class="panel-title">Panel warning</h3>
          </div>
          <div class="panel-body">
            Panel content
          </div>
        </div>
    </div>
</div>

<div class="row">
    <div class="col-md-4">
        <div class="panel panel-danger">
          <div class="panel-heading">
            <h3 class="panel-title">Panel danger</h3>
          </div>
          <div class="panel-body">
            Panel content
          </div>
        </div>
    </div>



   <div class="col-md-4">
        <div class="panel panel-info">
          <div class="panel-heading">
            <h3 class="panel-title">Panel info</h3>
          </div>
          <div class="panel-body">
            Panel content
          </div>
        </div>
    </div>

   <div class="col-md-4">
    </div>
</div>

{% highlight html %}
<div class="panel panel-primary">
    <div class="panel-heading">
    <h3 class="panel-title">Panel primary</h3>
    </div>
    <div class="panel-body">
    Panel content
    </div>
</div>
{% endhighlight %}

Change class ``panel-primary`` to ``panel-warning``, ``panel-success`` and so on to get other panels.