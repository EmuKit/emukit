---
layout: post
title: Documentation
img: image-5.png
---


# Installation: 
Fork the ``master`` branch and delete ``gh-pages`` branch in it. This is important because ``gh-pages`` branch is used here only to host the blog. You should be using the master branch as the source and create a fresh ``gh-pages`` branch.

Watch my video on instlallation
<iframe width="100%" height="360" src="https://www.youtube.com/embed/T2nx6tj-ZH4?rel=0" frameborder="0" allowfullscreen></iframe>

## How to delete old **gh-pages** branch?
After forking the repository, click on **branches**.

![delete gh-pages branch]({{site.baseurl}}/images/delete-github-branch.png)

Delete ``gh-pages`` branch.
![delete gh-pages branch]({{site.baseurl}}/images/delete-github-branch-2.png)

You have to create a new ``gh-pages`` branch using the master branch. Go back to the forked repository and create ``gh-pages`` branch.

![create gh-pages branch]({{site.baseurl}}/images/create-gh-pages-branch.JPG)

Now, go to settings and check the **Github Pages** section. You should see a URL where the blog is hosted.

This process will host the theme as a **Project Page**. You can also download the files for local development. 

Default theme will look like this

![webjeda cards jekyll theme]({{site.baseurl}}/images/webjeda-cards-jekyll-theme-1.png)

This theme is responsive.

![webjeda cards responsive jekyll theme]({{site.baseurl}}/images/webjeda-cards-responsive-jekyll-theme-2.png)
{: .text-center}


# Development
Make changes to the **master** branch and create a pull request. Do not use **gh-pages** branch as it is used to host the theme.


# License
MIT License

# Changelog
<pre>
version 1.0 - SEO, disqus, page-speed enhancement, compressed html.

version 0.9 - Sidebar default image and other minor improvements.  
  
version 0.8 - Bootstrap based cards layout theme.
</pre>
