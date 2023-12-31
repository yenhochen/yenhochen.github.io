---
layout: about
title: home
permalink: /
subtitle: 

# profile:
  # align: right
  # image: 
  # image_circular: false # crops the image to make it circular
  # address: 
# news: true  # includes a list of news items
# latest_posts: true  # includes a list of the newest posts
# selected_papers: true # includes a list of papers marked as "selected={true}"
# social: true  # includes social icons at the bottom of the page

scripts:
    - home/home.js 
style: home/home.css
img_path: home/bayes_inference.gif
---




<div class="center-screen">
  <p style="font-family:Helvetica; font-size:larger;" class="font-weight-bold">Yenho Chen</p>
  
  <div class="about-text"> I am a PhD Student in <a href="https://ml.gatech.edu/">Machine Learning at Georgia Tech</a>, where I work with Dr. <a href="https://siplab.gatech.edu/">Chris Rozell</a>. Previously, I developed a variety of statistical tools for neuroscience as a research fellow in the <a href="https://cmn.nimh.nih.gov/mlt">Machine Learning Team at the National Institute of Mental Health</a>. Currently, my work centers on developing new computational tools that help scientists reveal and understand hidden structure in complex time-varying systems. Moreover, I focus on fast, robust, and scalable approaches to ensure the practical application of these techniques. I am broadly interested in solving real world problems by advancing the areas of <span onmouseover="mover(this)" class="highlight-art generative-modeling">deep learning</span>, <span onmouseover="mover(this)" class="highlight-art ot">numerical optimization</span>, and <span onmouseover="mover(this)" class="highlight-art bayes">Bayesian inference</span>.

<!-- Currently, my work focuses on creating new computational tools that help scientists reveal and understand hidden structure in complex time-series data -->

  <!-- I am broadly interested in solving real-world problems by advancing the areas of deep learning, numerical optimization, and Bayesian inference. -->
  
  
  <!-- I am broadly interested in solving real world problems by advancing the areas of <span onmouseover="mover(this)" class="highlight-art generative-modeling">deep learning</span>, <span onmouseover="mover(this)" class="highlight-art ot">numerical optimization</span>, and <span onmouseover="mover(this)" class="highlight-art bayes">Bayesian inference</span>. -->
  </div>
  <br>
  <br>





  

<div class="home-bottom-container">
  <div class="home-links-container">   
      <span  markdown="1">[publications]({% link _pages/publications.md %})</span>     
      <p>
        <a class="home-link" href="{{ '/blog/' | relative_url }}">{{ site.blog_nav_title }}
          {%- if page.url contains 'blog' -%}
          <span class="sr-only">(current)</span>
          {%- endif -%}
        </a>
      </p>
      <p><a class="home-link" href="https://github.com/yenhochen">code</a></p>
    <p style="font-family:monospace">[firstname]@gatech.edu</p>
  </div>
    <div class="home-img-container">
      <img src="assets/img/bayes_inference4.gif" width="300px" height="300px" id="bayes"> 
      <!-- <video  autoplay="autoplay" id="bayes" muted loop>
        <source src="assets/img/bayes_inference4.mp4" type="video/mp4" />
      </video> -->
      <img src="assets/img/ot2.gif" width="300px" height="300px" id="ot"> 
      <!-- <video  autoplay="autoplay controls" id="ot" muted loop>
        <source src="assets/img/ot2.mp4" type="video/mp4" />
      </video> -->
      <img src="assets/img/generative2.gif" width="300px" height="300px" id="generative-modeling"> 
      <!-- <video autoplay="autoplay" id="generative-modeling" muted loop>
        <source src="assets/img/generative2.mp4" type="video/mp4" />
      </video> -->
    </div>
</div> 
</div>

