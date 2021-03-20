![Aestheta Logo](/Images/logo_1280x640.png)

# Aestheta
[![Python package](https://github.com/NSCC-COGS/Aestheta/actions/workflows/python-package.yml/badge.svg)](https://github.com/NSCC-COGS/Aestheta/actions/workflows/python-package.yml)

An Industrial Think Tank Focused on Developing and Promoting AI Technology for Geospatial Applications

<i>Let's get this thing rolling!</i>

Starting out: lets have a look at the previous tutorials from GDA2030 to get an idea where we can begin. There we can learn some [basics of using google colab](https://github.com/NSCC-COGS/GDAA2030/tree/master/tutorial1), we can see some [basic image loading examples in python](https://github.com/NSCC-COGS/GDAA2030/blob/master/tutorial1/kevinkmcguigan/GDAA2030_T1_kevinmc.ipynb), see an example of [accessing landsat data](https://github.com/NSCC-COGS/GDAA2030/blob/master/tutorial2/kevinkmcguigan/GDAA2030_T2_kevinmc_getLandsat.ipynb), see some more [advanced image maniuplation](https://github.com/NSCC-COGS/GDAA2030/blob/master/tutorial4/kevinkmcguigan/GDAA2030_T4_kevinmc.ipynb) techniques, and finally get an example of [building a model in scikitlearn](https://github.com/NSCC-COGS/GDAA2030/blob/master/tutorial5/kevinkmcguigan/GDAA2030_T5_kevinmc.ipynb).

let's do this!

Check out our latest ['Hello Earth'](/Experiments/Hello_Earth.ipynb) satellite classification and other weird tests in our [experiments](/Experiments) section... 

# Getting Started 

To get going fast, try the following in a new [Google Colab](https://colab.research.google.com/) notebook

```!git clone https://github.com/NSCC-COGS/Aestheta.git```

```import Aestheta.Library.core as core```

```core.getTile(source = 'google_sat', show=True)```

You should see earth appear - represented as a small numpy array! 
Stay tuned for more simple examples of what were doing with this data. 

# Requirements

We reccomend 64-bit python version 3.7.10 and higher. We use scikit-learn which includes numpy, imageIO. 

# Earth is Special
We're teaching an AI to understand what it means to be looking at our lovely planet. 

![Classified Earth](/Images/classified_earth.PNG)

There is lots of great data available and tools we can use! Early results are very promising. 

![Hello Earth](/Images/HelloEarth.png)

