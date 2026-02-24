# UC Berkeley Professional Certificate in ML + AI: Practical Exercise 1

**Author**: Jelani Gould-Bailey

### Overview

For this exercise we were asked to analaze a dataset related to coupon offers to drivers for nearby
businesses such as Bars, Coffee Shops, and Restaurants, to answer the question “Will a customer accept the coupon?” 

We were asked to analyze the data, create
charts, and find patterns. The first half of the assignment was guided, the second half was open-ended.

### Data

This data is from the UCI Machine Learning Repository and was collected via a survey on Amazon Mechanical Turk. The survey describes different driving scenarios, including the destination, current time, weather, and passenger, and then asks people whether they will accept the coupon if they are the driver. There are three possible answers people can choose from:

1. “Right away”
1. “Later, before the coupon expires”
1. “No, I do not want the coupon”

The first two responses are labeled as “Y = 1,” and the third is labeled as “Y = 0.” 

There are five different types of coupons: 
1. Less expensive restaurants (under $20)
1. coffee houses
1. carryout and takeaway
1. bars
1. more expensive restaurants ($20–$50).



### Summary of Findings:
### First Half:
The first half analysis focused on drivers who were ofered a Bar coupon, and grouped them by criteria 
such as how often they visited a bar in a month, their marital status and whether they had children
passengers or not.

Based on the analsyis, there was a much higher acceptance rate for Bar coupons for:

1. Drivers who go to bars more than 3 times a month (76%)
1. Drivers who go to bars more than once a month and are under the age of 30 (72%)
1. Drivers who go to bars more than once a month, had passengers that were not a kid, and are not widowed (71%)

All 3 of these groups had bar coupon acceptance rates well above the mean bar coupon acceptance rate (~ 40%) across the population of drivers who accepted bar coupons.

### Second Half:
After looking at several different sets of data, I chose to focus on drivers who were offered Coffee Shop coupons. 

Analysis indicated that there is a significantly higher coupon acceptance rate among drivers in certain age groups who already visit a coffee shop a few times a month (66% for 46 and over; 69% for 26 and under) vs.those who visit less frequently or never visit (37% for men in this group; 30% for women in this group). The mean acceptance rate for this coupon across all drivers was 50%.

### Next Steps & Recommendations:
I believe this analysis could be useful for the marketing department of a Coffee Shop chain, who could decide if there would be positive ROI for running a local offers campaign. The data indicates that the marketing team should tailor their marketing strategy and/or contnet based on target audience, e.g. repeat customers in certain age brackets, or infrequent customers in different gender groups, may be more receptive to marketing language and content that is tailored to their group.

### Reference links:

- [Jupyter notebook](https://github.com/jelanigb/uc_berkeley_ml_program/blob/main/module_05/module_05_prompt_jgb.ipynb) for this project
- [Generated plots](https://github.com/jelanigb/uc_berkeley_ml_program/tree/main/module_05/images/generated) from this project
