# Predictive Maintenance Scheduling of Aircraft
Course Project- AA228/CS238, "Decision Making Under Uncertainity" by Prof. Mykel Kochenderfer, Stanford University

The project was done in collaboration with Ms. Eleanore Jacquemet, MS candidate Mechanical Engineering,Stanford University.

## Introduction

For an airline, aging aircraft fleets present a challenge in that
they need more maintenance, and are less efficient than newly
purchased aircraft. Ideally, when a carrier has a mixture of
older and newer aircraft at their disposal, they would always
want to fly the newer, more efficient versions. But continuously flying the newer fleet is impossible, because regular
maintenance of these aircraft is required, which make them
unavailable for that time period. Combined with yearly trends
in passenger flying and stochastic fuel prices,optimal predictive maintenance of more efficient aircraft is beneficial to the
airline, which wants to maximise revenue.
In this work, we frame the given problem as an MDP and
solve it using several algorithms, exploring the comparison of
average earned profits between an optimal maintenance policy and other policies like a greedy policy, which is more com-
monly used.

## Methodology

This project is an attempt to frame the problem of optimizing maintenance planning looking into the future, as a Markov Decision Process. We model the problem as a two-aircraft problem, one of which has a higher return per passenger flown. The goal is to find the optimal "policy" or sequence of flying these aircraft(while the other is in maintenance) such that we may maximise profits earned by the carrier, even in the face of stochastic uncertainity of important variables like fuel price, load factor etc.

Unlike previous attempts at similar work which attempt to maximise revenue, we wish to directly model and maximise profits earned by a carrier. The problem dynamics and equations are created from a mixture of modelled real-world historical data and some speculation, the speculative part being one drawback of the work in it's current form. Obtaining policies of action using the Forward Search and Policy Iteration Algorithms, we then compare the results obtained to a greedy policy, which is using the "best" aircraft(aircraft with maximum return per passenger, which yields more profits) until it can no longer fly due to mandatory maintenance requirement, and the carrier must use the older plane.

Further details can be found in the write up included in the files in the repository.

## Files

- forward_search.py: File that reads real-world data from the file passengers.py and given certain parameters, calculates an optimal policy
- passengers.py: Uses data from several sources including the [Bureau of Transport](https://www.bts.dot.gov/product/passenger-travel-facts-and-figures) 
- AA228_finalpaper.pdf: This is the final writeup for the course project, demonstrating results, drawbacks and future work. Refer to this for detailed explanations of every component of the project


