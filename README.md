# HW4
Ai HW4
Zach Jaffee (zij)
George Hodulik (gmh73)

One thing to note about the code is how the features are implemented, through out our work on this programming assignment we have found that it was difficult to get the algorithm to consistantly producing winning results (e.g. 5/5 wins when not exploring) even after many interations. While it would do better after the first round of exploration, after that it started to wane and it felt almost random that we would occasionally lose. We have implemented a whole bunch of different features, yet for the most part, none of the ones other than the Chebyshev Distance, which has some of its own issues in that at the start of the program many of the combat_agents all have the same Chebyshev Distance value. This in turn results in functions doing some odd things such as all of the agents choosing the same enemy to attack first, which makes sense as a gready approach to solving the problem, but there are failings here in that sometimes our units get surrounded by the combat_agents and then get killed. However, most of the time they do move in a manner that does the opposite (e.g. out footmen surround the combat_agent), which is what happens most of the time after learning occurs anyways. 
We have left all of the other features that we thought would make a difference in the program, but left as comments so you can see what we have decided doesn't work as well for what we currently have for the rest of our functions. Its also worth noting that at least in our case, adding many of these additional features, make what is currently effective in our program have a much lower weight and renders our program, just as if not equal to, running the program with out any features at all, e.g. not having a general heuritic we use to determine which way is best to move and attack.