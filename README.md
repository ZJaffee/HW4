HW4
Ai HW4 
Zach Jaffee (zij) 
George Hodulik (gmh73)

One thing to note about the code is how the features are implemented, throughout our work on this programming assignment we have found that it was difficult to get a policy that consistently won all the time: ie. in the 10v10 case, we can get it to win (4-5)/5 times usually, but occasionally it only wins 1 or 2.  However, acting totally randomly, we win 0-1 times, so obviously our found policy is better than acting randomly.
We also noticed that, sometimes the algorithm would find weights that usually win (3-5)/5 times, but then it may "explore away" from this policy, winning by less, before re-converging to the more optimal policy.  Thus, we implemented code in the terminal state that sets epsilon to 0 after the agent wins 5/5 times.  Feel free to comment this out if that is not cool.
We tried out several features, but the only that seemed to have a real effect was whether or not the targeted footman was the closest to the attacker or not (+-1.0 feature value).  We left our old features commented out, to show that we did actually try several others.
Something odd about our project is that the 10v10 case works rather well, usually winning (3-5)/5 times, but the 5v5 case only wins about (2-3)/5 times usually.  I think this may be that, given that our only move option is attack(f,e), there is not a strategy much more optimal - if we could group the footmen above/below the enemies and swarm, that might work, but with only attack(f,e), that strategy is not possible.  However, in 10v10, this strategy is kind of possible because there are more footmen, and the vertical distance between footmen is greater, making a swarm-like shape possible.

There does appear to be a general trend of increasing average rewards and increasing # games won out of 5, but this trend is often extremely slight.  For example, at the beginning of running the 5v5 case, the average reward may jump between like 6 and 15, and by the end, its more like 11 and 20 -- there is a general trend of increasing, but there's still quite a bit of fluctuation.

We tried "discounting" rewards by saying that, when an enemy is killed, the reward is split among all the attackers.  However, we found that this punished teaming up on a target, and did not work as well as not discounting the rewards and rewarding coordination.

We define events as being when a SEPIA move is completed/ a unit dies, or a SEPIA move fails.  There is also an event whenever one of our units is attacked by an enemy that is not its target enemy and it is not attacked by its target enemy.

OPTIMAL WEIGHTS
After running 660 exploration episodes, the optimal weights settled at
[76.86935766543543, 12.520833207394514]
Note that w[1] seemed to converge, but w[0] was still slowly growing, thus, ~77 may not be the optimal value, but actually a value a bit greater.