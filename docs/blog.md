title: Debugging a custom ACT implementation


I wanted to choose a model that was efficient, and cheap with a lot of documentation to start the implementation. ACT provides a good route as it is supported in LeRobot, it's very ineffici efficient and cheap compared to other VLAs or world models and it has good performance on low number of demonstrations.

One error that we continually saw was the low training loss on offline training loss in general, but when we rolled out the robot online, it didn't seem to work at all.

We first started off with a pick and place task to place a piece of pasta inside of a bowl and realized that this was not a good task after many iterations, as we were not able to get a successful ACT working on this task, and we mainly attribute that to the pasta being extremely small and the image in it backbone not being able to actually see the pasta. That's why we moved over to the sponge which had a much brighter contrast with the table and it was also larger so the overhead camera was able to see it.

In our first main debugging implementation, we tried to do a lot of overfitting. So the first thing we tried to do was overfit with just a normal MLP to see if training loss would go down even further compared to what our ACT implementation training loss was, and so we did do that, and it went down extremely low. And then we also tried to overfit on a batch and overfit on an episode. However, we realized that the overfitting on bad single batcher episode was not a very good test because we were able to just actually it was a good test because it showed that the pre-processing pipeline was good or it was did not have any potential errors.

We ended up using LibRobot's infrastructure, but with our custom policy. So we used the pre-trained class to substitute our ACT config. This allowed us to match the same Lerobot pre and post processors as well as their eval slash rollout logic.

We chose to do a fully closed loop rollout using an async implementation mainly because our hardware was running on 10 hertz and we didn't have access to a larger GPU to run inference on locally, if we chose to do it online, we would have to deal with latency and being able to get the action chunks in time. and so doing a rollout like this allowed us just to execute our action chunk and then use an EMA smoother to blend the action chunks together. So we did not use any temporal ensembling in this implementation.

We also realized how important the correct offline setup was, as any changes that were very small and the camera placement overhead had significantly big impacts to the final results i.e. training on one set of camera images compared to another. We tried to add some noise, i.e., some extra stuff in our pre-processor to transform the images by including some light variants and saturation as well as some transforms like translation and rotation. And we saw that the translation and rotation That particular training method performed slightly worse than the normal LiRobot implementation when we evaluated it and we saw the one where we only changed light and saturation, hue, etc. And that performed much worse. And so we chose to evaluate our customer implementation against the standard stock LaRobot implementation.

One learning we also saw was collecting high quality data was more important than more data. So many times we tried training on lower amounts of data that were higher quality and it did better than the data set with more training examples.

We chose to evaluate the episodes on five distinct locations and orientations on our table to see if the pick and place would work. And these were kind of the results here. And we did the same evaluation across both.

One thing to note while doing this was that we initially did not account for sponge orientation. However, as we tried training and we eventually accounted for it, we realized that the orientation did have a tremendous impact on the result as in the initial training run we kept the very Very not or heavily biased orientations, so we had one orientation that would take up most of the data set. And then, when we tried to evaluate online, we realized that the policy didn't generalize at all. And so we went back and changed our data set to include more diverse orientations.

Our final results were that we reached a 10 out of 30 example eval on our custom, and the robot had an 11 out of 30 success rate, and so this kind of proved that we had an implementation that was at least on par or close to on par with the Labrova implementation for in-distribution performance. In both cases this was not a primary concern to evaluate out of distribution performance.

