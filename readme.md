
## Visual Similarity Search using Triplet Loss
#### Pytorch implementation of retrieving visually similar items from a collection using Triplet Loss.

#### Abstract:

We have implemented the Triplet Loss to find similar images, where one novice strategy is used to mine the Postive and Negative images for an Anchor image to form the triplets for training purpose.

### Mining Strategy:
In general the label value is used to mine the **positive** and **negative** for each anchor images. There are two different ways of training named **Online and Offline training**. In Offline Triplet training, we gather all the triplets before the start of the training and use them through-out for the training process, which is not the efficient way to do that. And in Online training, we mine the triplets on the go, that means in a single batch for each of the anchor image, we calculate the embedding distance with the other avialable images in the batch using the semi trained model and then use the image having the smallest distance as the positive and the largest distance one as the negative and that's how we form the triplets.
Now the short coming of this method is we select the positive withing the batch only while ignoring other possible triplets in the dataset. That issue can be tackled using a combination of offline and online mining strategy.
In this method, before starting the training itself we first use on pre-trained model to get the embeddings for each of the images and then we take each image as an anchor image and calculate the distance of that with all the available images and then we plot a distribution for those distances. The graph looks like somthing like this.


![Euclidean distance distribution](https://i.imgur.com/o4ZauaL.png)

Here we can see there are two peaks, one is for the images which are positive images, which will be lesser in number that's why it's height is smaller than the other peak. So, the second peak is all those images which are negative ones. Now, the images which are on the left side of the first peak we can consider them as **easy positives** and the ones which are on the right side of the first peak and closer to the peak those are the **hard positives**. Same way the examples on the left side of the second peak are **hard negatives** and the right side of the second peak are the **easy negatives**. Now according to the original triplet loss paper, we training get's better if we select **hard positive and  negative** for each of the anchors to make the training better.

So, following the above technique we can select few hard positives and negatives for each images and while training for each images in the batch we inject it's relevant hard positive in the batch and form the finest triplet in each of the batch which makes the whole training process much better and faster.

The idea of injecting postive images for each image in a batch is taken from the implementation of Olivier Moindrot.[(code)](https://github.com/omoindrot/tensorflow-triplet-loss)

### Results obtanined by using the strategy:
#### Example 1
![image_1](https://i.imgur.com/grUlgeN.jpg)

#### Example 2
![image_2](https://i.imgur.com/rdLtVaG.jpg)













