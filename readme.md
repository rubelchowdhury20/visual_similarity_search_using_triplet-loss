
#### Visual Similarity Search
* This is a generic implementation of visual similarity search algorithm using triplet loss. 
* The mining strategy demands available positive images for all the anchor images. And then we purposefully inject relevant positive images for each anchor image for each bach which makes the training faster.
* The code is inspired by the omoindrot triplet loss implementation with required modifications.
* This code can be used for any kind of visual similarity serach if you have the postive image collections for all the anchor images.


#### To_do_list

* learning rate strategies
* what to do with the momentum part of learning rate strategy
* for now keeping the architecture as resnet, but have to change the architecture similar to hyperface paper where initial weights can be initialised from the classification task and then later used for triplet loss.






