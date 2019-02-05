---
layout: post
title: "Microsoft AI Challenge India 2018: Approach and Techniques"
excerpt: "This post describes my approach for the AI Challenge India organized by Microsoft Research, India"
tags:
  - nlp
  - competition
last_modified_at: 2019-02-05T13:48:50+05:30
---
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

This post describes my approach for the AI Challenge India organized by Microsoft Research, India. It was held from November, 2018 to December, 2018. The challenge was held in two phases. In the first phase the participants trained their models on the training set and submittted their evaluations for the first test set. About 250 teams who got the highest score qualified for the second phase.

 The results were announced in January, 2019. **I achieved the 3rd rank in the first phase and was amongst the top 20 teams in the finals.** 



## Introduction

The problem statement of the challenge as mentioned by the organizers was: *“Given a user query and top passages corresponding to each, the task is to mark the most relevant passage which contains the answer to the user query.”* A sample from the dataset provided for the challenge is as given below:

| Query                                       | Passage                                                      | Label |
| ------------------------------------------- | ------------------------------------------------------------ | ----- |
| symptoms of a dying mouse                   | This can be fatal quite quickly to mice. 1  It is caused by dusty sawdust (not wood shavings)... | 0     |
| symptoms of a dying mouse                   | The symptoms of mites include: 1  excessive scratching (other than grooming) 2  fur loss. 3  loss of appetite... | 0     |
| symptoms of a dying mouse                   | Symptoms of Dog and Cat Poisoning. The symptoms of a poisoned pet depend on the type of poison he or she is exposed to... | 0     |
| symptoms of a dying mouse                   | The symptoms of mites include: excessive scratching (other than grooming) fur loss. loss of appetite. raw or cut skin. scabby skin... | 0     |
| symptoms of a dying mouse                   | Seizures and neurologic symptoms are caused by several poisons including strychnine, tobacco, aspirin, antidepressants, alcohol, marijuana... | 0     |
| symptoms of a dying mouse                   | The symptoms are similar but the mouse will be in much worse condition: runny eyes; sneezing; wheezing; shaking; fluctuating body temperature... | 1     |
| symptoms of a dying mouse                   | The symptoms are similar but the mouse will be in much worse condition: 1  runny eyes. 2  sneezing. 3  wheezing. 4  shaking... | 0     |
| symptoms of a dying mouse                   | Depending on the poison ingested, a poisoned dog or cat can have gastrointestinal, neurologic, heart, liver, and kidney symptoms... | 0     |
| symptoms of a dying mouse                   | Some plants also cause neurologic symptoms, including horse chestnuts and buckeyes. Bleeding and anemia from poisoning... | 0     |
| symptoms of a dying mouse                   | She described symptoms caused by permethrin: drooling (ptylism), staggering, vomiting, and depression. The family thought... | 0     |
| average number of lightning strikes per day | Lightning is a major cause of storm related deaths in the U.S. A lightning strike can result in a cardiac arrest (heart stopping) at the time of the injury... | 0     |
| average number of lightning strikes per day | Quick Answer. Lightning strikes reach the ground on Earth as much as 8 million times per day or 100 times per second, according to the National Severe Storms Laboratory... | 0     |
| average number of lightning strikes per day | An average lightning strike discharges about 30,000 amperes (20,000 amperes in the UK). The current in a lightning strike typically... | 0     |
| average number of lightning strikes per day | Lightning fatalities in the U.S.: A map of total lightning fatalities in the United States between 1959 and 2013. With 471 deaths... | 0     |
| average number of lightning strikes per day | Lightning is a sudden high-voltage discharge of electricity that occurs within a cloud, between clouds, or between a cloud and the ground... | 0     |
| average number of lightning strikes per day | Although many lightning flashes are simply cloud-to-cloud, there are as many as 9,000,000 reported lightning strikes that damage buildings... | 1     |
| average number of lightning strikes per day | According to the NWS Storm Data, over the last 30 years (1986-2015) the U.S. has averaged 48 reported lightning fatalities per year... | 0     |
| average number of lightning strikes per day | Approximately 300,000 lightning strikes hit the ground in Britain each year with 30 percent of reported lightning strikes causing severe damage... | 0     |
| average number of lightning strikes per day | Florida is also the state with the highest number of deaths from lightning strikes. Other states along the Gulf of Mexico coast, such as Alabama... | 0     |
| average number of lightning strikes per day | There is estimated to be around 2,000 lightning storm active around the global at one time creating over 100 strikes per second... | 0     |

Every every datapoint consisted of a query, a passage and a binary label indicating whether the passage was relevant to the query or not. Each query in the dataset had 10 passages associated with it, out of which only one was relevant. The task was to build a model that would assign a relevance score (a real number) to each query/passage pair such that the the most relevant query/passage gets the highest score. 

The evaluation was done on the basis of Mean Reciprocal Rank (MRR) which is defined as follows:

$$
MRR=\frac{1}{|Q|}\sum_{i=1}^{|Q|} \frac{1}{rank_i}
$$

So, suppose for a query, each of the ten passages get respectively a score of 4.1, 3.4, 1.0, 1.9, 1.1, 0.8, 3.3, 5.6, 1.9, 2.7, in order, that is the first passage gets a score of 0.1, second passage gets a score of 3.4 and so forth. Now, the correct passage for the query is lets say is passage 7 (which got a score of 3.3). It's rank according to the scores that were assigned, was 4 (since the scores assigned to 3 other passages were higher than 3.3. The higher scores being 5.6, 4.1 and 3.4). Since the passage got a rank of 4, the Reciprocal Rank (RR) score for this query is 0.25 (1/4). The final score will be the mean of all RRs across all queries.

## First Attempt

When participating in a machine learning competition, my first line of attack is to go over the data, see if it needs any preprocessing and train a fast model over the data, just to test the waters.

The first model was a combination of two [Residual Networks](https://arxiv.org/as/1512.03385) in parallel, one to extract query features, and the other to extract passage features. The features extracted by both the networks were concatenated, passed through a fully connected layer, that resulted in a single activation, on which a sigmoid was applied, to give the classification score. A variant of Residual Network, called a [Wide Residual Network](https://arxiv.org/abs/1605.07146) was used. I used the code provided by [Brandon Morris](https://brandonlmorris.com/2018/06/30/wide-resnet-pytorch/), with some minor modifications, for this experiement. 

To convert text into features, so as to be able to feed it to the model, every word in the vocabulary was first assigned an integer ID. The text, which was now converted into an array of numbers, was then passed through an [Embedding Layer](https://pytorch.org/docs/stable/nn.html#embedding), to get a dense representation. I used the [fastai](https://docs.fast.ai/) library for this purpose.

Now, the only problem that remained was to handle the variable number of words for every data point. This was handled at both batch level and inside the model. Since data are fed to the model in batches, it is required that the dimensions of each data point in the batch must be the same, so that the GPU can perform computations over the entire batch efficiently. To do this, when creating the [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), a custom collate function can be written, which would tell the dataloader how it should handle data points having variable dimensions in a batch. I simply decided to pad them with `1` (since `1` was the ID of "unknown" token in the vocabulary). The custom collate function is as follows:

```python
def pad_collate(batch):

    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        # Use 1 for padding, since xxpad has an index of 1
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs



    queries = [item[0][0] for item in batch]
    passages = [item[0][1] for item in batch]
    targets = torch.FloatTensor([item[1] for item in batch])
    # merge sequences
    qseqs = merge(queries)
    pseqs = merge(passages)

    return (qseqs, pseqs), targets.unsqueeze(1)
```

Now, this fixed the problem within a batch. However, the length of sequences would vary across batches, since, what the above function does, is to pad all the sequences in the batch so that the length of all the sequences becomes equal to the one with the maximum length. To handle this, the output from the Embedding Layer was passed through Adaptive Pooling layers. More specifically, the output was passed through both an [AdaptiveMaxPooling](https://pytorch.org/docs/stable/nn.html#adaptivemaxpool2d) and an [AdaptiveAvgPooling](https://pytorch.org/docs/stable/nn.html#adaptiveavgpool2d) layer. 

Following is the architecture of the model:

```python
class WideResNetParallel(nn.Module):
    def __init__(self, vocab_size, pretrained_wts_pth, emb_dim, n_grps, N, k=1, drop=0.3, first_width=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        # Intitialize embedding to GloVe weights
        self.emb.weight.data.copy_(torch.from_numpy(np.load(pretrained_wts_pth)))
        self.adap_avg_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.adap_max_pool = nn.AdaptiveMaxPool2d((32, 32))
        self.qwrn = WideResNetOpen(n_grps, N, k, drop, first_width)
        self.pwrn = WideResNetOpen(n_grps, N, k, drop, first_width)
        self.linear = nn.Linear(20, 1)

        
    def forward(self, q, p):
        q = self.emb(q) # Get an Embedding Matrix for Query features
        qap = self.adap_avg_pool(q)  # Adaptive Average Pooling for Query 
        qmp = self.adap_max_pool(q)  # Adaptive Max Pooling for Query 
        q = torch.cat([qap, qmp], dim=1) # Concatenate Query features
        q = q.unsqueeze(1) 
        q = self.qwrn(q)  # Pass through WideResnet

        p = self.emb(p) # Get an Embedding Matrix for Passage features
        pap = self.adap_avg_pool(p) # Adaptive Average Pooling for Passage
        pmp = self.adap_max_pool(p) # Adaptive Max Pooling for Passage 
        p = torch.cat([pap, pmp], dim=1) # Concatenate Passage features
        p = p.unsqueeze(1)
        p = self.pwrn(p) # Pass through WideResnet
        
        pq = torch.cat([p, q], dim=1)  # Concatenate features for Query and Passage
        pq = self.linear(pq) # Pass through a Fully Connnected Layer
        
        return torch.sigmoid(pq) # Apply Sigmoid on the resulting activation
```
The above model was then trained on a subset of the data provided for the competition using [fastai](https://docs.fast.ai/) for four epochs.

Unfortunately, this model did not work very well. Possible reasons could be: a) the model was random and more training was needed by the model to learn useful features; b) there was heavy class imbalance.

## Second Attempt

To address the problem of the model randomly initialized, I used pre-trained [BERT](https://arxiv.org/abs/1810.04805) models that were released by Google in October, 2018. For the challenge, I used the [PyTorch implementation](https://github.com/huggingface/pytorch-pretrained-BERT) of BERT by [huggingface](https://huggingface.co/), with some minor modifications.

To address the problem of class imbalance, the dataset was downsampled. Since the dataset was huge (5.24 million rows), 50% of the matching pairs were chosen and 20% of non-matching pairs were chosen randomly.

First, a `bert-base-uncased` was finetuned on the downsampled data. Due to a huge amount of data (even after it was downsampled), the weights of the model were saved after every 5000 iterations. It was trained for 3 epochs, with each epoch taking around 7 hours. The model at the end of the third epoch was used to make a submission. It got a Mean Reciprocal Rank score of 0.678 (rounded to three places). Then, an ensemble of models from the third epoch and the second epoch got a score of 0.683. An ensemble of models from all the three epochs boosted the MRR score to 0.686. Ensembling was done by averaging the classification scores of each model. 

Since, training the model further and using it for submission did not prove much effective, I then switched to `bert-large-uncased` model. However, I had a GTX 1050 Ti on my personal computer, therefore, training a `bert-large-uncased` on my machine was not feasible. I used [Google Compute Platform](https://cloud.google.com/compute/) to train the model (a big shout-out to Google for providing me with $300 worth of free credits). The model was trained on a Tesla V100 on GCP. The first model was trained for two epochs using the same data. It obtained a score of 0.704 on the leaderboard. It was clear from this, that the large model had more representational power than the base model. 

To make use of as much data as possible, I formed two partially overlapping subsets of the data using different seeds and trained two models in parallel, saving the weights after every 5000 steps, as before. Different combinations of the resulting instances of these models were used to create ensembles. **I made a submission using an ensemble of three of the instances got a score of 0.710 on the leaderboard in the first phase, mintues before the final submission dealine, which got me the third place in the first phase.**

The same ensemble was used for the second phase (the evaluation data was different for this phase). That got me in the top 20!

## Closing Thoughts

It was really fun. I learned a lot. I learned to create a dataloader with a custom collate function, explored more of fastai library, PyTorch, used GCP, for which I had to understand static IPs, creating and terminating instances, quotas, billing and what not. Finally, the power of transfer learning never ceases to amaze me. Picking a pretrained model and fintuning it can take you quite head than building everything from scratch. 

Looking forward to the next challenge.

I will share the code and the weights of the final models in a GitHub repository soon.  

{% include disqus.html %}