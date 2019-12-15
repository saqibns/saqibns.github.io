---
layout: post
title: "Machine Learning Challenges in 2020"
excerpt: "Try your hands on Machine Learning challenges organized under the umbrella of various conferences."
tags:
  - conferences
  - competition
  - list
last_modified_at: 2019-12-15T13:32:41+05:30
---

Competitions are a great way to excel in machine learning. They offer various advantages in addition to gaining knowledge and developing your skill-set. 

The problems and goals are very well defined. This saves you from the hassle of coming up with a problem, defining the goals rigorously, which are both achievable and non-trivial. You are also provided with data, which in most cases is ready for use. Someone has already done the painstaking work of collecting, preprocessing and organizing data. If it's a competition on supervised learning, you also get labels for the data.

If you're a procrastinator, you have deadlines to your rescue. They keep you focused and prevent you from going astray ;) 

Competition leaderboards (if the competition has one), push you to do better. They keep things in perspective by giving continuous feedback on how you're doing relative to others. You struggle to find better solutions, try to surpass yourself, and in the process keep growing.

Finally, the rewards. They come in various forms. Monetary rewards are one. The satisfaction of solving a challenging problems and growing is another. But the main motivation for writing this post is the third kind of reward. If you're a top performer in a competition organized under a conference, you get a chance to publish your results.

I was looking for a curated list of such competitions but couldn't find any. So, decided to make one. The table below summarizes all the competitions I could find. They have been ordered according to their deadlines. I plan on updating the list on a regular basis. As more conferences release information about the competitions on their website, I'll add them to the list.

If you know of any competition that is not on the list, please let me know in the comments or feel free to send a pull request.

| Name                                     | Conference                               | Starts                 | Ends                | Website                                  | Sub-Challenges |
| ---------------------------------------- | ---------------------------------------- | ---------------------- | ------------------- | ---------------------------------------- | -------------------- |
| <a name="herohe-table"></a>[HEROHE](#herohe-ecdp) | [ECDP](http://ecdp2020.org/) | 1st October, 19 | 15th January | [Link](https://ecdp2020.grand-challenge.org/) | - |
| <a name="ldnb-table"></a>[LNDb Challenge](#lndb-iciar)       | [ICIAR](https://www.aimiconf.org/iciar20/) | 20th November, 19 | 17th February | [Link](https://lndb.grand-challenge.org/)     | 03             |
| <a name="ntire-table"></a>[NTIRE](#ntire-cvpr) | [CVPR](http://cvpr2020.thecvf.com/) | 17th December, 19 | 21st February | [Link](http://www.vision.ee.ethz.ch/ntire20/) | 07 |
| <a name="monusac-table"></a>[MoNuSAC](#monusac-isbi) | [ISBI](http://2020.biomedicalimaging.org/) | 15th November, 19 | 25th February | [Link](https://monusac-2020.grand-challenge.org/) | - |
| <a name="endocv2020-table"></a>[Cell Tracking Challenge](#endocv-isbi) | [ISBI](http://2020.biomedicalimaging.org/) | 10th November, 19 | 1st March     | [Link](http://celltrackingchallenge.net/)     | 02             |
| <a name="celltracking-table"></a>[EndoCV2020](#celltracking-isbi) | [ISBI](http://2020.biomedicalimaging.org/) | 1st November, 19  | 6th March     | [Link](https://endocv.grand-challenge.org/)   | 02             |
| <a name="clic-table"></a>[Challenge on Learned Image Compression](#clic-cvpr) | [CVPR](http://cvpr2020.thecvf.com/)        | 22nd November, 19 | 20th March    | [Link](http://www.compression.cc/)                           | 02             |
| <a name="webvision-table"></a>[WebVision](#webvision-cvpr) | [CVPR](http://cvpr2020.thecvf.com/) | 1st March | 7th June | [Link](https://www.vision.ee.ethz.ch/webvision/challenge.html) | - |
| <a name="spacenet-table"></a>[SpaceNet](#spacenet-cvpr) | [CVPR](http://cvpr2020.thecvf.com/) | TBD | TBD | [Link](https://www.grss-ieee.org/earthvision2020/challenge.html) | - |



## <a name="herohe-ecdp"></a>HEROHE

> Unlike previous Challenges that evaluated the staining patterns present in IHC, this Grand Challenge new edition proposes to find an image analysis algorithm to identify with high sensitivity and specificity HER2 positive BC from HER2 negative BC specimens evaluating only the morphological features present on the hematoxylin and eosin (HE) slide.

[Challenge Website](https://ecdp2020.grand-challenge.org/) &#124; [Back](#herohe-table)



## <a name="lndb-iciar"></a>LNDb Challenge

> The main goal of this challenge is the **automatic classification of chest CT scans according to the** [**2017 Fleischner society pulmonary nodule guidelines**](https://pubs.rsna.org/doi/full/10.1148/radiol.2017161659) for patient follow-up recommendation. The Fleischner guidelines are widely used for patient management in the case of nodule findings, and are composed of 4 classes, taking into account the number of nodules (single or multiple), their volume (<100mm³, 100-250mm³ and ⩾250mm³) and texture (solid, part solid and ground glass opacities (GGO)). Furthermore, **three additional sub-challenges** will be held related to the different tasks needed to calculate a Fleischner score.

[Challenge Website](https://lndb.grand-challenge.org/) &#124; [Back](#ldnb-table)



## <a name="endocv-isbi"></a>EndoCV2020

> Endoscopy is a widely used clinical procedure for the early detection of numerous cancers (e.g., nasopharyngeal, oesophageal adenocarcinoma, gastric, colorectal cancers, bladder cancer etc.), therapeutic procedures and minimally invasive surgery (e.g., laparoscopy). During this procedure an endoscope is used; a long, thin, rigid or flexible tube with a light source and camera at the tip to visualise the inside of affected organs on an external screen. Quantitative clinical endoscopy analysis is immensely challenging due to inevitable video frame quality degradation from various imaging artefacts to the non-planar geometries and deformations of organs.
>
> After a great success of Endoscopy Artefact Detection challenge (EAD2019), this year EndoCV2020 is introduced with two sub-challenge themes this year.
>
> Each sub-challenge consists of **detection, semantic segmentation and out-of-sample generalisation tasks** for each unique dataset.

[Challenge Website](https://endocv.grand-challenge.org/) &#124; [Back](#endocv2020-table)



## <a name="celltracking-isbi"></a>Cell Tracking Challenge

> The fifth challenge edition will be organized as part of [ISBI 2020](http://2020.biomedicalimaging.org/challenges), taking place in Iowa City in April 2020. In this edition, the scope of the challenge will be broadened by adding two [bright-field microscopy datasets](http://celltrackingchallenge.net/2d-datasets) and one fully 3D+time dataset of developing Tribolium Castaneum embryo. Furthermore, silver segmentation ground truth corpora will be released for the training videos of nine existing datasets to facilitate the tuning of competing methods. The submissions will be evaluated and announced at the corresponding ISBI 2020 challenge workshop according to the ISBI 2020 challenge schedule, with a paper that reports on the results collected since the third edition being published in a top-tier journal afterward.

[Challenge Website](http://celltrackingchallenge.net/) &#124; [Back](#celltracking-table)



## <a name="monusac-isbi"></a>Multi-organ Nuclei Segmentation And Classification Challenge

> In this challenge, participants will be provided with H&E stained tissue images of four organs with annotations of multiple cell-types including epithelial cells, lymphocytes, macrophages, and neutrophils. Participants will use the annotated dataset to develop computer vision algorithms to recognize these cell-types from the tissue images of unseen patients released in the testing set of the challenge. Additionally, all cell-types will not have equal number of annotated instances in the training dataset which will encourage participants to develop algorithms for learning from imbalanced classes in a few shot learning paradigm. 

[Challenge Website](https://monusac-2020.grand-challenge.org/) &#124; [Back](#monusac-table)



## <a name="clic-cvpr"></a>Challenge on Learned Image Compression

> We host a lossy image and video compression challenge which specifically targets methods which have been traditionally overlooked, with a focus on neural networks, but we also welcome traditional approaches. Such methods typically consist of an encoder subsystem, taking images/videos and producing representations which are more easily compressed than pixel representations (e.g., it could be a stack of convolutions, producing an integer feature map), which is then followed by an arithmetic coder. The arithmetic coder uses a probabilistic model of integer codes in order to generate a compressed bit stream. The compressed bit stream makes up the file to be stored or transmitted. In order to decompress this bit stream, two additional steps are needed: first, an arithmetic decoder, which has a shared probability model with the encoder. This reconstructs (losslessly) the integers produced by the encoder. The last step consists of another decoder producing a reconstruction of the original images/videos.

[Challenge Website](http://challenge.compression.cc/motivation/) &#124; [Back](#clic-table)



## <a name="webvision-cvpr"></a>WebVision

> The WebVision dataset is composed of training, validation, and test set. The training set is downloaded from Web without any human annotation. The validation and test set are human annotated, where the labels of validation data are provided but the labels of test data are withheld. To imitate the setting of learning from web data, the participants are required to learn their models solely on the training set and submit classification results on the test set. The validation set could only be used to evaluate the algorithms during development (see details in Honor Code). Each submission will produce a list of 5 labels in the descending order of confidence for each image. The recognition accuracy is evaluated based on the label which best matches the ground truth label for the image.

[Challenge Website](https://www.vision.ee.ethz.ch/webvision/challenge.html) &#124; [Back](#webvision-table)



## <a name="ntire-cvpr"></a>NTIRE

> Image restoration, enhancement and manipulation are key computer vision tasks, aiming at the restoration of degraded image content, the filling in of missing information, or the needed transformation and/or manipulation to achieve a desired target (with respect to perceptual quality, contents, or performance of apps working on such images). Recent years have witnessed an increased interest from the vision and graphics communities in these fundamental topics of research. Not only has there been a constantly growing flow of related papers, but also substantial progress has been achieved.
>
> Each step forward eases the use of images by people or computers for the fulfillment of further tasks, as image restoration, enhancement and manipulation serves as an important frontend. Not surprisingly then, there is an ever growing range of applications in fields such as surveillance, the automotive industry, electronics, remote sensing, or medical image analysis etc. The emergence and ubiquitous use of mobile and wearable devices offer another fertile ground for additional applications and faster methods.

[Challenge Website](http://www.vision.ee.ethz.ch/ntire20/) &#124; [Back](#ntire-table)

## <a name="spacenet-cvpr"></a>SpaceNet

> In the SpaceNet 6 challenge, participants will be asked to automatically extract building footprints with computer vision and artificial intelligence (AI) algorithms using a combination of these two diverse remote sensing datasets. For training data, participants will be allowed to leverage both the electro-optical and SAR datasets. However, for testing models and scoring performance only a subset of the data will be made available. We hope that such a structure will incentivize new data fusion methods and other approaches such as domain adaptation.

[Challenge Website](https://www.grss-ieee.org/earthvision2020/challenge.html) &#124; [Back](#spacenet-table)