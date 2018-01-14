---
layout: post
title: "A List of Machine Learning Challenges in 2018"
excerpt: "Try your hands on Machine Learning challenges organized under the umbrella of various conferences."
tags:
  - conferences
  - competition
  - list
last_modified_at: 2018-01-14T12:25:36+05:30
---

Competitions are a great way to excel in machine learning. They offer various advantages in addtion to gaining knowledge and developing your skillset. 

The problems and goals are very welll defined. This saves you from the hassle of coming up with a problem, defining the goals rigorously, which are both achievable and non-trivial. You are also provided with data, which in most cases is ready for use. Someone has already done the painstaking work of collecting, preprocessing and organizing data. If it's a competition on supervised learning, you also get labels for the data.

If you're a procrastinator, you have deadlines to your rescue. They keep you focused and prevent you from going astray ;) 

Competition leaderboards (if the competition has one), push you to do better. They keep things in perspective by giving continuous feedback on how you're doing relative to others. You struggle to find better solutions, try to surpass yourself, and in the process keep growing.

Finally, the rewards. They come in various forms. Monetary rewards are one. The satisfaction of solving a challenging problems and growing is another. But the main motivation for writing this post is the third kind of reward. If you're a top performer in a competition organized under a conference, you get a chance to publish your results.

I was looking for a curated list of such competitions but couldn't find any. So, decided to make one. The table below summarizes all the competitons I could find. They have been ordered according to their deadlines. I plan on updating the list on a regular basis. As more conferences release information about the competitions on their website, I'll add them to the list.

If you know of any competition that is not on the list, please let me know in the comments or feel free to send a pull request.

| Name                                     | Conference                               | Starts               | Ends          | Website                                  | Sub-<br />Challenges |
| ---------------------------------------- | ---------------------------------------- | -------------------- | ------------- | ---------------------------------------- | -------------------- |
| <a name="ntire-cvpr-table"></a>[New Trends in Image Restoration and Enhancement (NTIRE) Challenge](#ntire-cvpr) | [CVPR](http://cvpr2018.thecvf.com/program/workshops) | 10th January         | 27th February | [Link](http://www.vision.ee.ethz.ch/en/ntire18/) | 3                    |
| <a name="ug2-table"></a>[UG<sup>2</sup> Prize Challenge](#ug2) | [CVPR](http://cvpr2018.thecvf.com/program/workshops) | 15th January         | 2nd April     | [Link](http://www.ug2challenge.org/)     | 2                    |
| <a name="clic-table"></a>[Challenge on Learned Image Compression (CLIC)](#clic) | [CVPR](http://cvpr2018.thecvf.com/program/workshops) | 24th December, '17   | 22nd April    | [Link](http://www.compression.cc/challenge/) | -                    |
| <a name="landmark-table"></a>[Large-Scale Landmark Recognition](#landmark) | [CVPR](http://cvpr2018.thecvf.com/program/workshops) | 1st January          | 1st May       | [Link](https://landmarkscvprw18.github.io/) | -                    |
| <a name="robust-vision-table"></a>[Robust Vision Challenge](#robust-vision) | [CVPR](http://cvpr2018.thecvf.com/program/workshops) | 1st February         | 15th May      | [Link](http://www.robustvision.net/)     | 6                    |
| <a name="activitynet-table"></a>[ActivityNet Large-Scale Activity Recognition Challenge](#activitynet) | [CVPR](http://cvpr2018.thecvf.com/program/workshops) | 7th December, '17    | TBA           | [Link](http://activity-net.org/challenges/2018/index.html) | 7                    |
| <a name="kddcup-table"></a>[KDD Cup](#kddcup) | [KDD](http://www.kdd.org/kdd2018/)       | 1st March (Expected) | TBA           | [Link](http://www.kdd.org/News/view/kdd-cup-2018-call-for-proposals) | -                    |
| <a name="djirobomaster-table"></a>[DJI RoboMaster AI Challenge](#djirobomaster) | [ICRA](http://icra2018.org/)             | TBA                  | TBA           | [Link](http://icra2018.org/dji-robomaster-ai-challenge/) | -                    |
| <a name="microrobotics-table"></a>[Mobile Microrobotics Challenge](#microrobotics) | [ICRA](http://icra2018.org/)             | TBA                  | TBA           | [Link](http://icra2018.org/mobile-microrobotics-challenge-2018/) | 3                    |
| <a name="interspeech-table"></a>[Interspeech Computational Paralinguistics ChallengE (ComParE)](#interspeech) | [Interspeech](http://interspeech2018.org/) | TBA                  | TBA           | [Link](http://compare.openaudio.eu/)     | -                    |
| <a name="aicity-table"></a>[Nvidia AI City Challenge](#aicity) | [CVPR](http://cvpr2018.thecvf.com/program/workshops) | TBA                  | TBA           | [Link](https://www.aicitychallenge.org/) | -                    |
| <a name="lowpowerir-table"></a>[Low-Power Image Recognition Challenge](#lowpowerir) | [CVPR](http://cvpr2018.thecvf.com/program/workshops) | TBA                  | TBA           | [Link](https://rebootingcomputing.ieee.org/lpirc) | -                    |
| <a name="lip-table"></a>[The Look Into Person (LIP) Challenge](#lip) | [CVPR](http://cvpr2018.thecvf.com/program/workshops) | TBA                  | TBA           | [Link](https://vuhcs.github.io/)         | -                    |
| <a name="davis-table"></a>[DAVIS Challenge on Video Object Segmentation](#davis) | [CVPR](http://cvpr2018.thecvf.com/program/workshops) | TBA                  | TBA           | [Link](http://davischallenge.org/challenge2018/) | -                    |
| <a name="tidyup-table"></a>[Tidy Up My Room Challenge](#tidyup) | [ICRA](http://icra2018.org/)             | TBA                  | TBA           | [Link](http://icra2018.org/tidy-up-my-room-challenge/) | -                    |



## <a name="ntire-cvpr"></a>New Trends in Image Restoration and Enhancement (NTIRE) Challenge 

> **NTIRE 2018 challenge on image super-resolution**
> In order to gauge the current state-of-the-art in (example-based) single-image super-resolution under realistic conditions, to compare and to promote different solutions we are organizing an NTIRE challenge in conjunction with the CVPR 2018 conference.
>
> **The challenge has 3 tracks:**
>
> **Track 1:** classic bicubic  uses the bicubic downscaling (Matlab imresize), the most common setting from the recent single-image super-resolution literature.
> **Track 2:** realistic mild adverse conditions  assumes that the degradation operators (emulating the image acquisition process from a digital camera) can be estimated through training pairs of low and high-resolution images. The degradation operators are the same within an image space and for all the images.
> **Track 3:** realistic difficult adverse conditions  assumes that the degradation operators (emulating the image acquisition process from a digital camera) can be estimated through training pairs of low and high-resolution images. The degradation operators are the same within an image space and for all the images.
> **Track 4:** realistic wild conditions assumes that the degradation operators (emulating the image acquisition process from a digital camera) can be estimated through training pairs of low and high images. The degradation operators are the same within an image space but DIFFERENT from one image to another. This setting is the closest to real "wild" conditions.
>
> **NTIRE 2018 challenge on image dehazing** 
> In order to gauge the current state-of-the-art in image dehazing for real haze as well as synthesized haze, to compare and to promote different solutions we are organizing an NTIRE challenge in conjunction with the CVPR 2018 conference. A novel dataset of real and synthesized hazy images with ground truth will be introduced with the challenge. It is the first image dehazing online challenge.
>
> **The challenge has 3 tracks:**
>
> **Track 1:** realistic haze uses synthesized hazy images, a common setting from the recent image dehazing literature.
> **Track 2:** real haze with ground truth
> **Track 3:** real haze with color reference
>
>
> **NTIRE 2018 challenge on spectral reconstruction from RGB images**
> In order to gauge the current state-of-the-art in spectral reconstruction from RGB images, to compare and to promote different solutions we are organizing an NTIRE challenge in conjunction with the CVPR 2018 conference. The largest dataset to date will be introduced with the challenge. It is the first spectral reconstruction from RGB images online challenge.
>
> **The challenge has 2 tracks:**
>
> **Track 1:** “Clean”  recovering hyperspectral data from uncompressed 8-bit RGB images created by applying a know response function to ground truth hyperspectral information.
> **Track 2:** “Real World”  recovering hyperspectral data from jpg-compressed 8-bit RGB images created by applying an unknown response function to ground truth hyperspectral information.
>

[Challenge Website](http://www.vision.ee.ethz.ch/en/ntire18/) &#124; [Back](#ntire-cvpr-table)



## <a name="ug2"></a>UG<sup>2</sup> Prize Challenge 

> What is the current state-of-the art for image restoration and enhancement applied to images acquired under less than ideal circumstances?
>
> Can the application of enhancement algorithms as a pre-processing step improve image interpretability for manual analysis or automatic visual recognition to classify scene content?
>
> The UG<sup>2</sup> Challenge seeks to answer these important questions for general applications related to computational photography and scene understanding. As a well-defined case study, the challenge aims to advance the analysis of images collected by small UAVs by improving image restoration and enhancement algorithm performance using the UG<sup>2</sup Dataset.

[Challenge Website](http://www.ug2challenge.org/) &#124; [Back](#ug2-table)



## <a name="clic"></a>Challenge on Learned Image Compression (CLIC) 

> Recent advances in machine learning have led to an increased interest in applying neural networks to the problem of compression.
> We propose hosting an image-compression challenge which specifically targets methods which have been traditionally overlooked, with a focus on neural networks (but also welcomes traditional approaches). Such methods typically consist of an encoder subsystem, taking images and producing representations which are more easily compressed than the pixel representation (e.g., it could be a stack of convolutions, producing an integer feature map), which is then followed by an arithmetic coder. The arithmetic coder uses a probabilistic model of integer codes in order to generate a compressed bit stream. The compressed bit stream makes up the file to be stored or transmitted. In order to decompress this bit stream, two additional steps are needed: first, an arithmetic decoder, which has a shared probability model with the encoder. This reconstructs (losslessly) the integers produced by the encoder. The last step consists of another decoder producing a reconstruction of the original image.

[Challenge Website](http://www.compression.cc/challenge/) &#124; [Back](#clic-table)



## <a name="landmark"></a>Large-Scale Landmark Recognition 

> This workshop is to foster research on image retrieval and landmark recognition by introducing a novel large-scale dataset, together with evaluation protocols. More details will be available soon.

[Challenge Website](https://landmarkscvprw18.github.io/) &#124; [Back](#landmark-table)



## <a name="robust-vision"></a>Robust Vision Challenge 

> The increasing availability of large annotated datasets such as Middlebury, PASCAL VOC, ImageNet, MS COCO, KITTI and Cityscapes has lead to tremendous progress in computer vision and machine learning over the last decade. Public leaderboards make it easy to track the state-of-the-art in the field by comparing the results of dozens of methods side-by-side. While steady progress is made on each individual dataset, many of them are limited to specific domains. KITTI, for example, focuses on real-world urban driving scenarios, while Middlebury considers indoor scenes and VIPER provides synthetic imagery in various weather conditions. Consequently, methods that are state-of-the-art on one dataset often perform worse on a different one or require substantial adaptation of the model parameters.
>
> The goal of this workshop is to foster the development of vision systems that are robust and consequently perform well on a variety of datasets with different characteristics. Towards this goal, we propose the Robust Vision Challenge, where performance on several tasks (eg, reconstruction, optical flow, semantic/instance segmentation, single image depth prediction) is measured across a number of challenging benchmarks with different characteristics, e.g., indoors vs. outdoors, real vs. synthetic, sunny vs. bad weather, different sensors. We encourage submissions of novel algorithms, techniques which are currently in review and methods that have already been published. 

[Challenge Website](http://www.robustvision.net/) &#124; [Back](#robust-vision-table)



## <a name="activitynet"></a>ActivityNet Large-Scale Activity Recognition Challenge 

> This challenge is the 3rd annual installment of the ActivityNet Large-Scale Activity Recognition Challenge, which was first hosted during CVPR 2016. It focuses on the recognition of daily life, high-level, goal-oriented activities from user-generated videos as those found in internet video portals.
>
> We are proud to announce that this year the challenge will hosts seven diverse tasks which aim to push the limits of semantic visual understanding of videos as well as bridging visual content with human captions. Three out of the seven tasks in the challenge are based on the [ActivityNet dataset](http://activity-net.org/), which was introduced in CVPR 2015 and organized hierarchically in a semantic taxonomy. These tasks focus on trace evidence of activities in time in the form of actionness/proposals, class labels, and [captions](http://cs.stanford.edu/people/ranjaykrishna/densevid/).

[Challenge Website](http://activity-net.org/challenges/2018/index.html) &#124; [Back](#activitynet-table)



## <a name="kddcup"></a>KDD Cup 

> SIGKDD-2018 will take place in London, UK in August 2018. The KDD Cup competition is anticipated to last for 2-4 months, and the winners will be notified by mid-June. The winners will be honored at the KDD conference opening ceremony and will present their solutions at the KDD Cup workshop during the conference. The winners are expected to be monetarily rewarded, with the first prize being in the ballpark of ten thousand dollars.

[Challenge Website](http://www.kdd.org/News/view/kdd-cup-2018-call-for-proposals) &#124; [Back](#kddcup-table)



## <a name="djirobomaster"></a>DJI RoboMaster AI Challenge 

> DJI started RoboMaster in 2015 as an educational robotics competition for talented engineers and scientists. The annual RoboMaster competition requires teams to build robots that use shooting mechanisms to battle with other robots. The performances of the robots are monitored by a specially designed referee system, converting projectile hits into health point deductions on hit robots. To visit past games and introductory videos visit <https://www.twitch.tv/robomaster>. To see the RoboMaster2018 promotional video, go to: <https://youtu.be/uI2uoV58pzQ>
>
> Each team will build 1 – 2 automatic AI robots. Robots will compete in a 5m x 8m arena, filled with various obstacles. Participants will design robots that autonomously shoot plastic projectiles. The objective is outcompeting advanced official DJI robots in a battle of the wits.

[Challenge Website](http://icra2018.org/dji-robomaster-ai-challenge/) &#124; [Back](#djirobomaster-table)



## <a name="microrobotics"></a>Mobile Microrobotics Challenge 

> The IEEE Robotics & Automations Society (RAS) Micro/Nano Robotics & Automation Technical Committee (MNRA) invites applicants to participate in the 2018 Mobile Microrobotics Challenge (MMC), in which microrobots on the order of the diameter of a human hair face off in tests of autonomy, accuracy, and assembly.
>
> Teams can participate in up to three events:
>
> 1. Autonomous Manipulation & Accuracy Challenge: Microrobots must autonomously manipulate micro-components around fixed obstacles to a desired position and orientation superimposed on the substrate.  The objective is to manipulate the objects as precisely as possible to their goal locations and orientations in the shortest amount of time.
> 2. Microassembly Challenge:  Microrobots must assemble multiple microscale components inside a narrow channel in a fixed amount of time. This task simulates anticipated applications of microassembly, including manipulation within a human blood vessel and the assembly of components in nanomanufacturing.
> 3. MMC Showcase & Poster Session: Each team has an opportunity to showcase and demonstrate any advanced capabilities and/or functionality of their microrobot system. Each participating team will get one vote to determine the Best in Show winner.

[Challenge Website](http://icra2018.org/mobile-microrobotics-challenge-2018/) &#124; [Back](#microrobotics-table)



## <a name="interspeech"></a>Interspeech Computational Paralinguistics ChallengE (ComParE) 

> The **Interspeech Computational Paralinguistics ChallengE (ComParE)** series is an open Challenge in the field of Computational Paralinguistics dealing with states and traits of speakers as manifested in their speech signal’s properties. The Challenges takes annually place at INTERSPEECH since 2009. Every year, we introduce new tasks as there still exists a multiplicity of not yet covered, but highly relevant paralinguistic phenomena. The Challenge addresses the Audio, Speech, and Signal Processing, Natural Language Processing, Artificial Intelligence, Machine Learning, Affective & Behavioural Computing, Human-Computer/Robot-Interaction, mHealth, Psychology, and Medicine communities, and any other interested participants.

[Challenge Website](http://compare.openaudio.eu/) &#124; [Back](#interspeech-table)



## <a name="aicity"></a>Nvidia AI City Challenge 

> There will be 1 billion cameras by 2020. Transportation is one of the largest segments that can benefit from actionable insights derived from data captured by these cameras. Between traffic, signaling systems, transportation systems, infrastructure, and transit, the opportunity for insights from these cameras to make transportation systems safer and smarter is immense. Unfortunately, there are several reasons why these potential benefits have not yet materialized for this vertical. Poor data quality, the lack of labels for the data, and the lack of high quality models that can convert the data into actionable insights are some of the biggest impediments to unlocking the value of the data. There is also need for platforms that allow for appropriate analysis from edge to cloud, which will accelerate the development and deployment of these models. The NVIDIA AI City Challenge Workshop at CVPR 2018 will specifically focus on ITS problems such as
>
> - Estimating traffic flow and volume
> - Leveraging unsupervised approaches to detect anomalies such as lane violation, illegal U-turns, wrong-direction driving. This is the only way to get the humans in the loop pay attention to meaningful visual information
> - Multi-camera tracking, and object re-identification in urban environments.

[Challenge Website](https://www.aicitychallenge.org/) &#124; [Back](#aicity-table)



## <a name="lowpowerir"></a>Low-Power Image Recognition Challenge 

> Detect all relevant objects in as many images as possible of a common test set from the ImageNet object detection data set within 10 minutes.

[Challenge Website (Old)](https://rebootingcomputing.ieee.org/lpirc) &#124; [Back](#lowpowerir-table)



## <a name="lip"></a>The Look Into Person (LIP) Challenge 

>  Developing solutions to comprehensive human visual understanding in the wild scenarios, regarded as one of the most fundamental problems in compute vision, could have a crucial impact in many industrial application domains, such as autonomous driving, virtual reality, video surveillance, human-computer interaction and human behavior analysis. For example, human parsing and pose estimation are often regarded as the very first step for higher-level activity/event recognition and detection. Nonetheless, a large gap seems to exist between what is needed by the real-life applications and what is achievable based on modern computer vision techniques. The goal of this workshop is to allow researchers from the fields of human visual understanding and other disciplines to present their progress, communication and co-develop novel ideas that potentially shape the future of this area and further advance the performance and applicability of correspondingly built systems in real-world conditions.
>
> To stimulate the progress on this research topic and attract more talents to work on this topic, we will also provide a first standard human parsing and pose benchmark on a new large-scale Look Into Person (LIP) dataset. This dataset is both larger and more challenging than similar previous ones in the sense that the new dataset contains 50,000 images with elaborated pixel-wise annotations with comprehensive 19 semantic human part labels and 2D human poses with 16 dense key points. The images collected from the real-world scenarios contain humans appearing with challenging poses and views, heavily occlusions, various appearances and low-resolutions. Details on the annotated classes and examples of our annotations are available at this link <http://hcp.sysu.edu.cn/lip/>.

[Challenge Website](https://vuhcs.github.io/) &#124; [Back](#lip-table)



## <a name="davis"></a>DAVIS Challenge on Video Object Segmentation 

> We present the 2017 DAVIS Challenge, a public competition specifically designed for the task of video object segmentation. Following the footsteps of other successful initiatives, such as ILSVRC and PASCAL VOC, which established the avenue of research in the fields of scene classification and semantic segmentation, the DAVIS Challenge comprises a dataset, an evaluation methodology, and a public competition with a dedicated workshop co-located with CVPR 2017. The DAVIS Challenge follows up on the recent publication of DAVIS (Densely-Annotated VIdeo Segmentation), which has fostered the development of several novel state-of-the-art video object segmentation techniques. In this paper we describe the scope of the benchmark, highlight the main characteristics of the dataset and define the evaluation metrics of the competition.

[Challenge Website](http://davischallenge.org/challenge2018/) &#124; [Back](#davis-table)



## <a name="tidyup"></a>Tidy Up My Room Challenge 

> Robust interaction in domestic settings is still a hard problem for most robots. These settings tend to be unstructured, changing and aimed at humans not robots. This makes the grasping and picking of a wide range of objects in a person’s home a canonical problem for future robotic applications. With this challenge, we aim to foster a community around solving these tasks in a holistic fashion, requiring a tight integration of perception, reasoning and actuation.
>
> Robotics is an integration discipline and significant efforts are put in by labs worldwide every year to build robotic systems, yet it is hard to compare and validate these approaches against each other. Challenges and competitions have provided an opportunity to benchmark robotic systems on specific tasks, such as pick and place, and driving. We envision this challenge to contain multiple tasks and to increase in complexity over the years.

[Challenge Website](http://icra2018.org/tidy-up-my-room-challenge/) &#124; [Back](#tidyup-table)



{% include disqus.html %}