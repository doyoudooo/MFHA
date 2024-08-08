Multimodal Feature Heterogeneous Attack: Enhancing the Efficacy of Adversarial Transferability of Vision-Language Pre-training Models(MFHA)

## Brief Introduction
Vision-Language Pre-training (VLP) models have achieved significant success in multimodal tasks but have exhibited vulnerability to adversarial examples. Although adversarial attacks are harmful, they are valuable in revealing the weaknesses of VLP models and enhancing their robustness. 

However, due to the insufficient exploitation of modal differences and consistent features by existing methods, the attack effectiveness and transferability of adversarial examples are not optimal.To address this issue, we propose the Multimodal Feature Heterogeneous Attack (MFHA) framework. To enhance the attack capability, We propose a feature heterogeneity method based on triple contrastive learning. First, we use data enhancement crossmodal guidance to attack different modal features. Then, within the same modality, we let the adversarial samples learn the difference feature of other different samples, losing their original  features. Finally, we use the global features of the text modality and the local features of the image modality to interfere with each other, further widening the feature difference between modalities, thereby obtaining stronger attack capabilities.To promote transferability, we propose a multi-domain feature perturbation method based on cross-modal variance aggregation. First, in both spatial and frequency domains, text-modal-guided feature attacks are employed to compute the dual-sampling aggregated gradient variance information. Then, by combining the gradient momentum information from the previous iteration, the consistent features of the modalities are disturbed, resulting in improved transferability.Extensive experiments conducted under various settings demonstrate the significant advantage of our proposed MFHA in terms of transferable attack capability, with an average improvement of 16.05\%.Furthermore, we emphasize that our MFHA also exhibits outstanding attack performance on large-scale multimodal models such as MiniGPT4 and LLaVA.

## Quick Start 
### 1. Install dependencies
See in `requirements.txt`.

### 2. Prepare datasets and models
Download the datasets, [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) and [MSCOCO](https://cocodataset.org/#home) (the annotations is provided in ./data_annotation/). Set the root path of the dataset in `./configs/Retrieval_flickr.yaml, image_root`.  
The checkpoints of the fine-tuned VLP models is accessible in [ALBEF](https://github.com/salesforce/ALBEF), [TCL](https://github.com/uta-smile/TCL), [CLIP](https://huggingface.co/openai/clip-vit-base-patch16).

### 3. Attack evaluation
From ALBEF to TCL on the Flickr30k dataset:
```python
python eval_albef2tcl_flickr.py --config ./configs/Retrieval_flickr.yaml \
--source_model ALBEF  --source_ckpt ./checkpoint/albef_retrieval_flickr.pth \
--target_model TCL --target_ckpt ./checkpoint/tcl_retrieval_flickr.pth \
--original_rank_index ./std_eval_idx/flickr30k/ --scales 0.5,0.75,1.25,1.5
```

From ALBEF to CLIP<sub>ViT</sub> on the Flickr30k dataset:
```python
python eval_albef2clip-vit_flickr.py --config ./configs/Retrieval_flickr.yaml \
--source_model ALBEF  --source_ckpt ./checkpoint/albef_retrieval_flickr.pth \
--target_model ViT-B/16 --original_rank_index ./std_eval_idx/flickr30k/ \
--scales 0.5,0.75,1.25,1.5
```

From CLIP<sub>ViT</sub> to ALBEF on the Flickr30k dataset:
```python
python eval_clip-vit2albef_flickr.py --config ./configs/Retrieval_flickr.yaml \
--source_model ViT-B/16  --target_model ALBEF \
--target_ckpt ./checkpoint/albef_retrieval_flickr.pth \
--original_rank_index ./std_eval_idx/flickr30k/ --scales 0.5,0.75,1.25,1.5
```

From CLIP<sub>ViT</sub> to CLIP<sub>CNN</sub> on the Flickr30k dataset:
```python
python eval_clip-vit2clip-cnn_flickr.py --config ./configs/Retrieval_flickr.yaml \
--source_model ViT-B/16  --target_model RN101 \
--original_rank_index ./std_eval_idx/flickr30k/ --scales 0.5,0.75,1.25,1.5
```


## Transferability Evaluation
The performance of SGA on four VLP models (ALBEF, TCL, CLIP<sub>ViT</sub> and CLIP<sub>CNN</sub>), the Flickr30k dataset.

<table style="border-collapse: collapse; width: 100%;"> <thead> <tr><th rowspan="2">Source</th><th rowspan="2">Attack</th><th colspan="2">ALBEF</th><th colspan="2">TCL</th><th colspan="2">CLIP<sub>ViT</sub></th><th colspan="2">CLIP<sub>CNN</sub></th></tr> <tr><th>TR R@1</th><th>IR R@1</th><th>TR R@1</th><th>IR R@1</th><th>TR R@1</th><th>IR R@1</th><th>TR R@1</th><th>IR R@1</th></tr> </thead> <tbody> <!-- ALBEF --> <tr><td rowspan="6">ALBEF</td><td>PGD</td><td>52.45</td><td>58.65</td><td>3.06</td><td>6.79</td><td>8.96</td><td>13.21</td><td>10.34</td><td>14.65</td></tr> <tr><td>BERT-Attack</td><td>11.57</td><td>27.46</td><td>12.64</td><td>28.07</td><td>29.33</td><td>43.17</td><td>32.69</td><td>46.11</td></tr> <tr><td>Sep-Attack</td><td>65.69</td><td>73.95</td><td>17.6</td><td>32.95</td><td>31.17</td><td>45.23</td><td>32.82</td><td>45.49</td></tr> <tr><td>Co-Attack</td><td>77.16</td><td>83.86</td><td>15.21</td><td>29.49</td><td>23.6</td><td>36.48</td><td>25.12</td><td>38.89</td></tr> <tr><td>SGA</td><td><strong>97.24</strong></td><td>97.24</td><td>45.42</td><td>55.25</td><td>33.38</td><td>44.16</td><td>34.93</td><td>46.57</td></tr> <tr style="background-color: #DDDDDD;"><td>MFHA(ours)</td><td>97.18</td><td><strong>97.26</strong></td><td><strong>75.97</strong></td><td><strong>79.55</strong></td><td><strong>47.85</strong></td><td><strong>59.86</strong></td><td><strong>54.32</strong></td><td><strong>66.53</strong></td></tr> </tbody> <tbody> <!-- TCL --> <tr><td rowspan="6">TCL</td><td>PGD</td><td>6.15</td><td>10.78</td><td>77.87</td><td>79.48</td><td>7.48</td><td>13.72</td><td>10.34</td><td>15.33</td></tr> <tr><td>BERT-Attack</td><td>11.89</td><td>26.82</td><td>14.54</td><td>29.17</td><td>29.69</td><td>44.49</td><td>33.46</td><td>46.07</td></tr> <tr><td>Sep-Attack</td><td>20.13</td><td>36.48</td><td>84.72</td><td>86.07</td><td>31.29</td><td>44.65</td><td>33.33</td><td>45.8</td></tr> <tr><td>Co-Attack</td><td>23.15</td><td>40.04</td><td>77.94</td><td>85.59</td><td>27.85</td><td>41.19</td><td>30.74</td><td>44.11</td></tr> <tr><td>SGA</td><td>48.91</td><td>60.34</td><td><strong>98.37</strong></td><td>98.81</td><td>33.87</td><td>44.88</td><td>37.74</td><td>48.3</td></tr> <tr style="background-color: #DDDDDD;"><td>MFHA(ours)</td><td><strong>81.45</strong></td><td><strong>84.81</strong></td><td>98.33</td><td><strong>99.02</strong></td><td><strong>49.82</strong></td><td><strong>61.44</strong></td><td><strong>54.11</strong></td><td><strong>66.72</strong></td></tr> </tbody> <tbody> <!-- CLIP ViT --> <tr><td rowspan="6">CLIP<sub>ViT</sub></td><td>PGD</td><td>2.5</td><td>4.93</td><td>4.85</td><td>8.17</td><td>70.92</td><td>78.61</td><td>5.36</td><td>8.44</td></tr> <tr><td>BERT-Attack</td><td>9.59</td><td>22.64</td><td>11.8</td><td>25.07</td><td>28.34</td><td>39.08</td><td>30.4</td><td>37.43</td></tr> <tr><td>Sep-Attack</td><td>9.59</td><td>23.25</td><td>11.38</td><td>25.6</td><td>79.75</td><td>86.79</td><td>30.78</td><td>39.76</td></tr> <tr><td>Co-Attack</td><td>10.57</td><td>24.33</td><td>11.94</td><td>26.69</td><td>93.25</td><td>95.86</td><td>32.52</td><td>41.82</td></tr> <tr><td>SGA</td><td>13.4</td><td>27.22</td><td>16.23</td><td>30.76</td><td><strong>99.08</strong></td><td><strong>98.94</strong></td><td>38.76</td><td>47.79</td></tr> <tr style="background-color: #DDDDDD;"><td>MFHA(ours)</td><td><strong>23.67</strong></td><td><strong>39.66</strong></td><td><strong>26.69</strong></td><td><strong>42.17</strong></td><td>98.16</td><td>98.52</td><td><strong>58.37</strong></td><td><strong>64.74</strong></td></tr> </tbody> <tbody> <!-- CLIP_CNN --> <tr><td rowspan="6">CLIP<sub>CNN</sub></td><td>PGD</td><td>2.09</td><td>4.82</td><td>4.21</td><td>7.81</td><td>10.1</td><td>6.6</td><td>86.46</td><td>92.25</td></tr> <tr><td>BERT-Attack</td><td>8.86</td><td>23.27</td><td>12.33</td><td>25.48</td><td>27.12</td><td>37.44</td><td>30.4</td><td>40.1</td></tr> <tr><td>Sep-Attack</td><td>8.55</td><td>23.41</td><td>12.64</td><td>26.12</td><td>28.34</td><td>39.43</td><td>91.44</td><td>95.44</td></tr> <tr><td>Co-Attack</td><td>8.79</td><td>23.74</td><td>13.1</td><td>26.07</td><td>28.79</td><td>40.03</td><td>94.76</td><td>96.89</td></tr> <tr><td>SGA</td><td>11.42</td><td>24.8</td><td>14.91</td><td>28.82</td><td>31.24</td><td>42.12</td><td><strong>99.24</strong></td><td><strong>99.49</strong></td></tr> <tr style="background-color: #DDDDDD;"><td>MFHA(ours)</td><td><strong>17.83</strong></td><td><strong>33.87</strong></td><td><strong>20.41</strong></td><td><strong>37.12</strong></td><td><strong>43.93</strong></td><td><strong>58.39</strong></td><td>98.9</td><td>99.29</td></tr> </tbody> </table>

## Visualization

![image](https://github.com/user-attachments/assets/4abf677c-a416-48f5-a090-ebae35d66a99)






### Citation
Kindly include a reference to this paper in your publications if it helps your research:

