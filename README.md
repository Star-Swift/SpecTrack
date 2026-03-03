
SpecTrack: Spectral-Heterogeneous MoE for Multispectral Object Tracking
Anonymous Submission
Abstract
Multispectral object tracking leverages spectral sig- 1
natures to robustly handle camouflage and back- 2
ground clutter. However, current methods rely 3
on an inefficient uniform processing paradigm that 4
treats all regions equally. This approach squanders 5
resources on redundant backgrounds and lacks the 6
adaptive capacity needed for discriminatively crit- 7
ical regions, such as object boundaries and spec- 8
trally similar distractors. To address these lim- 9
itations, we propose SpecTrack, a novel frame- 10
work that dynamically aligns model capacity with 11
the local spectral-spatial complexity of the scene. 12
The core is the Spectral-Heterogeneous Mixture 13
of Experts (SHMoE), which adaptively reallocates 14
computational budget through a set of specialized 15
experts. Also, We introduce a Frequency Spec- 16
tral Prompt Router that integrates spatial semantics 17
with frequency aware spectral cues to identify hard 18
regions, directing them to capacity heavy experts 19
for intensive modeling while assigning lightweight 20
experts to easy backgrounds. Experimental results 21
show that SpecTrack outperforms state-of-the-art 22
trackers by 5.5% and 4.1% AUC on MUST and 23
MSITrack datasets, respectively while also demon- 24
strating robust generalization ability on the large- 25
scale RGB benchmark GOT-10k. For flexible de- 26
ployment, we introduce five variants catering to de- 27
vices from edge to GPUs, striking a good trade-off 28
between speed and accuracy. 29
1 Introduction 30
Visual Single Object Tracking (SOT) serves as a cornerstone 31
in computer vision systems [Li et al., 2025; Chen et al., 32
2022], underpinning critical applications ranging from UAV 33
surveillance [Kou et al., 2025] to autonomous driving [Yang 34
et al., 2023]. While RGB-based trackers have witnessed re- 35
markable progress [Ye et al., 2022; Li et al., 2025], their 36
reliance on visible bands often renders them fragile in sce- 37
narios involving background clutter or camouflage [Wang et 38
al., 2024]. In such cases, visual appearance alone is insuffi- 39
cient for robust discrimination. Multispectral Imaging (MSI) 40
Template Target
Partial Occlusion
Template Target
Partial Occlusion
Template Target
Similar Background
Template Target
Similar Background
Template Target Background Template Target Background
422.5nm 487.5nm 550nm 602.5nm 660nm 725nm 785nm 887.5nm
R G B
Template Target Background
422.5nm 487.5nm 550nm 602.5nm 660nm 725nm 785nm 887.5nm
R G B
Template Target Background Template Target Background
422.5nm 487.5nm 550nm 602.5nm 660nm 725nm 785nm 887.5nm
R G B
Template Target Background
422.5nm 487.5nm 550nm 602.5nm 660nm 725nm 785nm 887.5nm
R G B
50
150
250
50
150
250
50
150
250
50
150
250
50
150
250
50
150
250
50
150
250
Spectral
Spectral
RGB
RGB
Figure 1: In challenging scenarios, the spatial features (e.g., color
and texture) of the tracked target closely resemble those of the back￾ground, making differentiation and localization difficult. However,
the target’s spectral information differs significantly from the back￾ground and aligns with the template’s spectral data, providing robust
features for reliable tracking.
emerges as a compelling alternative by capturing material- 41
dependent reflectance across visible and near-infrared bands 42
[Xiong et al., 2020a]. As illustrated in Fig. 1, distinct spectral 43
signatures reveal the intrinsic physical properties of objects, 44
enabling reliable tracking even when the target and back- 45
ground exhibit indistinguishable colors or textures [Qin et al., 46
2025; Feng et al., 2025]. 47
Despite the promise of MSI, existing multispectral ob- 48
ject tracking methods largely follow a uniform processing 49
paradigm, employing identical feature extraction and re- 50
lation modeling strategies across all image regions. The 51
paradigm persists across various architectures, including en- 52
semble frameworks [Li et al., 2023; Li et al., 2020; Li et al., 53
2021], Siamese architectures [Liu et al., 2022; Chen et al., 54
2023c] and unified Transformers networks [Qin et al., 2025]. 55
Existing methods fail to account for the inherent imbalance 56
in discriminative difficulty across the scene. 57
In tracking scenarios, vast background regions often ex- 58
hibit semantic redundancy. Conversely, a small subset of 59
discriminatively critical regions, including object boundaries 60
and spectrally similar distractors, exhibits severe ambiguity. 61
Intensity Intensity Intensity Intensity
Unlike the redundant background, these regions demand deep 62
reasoning. Applying a uniform processing strategy to such di- 63
verse regions is inefficient. On the one hand, allocating heavy 64
computational resources to easy background tokens incurs 65
unnecessary redundancy, limiting the model’s efficiency. On 66
the other hand, simple interaction mechanisms shared glob- 67
ally may fail to disentangle the complex mixed pixel signa- 68
tures in hard regions, leading to diluted discriminative fea- 69
tures. These limitations motivate our question: How to de- 70
sign a tracker that adaptively allocates representational 71
capacity based on the local spectral-spatial complexity of 72
the scene? 73
To address this question, we propose SpecTrack, a novel 74
multispectral tracking framework that explicitly accounts for 75
spectral heterogeneity. Instead of previous methods that rely 76
on static architectures, SpecTrack dynamically adjusts its pro- 77
cessing strategy through a Spectral Heterogeneous Mixture of 78
Experts (SHMoE). Inspired by the success of MoE [Aljundi 79
et al., 2017; Tang et al., 2024] in handling diverse data pat- 80
terns, we employs a set of specialized experts with vary- 81
ing receptive fields and interaction granularities. Specifi- 82
cally, lightweight experts efficiently process confident back- 83
ground regions to conserve resources, while capacity-heavy 84
experts concentrate computation on resolving ambiguity in 85
hard foreground tokens and distractors.Furthermore, to pre- 86
vent the fragmentation of global context caused by sparse ex- 87
pert activation, we incorporate a spectral-domain global in- 88
teraction module to maintain long-range information propa- 89
gation. To ensure precise expert assignment, we design a Fre- 90
quency Spectral Prompt Router. Instead of merely detecting 91
high-frequency edges, this router integrates spatial seman- 92
tics with frequency-aware spectral cues to identify discrim- 93
inatively hard regions that require intensive modeling. 94
Benefiting from the adaptive resource allocation of 95
SHMoE, SpecTrack achieves a superior trade-off between 96
tracking robustness and computational efficiency. Contrary to 97
dense models that process all tokens with equal complexity, 98
our method focuses representational capacity on high value 99
regions, effectively suppressing background noise while 100
sharpening target features. We validate the effectiveness of 101
SpecTrack through extensive experiments on four diverse 102
benchmarks. Empirical results demonstrate that our frame- 103
work sets new state-of-the-art standards across multispectral 104
datasets (MUST [Qin et al., 2025], MSITrack [Feng et al., 105
2025]) and the hyperspectral benchmark (HOT2022 [Xiong 106
et al., 2020a]). Furthermore, we extend our evaluation to 107
the large-scale RGB dataset GOT-10k [Huang et al., 2021], 108
verifying the scalability and generalization capability of our 109
tracker in generic scenarios. As shown in Tab. 2, Tab. 3, 110
Tab. 4 and Tab. 5, SpecTrack-B224 achieves or surpasses the 111
performance of previous state-of-the-art trackers while main- 112
taining comparable inference speeds. 113
Our contributions are summarized as follows: (1) We iden- 114
tify the inefficiency of uniform processing in MSI track- 115
ing and propose SpecTrack, the first framework to intro- 116
duce MoE for adaptive spectral-spatial modeling. (2) We de- 117
sign the SHMoE module equipped with a Frequency Spec- 118
tral Prompt Router, which effectively decouples simple back- 119
grounds from complex foregrounds, aligning model capacity 120
with region-specific demands. (3) Extensive experiments on 121
three multispectral benchmarks demonstrate that SpecTrack 122
consistently outperforms state-of-the-art trackers, validating 123
the effectiveness of our adaptive modeling strategy
