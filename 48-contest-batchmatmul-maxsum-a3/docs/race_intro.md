【初赛】BatchMatmulMaxSum算子
一、赛题背景
在神经信息检索和RAG重排场景中，ColBERT使用Late Interaction计算query与document的相关性：先计算token级点积相似度，再对每个query token执行MaxSim，最后求和得到query-document分数。该流程由BatchMatMul、ReduceMax和ReduceSum组成，融合后可减少Kernel Launch与中间数据搬运。

本题要求基于PyTorch中torch.bmm、torch.amax和torch.sum的核心业务逻辑，采用Ascend C编程语言进行算子原生开发，在昇腾NPU硬件上实现一款高性能的BatchMatmulMaxSum算子。

二、算子功能描述
实现的BatchMatmulMaxSum算子需完成以下核心计算：

批量矩阵乘：对B组一一对应的矩阵执行BatchMatMul，计算query token与document token的点积相似度。
MaxSim归约：沿document-token维N取最大值，得到每个query token的最高相似度。
求和归约：沿query-token维M求和，输出每个query-document pair的相关性分数。
输入embedding的L2归一化由上游网络完成，本算子不执行L2归一化，也不接收padding、有效长度或mask输入。

算子的核心难点在于：在支持四种输入存储布局组合、非对齐尾块、较小Batch和最大可达8192的M/N/K维度时，协调Cube矩阵乘与Vector归约流水，降低相似度计算与两级归约之间的数据搬运开销。

三、核心定义与约束
3.1 参考算子
PyTorch原生算子：torch.bmm、torch.amax、torch.sum

等价python实现：

import torch


def batch_matmul_max_sum(x1_logical, x2_logical):
    # x1_logical: [B, M, K]
    # x2_logical: [B, K, N]
    similarity = torch.bmm(
        x1_logical.to(torch.float32),
        x2_logical.to(torch.float32),
    )
    max_sim = torch.amax(similarity, dim=-1)
    return torch.sum(max_sim, dim=-1, dtype=torch.float32)

上述代码描述固定逻辑shape下的计算语义。transposeX1和transposeX2只声明输入的storage shape，不参与数学计算，也不表示算子需要额外执行transpose操作。算子行为、计算结果需与上述PyTorch组合实现的数学语义对齐。标准golden使用输入实际存储值在FP64精度下计算，最后转换为FP32。

3.2 数学公式
参与BatchMatMul的逻辑矩阵固定为：

X
1
∈
R
B
×
M
×
K
,
X
2
∈
R
B
×
K
×
N
X 
1
​
 ∈R 
B×M×K
 ,X 
2
​
 ∈R 
B×K×N
 
阶段一——批量矩阵乘：

A
[
b
,
m
,
n
]
=
∑
k
=
0
K
−
1
X
1
[
b
,
m
,
k
]
×
X
2
[
b
,
k
,
n
]
,
A
∈
R
B
×
M
×
N
A[b,m,n]= 
k=0
∑
K−1
​
 X 
1
​
 [b,m,k]×X 
2
​
 [b,k,n],A∈R 
B×M×N
 
阶段二——MaxSim归约：

R
[
b
,
m
]
=
max
⁡
0
≤
n
<
N
A
[
b
,
m
,
n
]
,
R
∈
R
B
×
M
R[b,m]= 
0≤n<N
max
​
 A[b,m,n],R∈R 
B×M
 
阶段三——求和归约：

y
[
b
]
=
∑
m
=
0
M
−
1
R
[
b
,
m
]
=
∑
m
=
0
M
−
1
(
max
⁡
0
≤
n
<
N
A
[
b
,
m
,
n
]
)
,
y
∈
R
B
y[b]= 
m=0
∑
M−1
​
 R[b,m]= 
m=0
∑
M−1
​
 ( 
0≤n<N
max
​
 A[b,m,n]),y∈R 
B
 
第b个x1仅与第b个x2配对，不进行batch broadcast或跨batch笛卡尔积计算。

3.3 输入输出与属性总览
类型	参数名	类型	逻辑shape / storage shape	支持数据类型	数据格式	备注
INPUT(必选)	x1	tensor	逻辑shape固定为
(
B
,
M
,
K
)
(B,M,K)；storage shape在transposeX1=false时为
(
B
,
M
,
K
)
(B,M,K)，true时为
(
B
,
K
,
M
)
(B,K,M)	FLOAT16、BFLOAT16	ND	Query token embedding
INPUT(必选)	x2	tensor	逻辑shape固定为
(
B
,
K
,
N
)
(B,K,N)；storage shape在transposeX2=false时为
(
B
,
K
,
N
)
(B,K,N)，true时为
(
B
,
N
,
K
)
(B,N,K)	FLOAT16、BFLOAT16	ND	Document token embedding的矩阵乘布局
OUTPUT(输出)	y	tensor	
(
B
,
)
(B,)	FLOAT32	ND	每个query-document pair的相关性分数
ATTR(可选)	transposeX1	bool	-	BOOL	-	仅声明x1的storage shape，默认false
ATTR(可选)	transposeX2	bool	-	BOOL	-	仅声明x2的storage shape，默认false
3.4 关键输入约束
维度取值范围（均为正整数）：

1
≤
B
≤
64
1≤B≤64。
1
≤
M
≤
8192
1≤M≤8192。
1
≤
N
≤
8192
1≤N≤8192。
32
≤
K
≤
8192
32≤K≤8192，且
K
K为8的整数倍。
输入规模同时满足
B
×
M
×
K
≤
2
26
B×M×K≤2 
26
 和
B
×
N
×
K
≤
2
26
B×N×K≤2 
26
 。
形状约束：

x1和x2均为3维Tensor。
x1的逻辑形状固定为
(
B
,
M
,
K
)
(B,M,K)，x2的逻辑形状固定为
(
B
,
K
,
N
)
(B,K,N)；transposeX1和transposeX2仅用于确定对应输入的storage shape。
两个输入的B和K必须分别相等，不支持batch broadcast。
M、N建议为16的整数倍；非对齐场景必须正确处理尾块。
数据类型约束：

x1和x2的数据类型必须相同。
x1和x2仅支持FLOAT16、BFLOAT16。
K维点积、MaxSim和Sum Reduction采用FLOAT32累加或具有等效精度的实现。
其他约束：

不支持空Tensor，仅支持连续ND Tensor。
输入不包含NaN或正负Inf，输入元素允许为负数。
M和N均表示参与计算的实际token数，不提供padding、有效长度或mask输入。
算子不得修改输入。
3.5 核心属性说明
transposeX1（bool，默认false）：用于说明x1的storage shape。false时storage shape为
(
B
,
M
,
K
)
(B,M,K)；true时storage shape为
(
B
,
K
,
M
)
(B,K,M)。该属性不改变x1的逻辑shape。
transposeX2（bool，默认false）：用于说明x2的storage shape。false时storage shape为
(
B
,
K
,
N
)
(B,K,N)；true时storage shape为
(
B
,
N
,
K
)
(B,N,K)。该属性不改变x2的逻辑shape。
两个属性仅描述输入数据的存储布局，不改变逻辑BatchMatMul与输出结果的定义。

3.6 输出严格要求
形状约束：输出y的shape固定为
(
B
,
)
(B,)。
类型约束：输出y的数据类型固定为FLOAT32。
计算顺序：必须按照BatchMatMul → Max(N) → Sum(M)计算，Max与Sum不可交换。
有限性：所有输出均不得包含NaN或正负Inf。
3.7 特殊值处理规则
全负相似度：MaxSim的初始值必须为负无穷或归约行的首个元素，不得初始化为0；当某行全部为负数时仍需返回其中最大负数。
大K累加：K维点积采用FLOAT32累加或具有等效精度的实现，避免低精度长归约造成明显误差。
无效输入：题目用例不包含NaN、正负Inf、空Tensor或形状不匹配输入。
四、规则要求
数值一致性规则：给定相同输入与属性，算子多次执行结果应保持一致。

一一配对规则：x1与x2的batch维必须相同，第b组x1只与第b组x2计算，不允许batch broadcast。

归约顺序规则：计算顺序固定为先沿N维取最大值，再沿M维求和，不得改写为先求和再取最大值。

存储布局规则：transposeX1和transposeX2仅声明对应输入的storage shape，不表示执行转置操作；四种storage shape组合均需正确支持。

性能要求：在保证数值精度和正确性的前提下，充分利用NPU硬件特性：

合理利用Cube与Vector计算单元，设计BatchMatMul和两级归约的协同流水。
优化GM、L1、L0和UB之间的数据搬运与复用。
沿B维和/或M维合理分配多核任务，兼顾较小Batch、大M/N/K和尾块负载均衡。
鼓励使用Double Buffer或Ping-Pong机制重叠计算与搬运。
性能基线为BatchMatMul + ReduceMax + ReduceSum拆分实现的总耗时，按融合实现相对基线的加速比评分。
使用msprof采集Task Duration、Cube利用率、Vector利用率及MTE带宽利用率。
五、精度判断规则
精度要求：计算结果需满足以下精度误差要求：

float32：相对误差 < 1e-4，绝对误差 < 1e-4（双万分之一精度）
float16、bfloat16：相对误差 < 1e-3，绝对误差 < 1e-3（双千分之一精度）
int32：要求计算结果完全准确，无误差
六、得分规则
本次比赛共15个测试点，所有case点精度全部通过才会计分。
每个测试点单独计分，逻辑如下（T为最优性能，t为当前提交性能）：
100
1
+
log
⁡
1.5
t
T
1+log 
1.5
​
  
T
t
​
 
100
​
 
排行榜显示的最终分数为所有case得分的均值。若得分计算一致，则以提交时间进行排序，提交越早，排名越高。
七、示例说明
示例1：基础计算

x1：shape=(1,2,2)，值为[[[1,0],[0,1]]]。
x2：shape=(1,2,3)，值为[[[1,0,-1],[0,1,0]]]。
transposeX1=false，transposeX2=false。
BatchMatMul结果为[[[1,0,-1],[0,1,0]]]。
MaxSim结果为[[1,1]]，最终输出y=[2.0]。
示例2：转置存储布局

逻辑矩阵内容与示例1相同。
x1按
(
B
,
K
,
M
)
(B,K,M)物理布局输入，x2按
(
B
,
N
,
K
)
(B,N,K)物理布局输入。
transposeX1=true，transposeX2=true。
两个属性仅说明上述storage shape；x1和x2的逻辑shape仍分别为
(
B
,
M
,
K
)
(B,M,K)和
(
B
,
K
,
N
)
(B,K,N)。算子不额外交换维度，而是直接按照对应存储布局完成相同逻辑计算，输出仍为y=[2.0]。
示例3：全负相似度

x1：shape=(1,1,2)，值为[[[1,0]]]。
x2：shape=(1,2,2)，值为[[[-1,-2],[0,0]]]。
BatchMatMul结果为[[[-1,-2]]]，MaxSim结果为[[-1]]。
最终输出y=[-1.0]，不能因错误地将MaxSim初始值设为0而输出0。
