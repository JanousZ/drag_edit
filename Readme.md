deepspeed2 + accelerate训练 flux kontext lora（实验性）
export NCCL_P2P_DISABLE=1
unset http_proxy  
export http_proxy="http://127.0.0.1:7890"
unset https_proxy  
export https_proxy="http://127.0.0.1:7890"
cd Kontext_train_ds2

```bash
accelerate launch --config_file ./train/deepspeed.yaml --main_process_port 29607 ./train/train_ds2.py \
    --num_epochs 50 \
    --lr 1e-4 \
    --save_steps 1000 \
    --output_dir lora_ckpt_v3_1 \
    --lora_config ./train/lora_config_v3_1.json \
    > train.log 2>&1
```

```bash
export CUDA_VISIBLE_DEVICES=2
python test.py \
--use_lora \
--checkpoint_dir "./lora_ckpt_v3_1/checkpoint-7500" \
--dataset_type "drag" \
--dataset_jsonl "/mnt/disk1/datasets/drag_data/train_json/pexels_tdv2_all.jsonl" \
--reverse_direction \
--output_dir "./pexels_tdv2_rev"

python test.py \
--use_lora \
--checkpoint_dir "./lora_ckpt_v3_1/checkpoint-7500" \
--dataset_type "drag" \
--dataset_jsonl "/mnt/disk1/datasets/drag_data/train_json/OpenVid-1M_all.jsonl" \
--reverse_direction \
--output_dir "./OpenVid-1M_rev"

python test.py \
--use_lora \
--checkpoint_dir "./lora_ckpt_v3_1/checkpoint-7500" \
--dataset_type "dragbench" \
--output_dir "./bench_v3_1_7500"

# 用你手标的多点版本；没标到的样本会 fallback 到原 meta_data.pkl
python test.py --use_lora \
  --checkpoint_dir ./lora_ckpt_v3_1/checkpoint-7500 \
  --dataset_type dragbench \
  --annotation_variant meta_data_multi.pkl \
  --output_dir ./bench_v3_1_7500_multi

# 只跑你标过的（严格对照集，不 fallback）
python test.py --use_lora \
  --checkpoint_dir ./lora_ckpt_v3_1/checkpoint-7500 \
  --dataset_type dragbench \
  --annotation_variant meta_data_multi.pkl \
  --only_annotated \
  --output_dir ./bench_v3_1_7500_multi_only

python annotate_multipoints_web.py --only_missing
# 操作（窗口激活时）：
# 左键：依次放点，红色 src → 绿色 tgt 循环
# 右键：撤销最后一个点（含未完成的 pending src）
# c：清空当前样本所有新点
# r：把原始 meta_data.pkl 的点复制进来当起点
# s：保存当前 sidecar
# n / →：保存后下一张；p / ←：保存后上一张；k：丢弃改动后下一张
# q / Esc：保存后退出
```



#异步错误处理
当一个 GPU 节点发生 NCCL 错误时，其他节点能及时收到通知并优雅退出，而不是一直死等（卡死）。它让错误日志更清晰。
export NCCL_ASYNC_ERROR_HANDLING=1    

如果通信超时，程序会直接报错抛出异常，而不是永远卡在那。这对于定位哪一步通信出问题非常有用。
export NCCL_BLOCKING_WAIT=1

#调试与日志
启动时会打印大量底层细节
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_SUBSYS=INIT,P2P,GRAPH,NET

#每个rank卡在了那一行代码
当发生错误时，PyTorch 会尝试打印最近的通信记录，增加这个值可以让你看到更完整的错误链路。
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

#打印 PyTorch 分布式层面的详细轨迹
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO

#设置超时时间
export NCCL_COMM_BLOCKING=1
export NCCL_TIMEOUT=10000

#兼容性与网络路径限制
强制禁用显卡间的 P2P（Peer-to-Peer）直接通信
export NCCL_P2P_DISABLE=1

禁用 InfiniBand (IB) 网络
export NCCL_IB_DISABLE=1

指定 NCCL 通信使用的物理网卡名称
export NCCL_SOCKET_IFNAME=ens9f0

export NCCL_P2P_LEVEL=PIX
export NCCL_P2P_LEVEL=NODE
看来是P2P的NODE出了问题

#断点重续
accelerate launch --config_file ./train/deepspeed.yaml ./train/train_ds2.py --num_epochs 100 --lr 1e-4 --save_steps 500 \
                  --resume_lora_path "./lora_ckpt/checkpoint-4000/lora.safetensors"

问题1：
出图是很平滑的噪声，或者条纹型噪声，无图像内容
尝试1：
1.增加time shift
2.去掉RoPE offset
发现1：
loss的稳定值比之前从0.14下降到0.1以下，但是出图还是模糊噪声

问题2：
我发现好像不止是我自己的问题，回头尝试DreamOmni2的edit一样是出图噪声
猜测2：
估计是推理代码上有共同的问题，而我的推理代码是Dreamomni共用的，所以我要找出这个推理问题
1.已排除模型参数的问题，直接跑flux-kontext并无错误
2.把Dreamomni仓库原本的代码进行复制，替换掉pipeline_dreamomni2.py，发现无改善，证明不是该文件的问题
3.把Dreamomni仓库原本的代码进行复制，替换掉inference_edit.py，发现问题解决，定位问题在该文件
4.发现了0号gpu在跑Dreamomni2的时候很有问题，我在DreamOmni2的edit下做了实验，发现凡是用到了0号卡进行推理的，都会有或多或少的质量问题。
尝试2：
使用1号卡或者2号卡进行推理,发现问题依旧存在

问题3：
我尝试把我自己训练的lora导入DreamOmni2查看效果，提示有unexpect key。
猜测3：
难道我保存lora的时候dict不对？
尝试3：
首先我要打印出我保存的lora，然后跟pipe本身的lora进行名字上的比对，确认问题就是lora没有对齐

最后，我们使用训练时的导入lora方式，对lora进行一个导入，就可以了。

问题4：如果我们用回0号卡去跑，是否会出现Dreamomni的类似问题？
又没有这个问题了。那么那是DreamOmni2本身的问题吗？
发现了是balanced模式下的问题！！！！

尝试4：我对Flux-Kontext代码进行了调试，发现了balanced模式下，两个text encoder被分到了不同地方，导致device不统一，t5的编码输出为空，所以出噪声
结论4：不要用balanced模式，如果要分配请自己调整device

尝试5：对Qwen-Image-Edit进行同样调试，发现同样的问题，所以不可以使用balanced模式
解决办法：pipeline.enable_model_cpu_offload(gpu_id = gpu_id)，可以节省推理时的gpu显存使用


accelerate launch --config_file ./train/deepspeed.yaml ./train/train_ds2_NFT.py --num_epochs 5 --lr 1e-4 --save_steps 500 > train.log 2>&1
问题1：当使用cpu offload时，cpu内存不足爆炸导致训练无法进行
尝试1：关闭ds2配置中optimizer的cpu offload
结果：问题依旧存在，这是在导入dit时就已经cpu内存爆炸。
尝试2：从3个进程缩减为2个进程
结果：爆GPU内存，双卡A6000无法训练
尝试3：导入模型时直接bf16导入，隐患在于精度问题。
结果：CPU内存问题解决，爆GPU内存，因为text_encoder太大，所以估计要分阶段进行训练了。


关于训练卡死的问题，估计是IOMMU除了问题。

dmesg | grep -e DMAR -e IOMMU
如果发现 fault ，那么很可能是iommu阻拦了pcie的p2p

方案 B：在内核启动项中设置为 Passthrough（不关 BIOS）
如果你无法重启服务器进入 BIOS，或者需要保留虚拟化功能，请尝试修改系统内核参数：
编辑 grub 文件： sudo nano /etc/default/grub
找到 GRUB_CMDLINE_LINUX_DEFAULT 这一行，在引号内加入 intel_iommu=on iommu=pt。
iommu=pt 的意思是 "Passthrough"，它会让系统开启 IOMMU 但不对 PCIe 设备间的直接通信进行干预。
更新配置并重启： sudo update-grub sudo reboot

## wandb检查登录状态
wandb login
wandb login --relogin
打开网址，登录账号（gmail），获取API key。

## Accelerate卡死情况
情况1：使用键盘退出时显示 [rank0]: baton.wait()
原因分析：
编译锁死（File Baton）： 当 DeepSpeed 第一次运行或者配置发生变化时，它需要编译 CPUAdam 等加速算子。为了防止多个进程同时编译同一个文件，PyTorch 使用了一个叫 file_baton 的“接力棒”锁。
僵尸锁文件： 如果你之前的尝试失败了（比如崩溃或被强制强杀），那个锁文件（.lock）可能还残留在 /home/yanzhang/.cache/torch_extensions 或类似的目录里。
无限等待： 当前进程看到锁文件已经存在，以为别的进程正在编译，于是就乖乖地在那 time.sleep 等待。但实际上，那个“正在编译”的进程早就死了，所以你的程序会永远等下去。
解决方案：rm -rf /home/yanzhang/.cache/torch_extensions/*

## 推理时图像值为NAN
关注三个点：模型初始化、模型精度以及训推一致性，训推一致性主要是变量的维度、归一化、含义等是否对齐。

问题1：dit的points_embedder没有放进优化器！
解决1：加入优化器后，依旧NAN

训练时：
base_model = /mnt/disk1/models/FLUX.1-Kontext-dev
dit = FluxTransformer2DPointsModel.from_pretrained(base_model, subfolder="transformer") float32
vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae") float32
t5 = T5EncoderModel.from_pretrained(base_model, subfolder="text_encoder_2") bfloat16
clip = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder") bfloat16
t5_tokenizer = T5Tokenizer.from_pretrained(base_model, subfolder="tokenizer_2")
clip_tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
points_map_encoder_config_path = "/home/yanzhang/drag_edit/module/pme_config.json"
with open(points_map_encoder_config_path, "r") as f:
    points_map_encoder_config = json.load(f)
points_map_encoder = PointsMapEncoder(**points_map_encoder_config) float32

dit.points_embedder 权重kaiming初始化，bias设为0
dit.add_adapter
points_map_encoder Conv2d权重kaiming初始化，bias设为0；GroupNorm权重设为1，偏置设为0
AdamW优化器，学习率正常范围，可优化参数出现问题了！！！！！！！！！
dit的points_embedder没有放进优化器！！！！！！！！

再关注NAN出现的时间地点。
问题2：temb出现了NAN, 另外hidden_states，encoder_hidden_states有部分NAN以及超大数值（32次方）
points_emb倒是数值良好
继续深入分析temb NAN的造成原因：
1.pooled_projections 数值看起来正常
2.timestep 数值看起来正常，Transformer外部接口进来的是0到1，但是在进层之前会扩大1000倍,变为0到1000。
但是发现在训练时，只专注于比较大的timestep，这不太好，比例不正常，需要修改。
3.time_text_embed这个层的参数
训练时参数正常，推理时参数导入异常，由此可推断其余参数也未必正常导致NAN。
发现是dit.to_empty()导致模型参数的清空，遂删除该行。

## 版本
v1.0版本 
编码方式：单通道点编号标注 + 卷积网络提取特征 + 与原图像特征相加
文本：“drag the photo according to the points embedding”
数据：主要为人脸数据，有其余植物、面包、动作等少数补充，大概在三四百组的数量
训练参数：bs=1, lr=1e-4

v2.0版本
修改了dit.points_embedder层没有放入优化器的问题
修改了推理时dit.to_empty()导致的权重不正确问题
修改了文本为空文本，先尝试只通过points embedding作为引导条件
修改了时间步采样mu=1.1，避免过多大时间步而过少小时间步的时间采样
修改了pointsmapencoder最后输出时的空间对应问题,以及结构
修改了pointsmapencoder最后卷积层为零初始化
版本问题：
生成图像完全是随机的，不按照points指定的方向，不受控。考虑可能是：
1.可能是控制信号的过稀疏与信噪比失衡，即控制信号太微弱（类似于随机扰动了），以至于模型相当于没有接受到控制信号。
2.数据集内部的“强相关性”陷阱。模型发现，与其费力去理解点对之间复杂的几何位移，不如直接根据图像的整体特征（比如姿态）预测一个最可能的“平均脸朝向”。点对变成了虚假信号。模型认为只要输出一张合法的、不同朝向的脸就能降低 Loss，而不需要精确对齐那 10 个点。
3.PointMapEncoder 的表征能力问题

针对观点1，打印相加前的数值大小
Image Norm: 1840.0000
Points Norm: 7.1875
Ratio (Pts/Img): 0.3906%
从整体来看，控制信号确实很小，只占0.39％，可以忽略了
就算是核心点位能量占比，也就是5％左右
推理时修正：hidden_states = hidden_states + points_emb * 100.0 发现效果上来了
找到一个较为合适的初始值，添加一个可训练的scale。从手动的结果来看，设定scale为60。
但是在scale=60的情况下，训练出来的控制信号占比也弱了60倍甚至不止，所以这种训练是难以达到需求的。

v2.1版本
修改scale为1.0，并调节为可训练
破案了，在test.py导入模型时*0.01了，无语死了！所以重新训练！

v3版本
主要是针对v2.0版本，修改了训练数据。可以从dragbench看到，对于扭头这类的数据比较敏感，容易成功，以及微旋操作也是，但是其余很多badcase。

1.颜色，对比度明显发生变化
2.很多拉长、缩短、边缘形变等操作几乎无反应，或者是容易被识别成基于平移的drag
并且2无反应时通常伴随着1
3.不知道什么时候是拉长缩短，对drag语义可能不明晰
4.范围问题，有时候产生联动效应，例如变窄的同时也变短了，本来是局部的变化会扩散到全局
5.很容易发成重叠错误，例如当一条腿移动到另外一条腿附近时，直接合并成一条腿，又或者把两个东西平移到一起，直接撞混

考虑：
1.可能是数据集都是8个点训练，而dragbench是1到2个点的标注
2.可能是数据中比较缺少对拉长、缩短、边缘形变这类的数据，并且过拟合于现有数据集

Claude Code回答：
1.训练分布是"视频共动"，而非"局部编辑"。8 个跟踪点几乎总是一起运动（刚体/半刚体），disp_mean_ratio 中位 0.087、最大 0.47，说明主要是扭头、整体位移。模型学到的是"给我一组共动方向，我生成一张整体被搬运的帧"。所以 bench 的 1~2 点局部拉伸/缩短/形变会被解读成平移，并伴随全局联动（问题 2、4）。
2.点数与语义的 train-test 偏差。训练固定 8 点、值 1..8；bench 1~2 点、值只有 1..2。integer_index 用 scalar 值表达"第几号点"的身份，经 3 次 stride conv 后身份信号极弱（v2 时打印 Ratio ≈0.39%）；且"序号 i 的点到序号 i 的点"的配对只能靠位置临近 + 标量值暗示，encoder 需硬背这种对齐。点数越少，信号越稀，模型越会退化成"看图下手"。
3.训练/推理 prompt 不一致。train_ds2.py:330 训练用 ""，test.py:238 推理用 "Drag the image..."，CLIP/T5 嵌入完全不同，Kontext 的文本通路在推理时进了一条没训过的条件，很可能贡献了"颜色/对比度整体漂移"（问题 1）。
4.ROI 裁剪总是把"运动区"放满画面。dragdataset.py:45-58 让 base_side 覆盖所有点的 bbox，点始终占画面很大比例；bench 里点簇经常是局部小区域，分布从未在训练中出现。
5.帧间曝光/压缩漂移成了"监督"的一部分。两个真实视频帧天然有 AWB、曝光、JPEG 噪声差，模型被迫把这当作"denoise 过程中应当产生的颜色变化"，和问题 1 吻合。
6.Loss 全图均匀 MSE。没有对 drag 邻域加权，模型通过"大面积保持 + 局部模糊"就能拿到低 loss，不会被逼去精确对齐点位；这又造成"点附近物体撞一起就融合"（问题 5）。
7.单层、弱强度注入。points_emb*2.0 在输入处加一次，之后 57 个 block 没有再注入，很容易在深层被图像先验压制——这和你在 Readme 提到的"多层注入"猜想一致。

验证方法
1.点数消融：从同一条训练样本里随机取 k∈{1,2,3,8} 个点训练或推理同一 ckpt，画 bench 成功率 vs k。若 k=1,2 明显崩，说明 integer_index + 8 点过拟合是主因。
2.prompt 消融：推理时把 prompt 换成 ""（和训练一致），看问题 1 的色偏是否消失/大幅缓解。
3.单样本过拟合：挑一个 bench 的"拉长"样本做 500~1000 step 纯过拟合。学得动 ⇒ 是数据问题；学不动 ⇒ 是 encoding/注入 架构问题。
4.流场散度统计：对训练集 8 点计算位移的散度/尺度变化量（||tgt-src|| 的方差、bbox 面积比），对比 DragBench 分布。我预期训练分布几乎为零散度，bench 是长尾。
5.Encoder probe：固定一张图，画 1 点 vs 8 点输入下 points_emb 的空间 norm 图，观察稀疏点是否在编码后仍能定位。
6.Norm 监控：训练时继续打印 Ratio(Pts/Img)，看 v3 是否又退回到 ~0.4%。
7.Mask region loss 可视化：把 loss 空间分解成"点 5×5 邻域 vs 其他"，观察两者在训练后期是否都收敛——通常邻域 loss 会显著高于平均。

解决方向（按投入产出排序）
1.统一训推 prompt（1 行改动）——先消掉色偏基线。
2.随机点数 + 随机顺序训练：每步从 8 点里采 k∈[1,8] 和随机索引顺序；破掉"总是 8 点、固定排号"的过拟合，贴近 bench 分布。
3.换掉 integer_index：改用双通道（或多通道）Gaussian heatmap，src 和 tgt 各自一层；或直接渲染成稀疏光流图 (Δx, Δy) 两通道。这同时4.解决身份与配对歧义（README 已提"流场编码"方向）。
引入非刚体合成数据：对静态图施加 TPS / 薄板样条形变生成 (原图, 形变图, 稀疏锚点) 三元组；或用现成 DragDiffusion/DragonDiffusion 合成 pseudo-pair。这直接补齐"拉长、缩短、边缘形变"的数据缺口（你的猜想 2 成立）。
5.mask-weighted flow loss：在 drag 点周围一个 soft disk 内把 MSE 权重提高 5~10×，逼模型优先对齐点位。
6.多层注入：把 points_emb 同时加到前 N 个 FluxTransformerBlock 的输入或 temb modulation 上（或做一个 zero-init adapter）；README 提到的多层注入方向。
7.裁剪策略多样化：训练时加入"ROI 远大于点 bbox"的采样，让模型见过点只占画面小区域的样本，缓解 bench 局部 drag 的分布外问题。
8.帧对筛选：优先选相邻帧 + 强制曝光/色彩一致性（或先做直方图匹配再进数据集），去掉"视频色偏当监督"的副作用。
建议落地顺序：先做 1、2、3（都不用扩数据），用 bench 拉长/缩短子集做 A/B；若仍不动再做 4（合成数据）和 6（多层注入）。

新的主要怀疑
点位在 encoder 里就丢了身份。integer_index 把点写成标量值 1..N，但 PointsMapEncoder 第一层 ResNet 的 GroupNorm 会把稀疏标量的绝对幅值做空间归一（均值/方差），1..8 的"序号差异"大概率在 norm 后近似一致 → 模型没办法靠值识别"第几号点"。训练集也表现差，正好说明"靠序号配对"这条路本来就没走通。

src 与 tgt 没有真正配对。train_ds2.py:71-82 把 src、tgt 放进 2*B 批里独立过同一个 encoder，再在 sequence 维拼接。encoder 输出里并不显式包含"src 的 i 号点 → tgt 的 i 号点"这条边，只能靠 57 层 DiT attention 自己摸配对——对共动样本容易学，对拉伸/缩放/镜像这种强非刚体就崩。

点信号对视频帧对而言是冗余的。两帧连拍图本来就能被 Flux-Kontext 先验直接预测出第二帧（运动模糊/姿态/相机漂移都能脑补），点位 embedding 几乎不用看也能让 loss 下降。所以训练 loss 是低的，但模型并没有真正依赖点——这就解释了为什么训练集样本推理出来也乱：点被学成了"可有可无的装饰"。

验证：拿已训好的 v3，推理时把 points 随机打乱 / 全清零，看输出是否几乎不变。若几乎不变，就坐实这个假设。
注入太浅、太弱。points_emb * 2.0 只加在 x_embedder 的输出上，后面没有再次注入；v2 打印里 Ratio 约 0.4%。深层 blocks 很容易把它淹没。

全图 MSE 的"背景免费奖励"。视频帧对大多数像素都基本没动，模型只要把背景 copy-paste 就能拿到 90% 的 loss，点邻域是否对齐对总 loss 几乎无影响。训练信号对 drag 本身极度不敏感。

颜色/对比度漂移来自训练分布本身。训练的 tgt 是视频下一帧，天然带 AWB/曝光/压缩漂移；LoRA 把这条"每步轻微调色"的映射学了进去，所以所有输出都带色偏。不是 prompt 问题。

重叠/撞一起是 (1)+(2) 的直接后果：无法区分两个点身份时，近邻点就会被 attention 当成同一个目标，位置平均，物体合并。

可以做的验证（低成本、排他性强）
点扰动实验：同一 ckpt 同一样本，把 points 随机打乱顺序、整体平移、或全部置零，对比输出差异。差异很小 ⇒ 模型没用点（坐实第 3 点）。
identity 实验：src_points 与 tgt_points 设为完全相同，期望应输出 src 本身。若输出仍变化/色偏，说明 LoRA 自带 shift（第 6 点）。
encoder 探针：构造一张 1 点 map 和 8 点 map，打印每一层输出的空间 norm 分布。如果 ResNet 后点位的峰值相对背景不超过 2~3×，身份信号基本没了。
loss 分解：把 loss 拆成"点 5×5 邻域 vs 其他"，看训练后期邻域 loss 是否下降很慢或根本不下降。
关掉 VAE shift：对比 "相同 ckpt, 关掉点条件" vs "正常推理"，颜色/对比差是否一致 → 判断色偏来源是 VAE-LoRA 还是点路径。
修复建议（针对"训练集自己也崩"这个结论）
按收益顺序：

换编码：双通道 / 多通道 heatmap + 带身份通道。推荐把一张 integer-index 图拆成：

C0 = src Gaussian heatmap（σ≈2~4 px）
C1 = tgt Gaussian heatmap
C2, C3 = 每像素的 (Δx, Δy) 稀疏光流，仅在 src 点位置非零 这样 GroupNorm 不会吃掉身份，且 src↔tgt 的配对天然在一张图里，attention 不用自己猜。
把 src 和 tgt 拼成一张图过 encoder（不是 2*B 独立过）。例如直接在通道维拼，把"谁对谁"变成局部卷积可学的模式。

强化 loss on drag 邻域：在 latent 空间对应 drag 点周围做 soft mask，损失加权 5~10×，强迫模型关心点对齐。

训练数据补非刚体样本：TPS/薄板样条合成 + 现有 drag 方法蒸馏，哪怕只占 20%，都能把"形状变化"这条支路激活。

多层注入 + 可学 scale：把 points_emb 以 zero-init adapter 形式加到前 10~20 个 DiT block 的 temb 或 hidden_states modulation，避免单点注入被深层淹没。

帧对预处理去色偏：直方图匹配 tgt→src，或者从训练对里剔除曝光差异超过阈值的样本；从根上断掉"学会调色"的捷径。

prompt 固定为训练时的空串：这是老问题，顺手改掉。

有没有可能我把src point，以及tgt point，四个坐标提取傅里叶embedding，然后作为一个token，专门出一个类似attention层训练？

这是一个非常有深度且完全符合当前生成模型研究前沿（Sparse Control）的思路。将点对（Point Pairs）编码为 Token 并注入 Attention 层，本质上是将“几何控制”转化为“语义交互”问题，这比使用 ControlNet 进行像素级干扰要优雅得多，且计算代价更小。

你的方案在学术上被称为 Point-Conditioned Transformer 或 Sparse Control Tokens。以下是该架构设计的关键要点和潜在的改进方向：

1. 架构方案建议：将点对转化为“动态控制指令”
不要将其仅仅视为位置信息，而应将其视为**“风格迁移指令”**。

A. Embedding 策略
傅里叶编码 (Fourier Embeddings)：这是处理坐标信息的标准做法。你提到的对 4 个坐标（x 
s
​
 ,y 
s
​
 ,x 
t
​
 ,y 
t
​
 ）分别编码是正确的。

Token 的构成：

建议不要简单的拼接。最好的做法是：

Embed_s = MLP(Fourier(x_s, y_s))

Embed_t = MLP(Fourier(x_t, y_t))

Token_i = Embed_s + Embed_t + Relative_Vector(x_t-x_s, y_t-y_s)

关键创新点：引入“相对位移矢量”的 Embedding。这对 Drag 任务至关重要，它直接告诉模型运动的方向和幅度。

B. 注入方式（Attention 层设计）
你并不需要从零训练一个 Attention 层。复用原有的 Cross-Attention 层是最高效的：

Token 级联：将这组 N 个点对的 Tokens 直接拼接到原本的 Text_Embeddings 之后。

修改 Cross-Attention：

原本：Attention(Q=Image, K=Text, V=Text)

改良：Attention(Q=Image, K=[Text, Points], V=[Text, Points])

效果：模型在计算注意力时，图像特征会同时“关注”文本 Prompt 和这些几何点对。当它看到一个点对 Token 时，它会产生“在这里发生特征变化”的响应。

2. 为什么你的这个思路能破解“Style Edit”难题？
目前你遇到的最大瓶颈是：只有点对，没有语义。通过将“Style”注入到 Token 中，你可以打破这个限制：

核心建议：特征提取器注入（Feature Injection）
在将点对转化为 Token 时，不要只输入坐标。同时输入该坐标点的“局部视觉特征”。

具体做法：

在 P 
s
​
  处，从 U-Net 的中间层提取特征向量 F 
source
​
 。

将 F 
source
​
  与你的傅里叶坐标编码拼接，一起投射为 Point Token。

这样，你的 Token 就变成了：[坐标信息] + [纹理/颜色信息]。

结果：Token 不再仅仅说“我要在这里动”，它是在说“把这个纹理搬到那边去”。这就实现了 Style Edit 的核心逻辑。

3. 需要攻克的工程细节
变长输入（Variable N）：

因为 Drag 的点对数量不固定（可能是 1 对，也可能是 10 对），你需要设置一个 Max_Tokens（例如 20）。不足的填入 Padding_Token，并在 Attention Mask 中遮盖掉 Padding 部分。

位置对齐的“软”处理：

训练时，单纯让模型通过 Token 对齐可能很难收敛。建议添加一个辅助 Loss：Feature Similarity Loss。

在训练阶段，强制要求 P 
t
​
  处的生成特征与 P 
s
​
  处的输入特征在语义空间上保持一致。

4. 与“ControlNet”或“高斯热图”相比的优势
模型权重（Weights）更轻：不需要像 ControlNet 那样复制整个 U-Net 权重，只需训练一个轻量的 MLP 映射层和几层交叉注意力层，显存占用极低。

稀疏性更友好：对于稀疏点，Attention 机制天生比 CNN 卷积更敏感，能够精准定位，不会像卷积那样造成“全局扩散”导致背景被误改。

给你的实验建议
Sanity Check：先不要训练整个 Diffusion 模型。写一个简单的 Attention Block，输入随机的 Point Tokens，看它是否能通过 Cross-Attention 将图像特征“引导”到特定的位置。

可视化注意力图 (Attention Maps)：这是你论文的核心论据。训练后，把 Cross-Attention 的 Heatmap 画出来，看模型是否真的在处理点对 P 
t
​
  时，将注意力集中在了 P 
s
​
  的区域。如果它是这么做的，你证明了模型确实学会了“根据点对进行风格对齐”。

## 后续优化
数据集收集
1.注意按类别分门别类
观察现在在哪些类别做得好/不好，然后针对性扩充
2.短程drag数据
3.少点标注数据

方法改进
1.只改编码方式：
1-N序号填充 -> 高斯填充 + 位移填充（3通道） v4

2.是否需要points encoder
points encoder v3
只有下采样的伪points encoder v3.2

3.将src points、tgt points经过傅里叶编码，合为N个token，N为pair组数，拼接在text后面  v5
  找到一个token，使得对目标点和源点注意力剧增

4.lora参数修改问题，ff.net.0, ff.net.2  v3.1