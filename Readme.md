deepspeed2 + accelerate训练 flux kontext lora（实验性）
export NCCL_P2P_DISABLE=1
unset http_proxy  
export http_proxy="http://127.0.0.1:7890"
unset https_proxy  
export https_proxy="http://127.0.0.1:7890"
cd Kontext_train_ds2

```bash
accelerate launch --config_file ./train/deepspeed.yaml --main_process_port 29606 ./train/train_ds2.py \  
    --num_epochs 50 --lr 1e-4 --save_steps 500 > train.log 2>&1
```

```bash
accelerate launch --config_file ./train/deepspeed.yaml --main_process_port 29607 ./train/train_ds2.py --num_epochs 50 --lr 1e-4 --save_steps 500 --output_dir lora_ckpt_v2.1_ > train2.log 2>&1
```

```bash
python test.py \
--use_lora \
--checkpoint_dir "./lora_ckpt_v2.1_/checkpoint-23000" \
--output_dir "./output_2.1" \
--dataset_jsonl "/home/yanzhang/dragdatasets/paired_frames.jsonl"

python test.py \
--use_lora \
--checkpoint_dir "./lora_ckpt/checkpoint-23000" \
--output_dir "./bench" \
--dataset_jsonl "/home/yanzhang/dragdatasets/paired_frames.jsonl"
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

## 后续优化
数据集收集
方法改进
：多层注入
：傅里叶嵌入编码
：从分开编码 -> 流畅流场编码

## claude code请你分析问题
我有两版方案
v2.0:训练时不添加points_emb的缩放因子，训练时核心点位能量占比达到100％-500％
推理时核心点位能量占比达到5％左右
推理时不添加缩放因子则模型忽略控制条件随意生成，添加100.0的缩放因子则能遵循控制条件，但是容易扭曲有伪影。

v2.1训练时添加points_emb的缩放因子100.0，训练时核心点位能量占比达到2％-5％
推理时核心点位能量占比达到0.05％左右
推理时无论是无缩放因子还是100.0的缩放因子，模型都忽略控制条件随意生成

我在训练drag-style image edit的模型，基模是Flux-Kontext
请你阅读point.py,dit.py,train_ds2.py等代码，首先寻找是否存在训练推理不一致的地方，或者是其他常见错误；
另外分析出现这类问题的原因，以及给出可能的解决方案。