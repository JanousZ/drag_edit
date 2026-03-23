deepspeed2 + accelerate训练 flux kontext lora（实验性）
export NCCL_P2P_DISABLE=1
unset http_proxy  
export http_proxy="http://127.0.0.1:7890"
unset https_proxy  
export https_proxy="http://127.0.0.1:7890"
cd Kontext_train_ds2

accelerate launch --config_file ./train/deepspeed.yaml --main_process_port 29606 ./train/train_ds2.py --num_epochs 5 --lr 1e-4 --save_steps 500 > train.log 2>&1

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