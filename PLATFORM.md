### Evaluation platform

Our evaluation platform relies on the following software
 - [GNU Bash](https://www.gnu.org/software/bash/) to run the initialization and evaluation scripts
 - Singularity 3.7
 - Ubuntu 20.04 as our singularity container
 - Python 3.7
 - `conda` + `pip` to install the software dependencies

Note that your code must be compatible with those specifications at evalution time only. For training,
participants do not have to use this exact setup (development can be done under MacOS for example).

Regarding hardware, every evaluation will be run on a separated virtual machine with the following specifications.
 - 1 CPU core with 20GB RAM
```
processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 85
model name	: Intel Xeon Processor (Skylake, IBRS)
stepping	: 4
microcode	: 0x1
cpu MHz		: 2494.140
cache size	: 16384 KB
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology eagerfpu pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 arat pku ospke avx512_vnni md_clear spec_ctrl intel_stibp
bogomips	: 4988.28
clflush size	: 64
cache_alignment	: 64
address sizes	: 46 bits physical, 48 bits virtual
```
 - 1 vGPU V100-8G + CUDA 11.0
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.102.04   Driver Version: 450.102.04   CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GRID V100D-8C       On   | 00000000:00:05.0 Off |                    0 |
| N/A   N/A    P0    N/A /  N/A |    560MiB /  8192MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
