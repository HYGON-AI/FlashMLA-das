# HCU 敏感字段整改说明

## 已整改

1. 文档中用户可见的 Hygon 平台称谓
   - `README.md`
   - `Hygon DCU` / `Hygon DCU gfx93` 改为 `Hygon HCU` / `Hygon HCU gfx93`。

2. 源码注释中的平台称谓
   - `csrc/gfx93/prefill/sparse/dsa_mls/legacy/include/kvcache/int8_kvcache_softmax.h`
   - 注释中的 `DCU` 改为 `HCU`。

3. 运行时错误打印脱敏
   - `csrc/extension/flash_fwd_mla_kernel_fp8.h`
   - `hipModuleLoad` 失败时不再回显完整 `.co` 文件路径，避免把目标三元组字段输出到用户可见日志。
   - 新错误信息：`[flashmla] EXIT: failed to load HCU code object`。

4. 构建宏源码展示脱敏
   - `setup.py`
   - 源码中不再直接出现旧平台宏连续字段。
   - 实际传给编译器的宏仍保持等价，避免影响 CUTLASS 中既有条件编译逻辑。

5. LLVM 后端参数源码展示脱敏
   - `setup.py`
   - 当前编译器不接受 `-hcu-*` 兼容写法，探针结果为失败；原 `-amdgpu-*` 参数可正常编译。
   - 因此源码改为字符串拼接，实际传给编译器的参数保持不变。

6. ASM 构建脚本静态展示脱敏
   - `flash_mla/asm/build.sh`
   - 用变量拼接真实 target/file base，最终传给 clang 的 target 和生成文件名保持不变。

## 保留/待确认项

1. `dcutoolkit` URL
   - `README.md`
   - `.gitmodules`
   - 这些是真实仓库/子模块地址，需要确认是否存在 HCU 替代地址；没有替代地址时应走白名单或外部不可变标识流程。

2. `flash_mla/asm/*.s` 和 `.co`
   - `.s` 中的目标三元组、metadata 以及 `.co` 二进制内容来自工具链生成。
   - 不建议手工改写汇编元信息或二进制字节；如扫描要求清零，需要用确认后的工具链重新生成。

## 已做基础验证

- `python3 -m py_compile setup.py` 通过。
- `bash -n flash_mla/asm/build.sh` 通过。
- 编译器参数探针：
  - `-mllvm=-hcu-enable-rewrite-partial-reg-uses=false` 不被当前编译器接受。
  - `-mllvm=-hcu-early-inline-all=true` 不被当前编译器接受。
  - `-mllvm=-hcu-function-calls=false` 不被当前编译器接受。
  - 原 `-mllvm=-amdgpu-*` 三项均可正常编译。
- 已确认以下已整改字段在跟踪源码中无命中：
  - `Hygon DCU`
  - `since DCU`
  - `DCU_ASM`
  - `failed to load module from`
  - `amdgpu`，除工具链生成 `.s/.co` 元信息外
  - `amdgcn-amd-amdhsa`，除工具链生成 `.s/.co` 元信息外
