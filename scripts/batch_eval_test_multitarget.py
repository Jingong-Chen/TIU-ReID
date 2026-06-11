#!/usr/bin/env python3
"""
批量为 multitarget_v2 的所有模型补充 test 评估
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

def _env():
    env = os.environ.copy()
    env.setdefault("REID_DATA_DIR", str(REPO / "data"))
    env.setdefault("REID_OUTPUT_DIR", str(REPO / "output"))
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{REPO}:{pythonpath}"
    else:
        env["PYTHONPATH"] = str(REPO)
    return env

def main():
    # 查找所有需要评估的模型目录
    removable_dir = REPO / "output" / "removable"
    model_dirs = sorted(removable_dir.glob("removable_market1501_tid*_seed*_retfixF"))
    
    print(f"找到 {len(model_dirs)} 个模型目录")
    
    # 统计需要评估的模型
    to_eval = []
    for model_dir in model_dirs:
        sanity_report = model_dir / "sanity_report_removable.json"
        if not sanity_report.exists():
            print(f"⚠️ 跳过（无 sanity_report）: {model_dir.name}")
            continue
        
        # 检查是否已有 test 指标
        with open(sanity_report) as f:
            report = json.load(f)
        
        if "test_without_mAP" in report and report["test_without_mAP"] is not None:
            print(f"✓ 已有 test 指标: {model_dir.name}")
            continue
        
        to_eval.append(model_dir)
    
    print(f"\n需要评估 {len(to_eval)} 个模型")
    
    if not to_eval:
        print("✅ 所有模型都已有 test 指标")
        return
    
    # 批量评估
    failed = []
    # 读取 teacher config 路径
    teacher_dir = Path(os.environ.get("REID_OUTPUT_DIR", REPO / "output")) / "transreid" / "market_teacher_r50"
    cfg_path_file = teacher_dir / "teacher_cfg_path.txt"
    if not cfg_path_file.exists():
        print(f"❌ 找不到 teacher config: {cfg_path_file}")
        return
    teacher_cfg = cfg_path_file.read_text().strip()
    print(f"使用 teacher config: {teacher_cfg}")
    
    for model_dir in tqdm(to_eval, desc="评估 test split"):
        try:
            # 从目录名提取 tid
            # 格式: removable_market1501_tid{tid}_seed{seed}_retfixF
            dir_name = model_dir.name
            parts = dir_name.split("_")
            tid_str = [p for p in parts if p.startswith("tid")][0]
            tid = tid_str.replace("tid", "")
            
            # 运行 eval_test_split.py
            cmd = [
                "python3",
                str(REPO / "scripts" / "eval_test_split.py"),
                "--mvp_dir", str(model_dir),
                "--cfg", teacher_cfg,
                "--forget_id", str(tid),
                "--target_base_scale", "0.0"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(REPO),
                env=_env(),
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
                check=False
            )
            
            if result.returncode != 0:
                print(f"\n❌ 评估失败: {model_dir.name}")
                print(f"   错误: {result.stderr[:200]}")
                failed.append(model_dir.name)
            else:
                # 验证是否成功写入
                sanity_report = model_dir / "sanity_report_removable.json"
                with open(sanity_report) as f:
                    report = json.load(f)
                if "test_without_mAP" in report:
                    tqdm.write(f"✓ {model_dir.name}: test_mAP={report['test_without_mAP']:.4f}")
                else:
                    tqdm.write(f"⚠️ {model_dir.name}: 未找到 test 指标")
                    failed.append(model_dir.name)
        
        except subprocess.TimeoutExpired:
            print(f"\n⏱️ 超时: {model_dir.name}")
            failed.append(model_dir.name)
        except Exception as e:
            print(f"\n❌ 异常: {model_dir.name}: {e}")
            failed.append(model_dir.name)
    
    # 总结
    print("\n" + "=" * 70)
    print(f"✅ 成功评估: {len(to_eval) - len(failed)} 个")
    print(f"❌ 失败: {len(failed)} 个")
    if failed:
        print("\n失败的模型:")
        for name in failed:
            print(f"  - {name}")
    print("=" * 70)

if __name__ == "__main__":
    main()
