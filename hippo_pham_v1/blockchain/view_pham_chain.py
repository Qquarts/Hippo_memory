#!/usr/bin/env python3
"""
📜 PHAM Chain Viewer — 블록체인 체인 파일 시각화 도구

Usage:
    python3 view_pham_chain.py                    # 모든 체인 파일 목록 표시
    python3 view_pham_chain.py <chain_file.json>  # 특정 체인 상세 보기
    python3 view_pham_chain.py --simple <file>    # 간단한 요약 보기
    python3 view_pham_chain.py --compact          # 모든 체인 한눈에 보기
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# 🎨 색상 코드
class Color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def format_label(label):
    """기여도 라벨에 색상 적용"""
    colors = {
        "A_HIGH": Color.GREEN,
        "A_MID": Color.CYAN,
        "B_HIGH": Color.YELLOW,
        "B_MID": Color.YELLOW,
        "C_LOW": Color.RED
    }
    color = colors.get(label, "")
    return f"{color}{label}{Color.END}"

def format_size(size):
    """바이트 크기를 읽기 쉽게 변환"""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size/1024:.1f} KB"
    else:
        return f"{size/(1024*1024):.1f} MB"

def print_signal_bar(signal_name, value, compact=False):
    """신호 값을 ASCII 바 그래프로 표시"""
    if compact:
        bar_length = int(value * 10)
        bar = "█" * bar_length + "░" * (10 - bar_length)
        return f"{signal_name[0].upper()}:{bar}"
    else:
        bar_length = int(value * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"         {signal_name:6s}: {bar} {value:.4f}")

def compact_view():
    """모든 체인을 한눈에 보기"""
    chains = sorted(Path(".").glob("pham_chain_*.json"))
    
    if not chains:
        print(f"{Color.RED}❌ 체인 파일을 찾을 수 없습니다.{Color.END}")
        return
    
    print(f"\n{Color.BOLD}{'='*80}{Color.END}")
    print(f"{Color.CYAN}📦 PHAM 블록체인 요약 ({len(chains)}개 체인){Color.END}")
    print(f"{Color.BOLD}{'='*80}{Color.END}\n")
    
    for chain_path in chains:
        try:
            with open(chain_path, "r") as f:
                chain_data = json.load(f)
            
            contribution_blocks = [b for b in chain_data if b["index"] > 0]
            
            if not contribution_blocks:
                continue
            
            # 파일명에서 타이틀 추출
            title = chain_path.stem.replace("pham_chain_", "")
            
            print(f"{Color.BOLD}🔗 {title}{Color.END}")
            
            for block in contribution_blocks:
                data = block["data"]
                idx = block["index"]
                score = data.get("score", 0)
                label = data.get("label", "Unknown")
                timestamp = data.get("timestamp", "Unknown")
                
                # 신호 바 (compact)
                signals = data.get("signals", {})
                signal_bars = [
                    print_signal_bar("byte", signals.get("byte", 0), compact=True),
                    print_signal_bar("text", signals.get("text", 0), compact=True),
                    print_signal_bar("ast", signals.get("ast", 0), compact=True),
                    print_signal_bar("exec", signals.get("exec", 0), compact=True)
                ]
                signal_str = " ".join(signal_bars)
                
                print(f"  [{idx}] {score:.3f} {format_label(label):20s} | {signal_str} | {timestamp}")
            
            print()
            
        except Exception as e:
            print(f"  {Color.RED}❌ {chain_path.name}: 오류 ({e}){Color.END}\n")
    
    print(f"{Color.BOLD}{'='*80}{Color.END}\n")

def simple_view(chain_path):
    """간단한 요약 보기"""
    if not Path(chain_path).exists():
        print(f"{Color.RED}❌ 파일을 찾을 수 없습니다: {chain_path}{Color.END}")
        return
    
    try:
        with open(chain_path, "r") as f:
            chain_data = json.load(f)
    except Exception as e:
        print(f"{Color.RED}❌ 파일 읽기 실패: {e}{Color.END}")
        return
    
    contribution_blocks = [b for b in chain_data if b["index"] > 0]
    
    print(f"\n{Color.BOLD}{'='*70}{Color.END}")
    print(f"{Color.CYAN}📜 {Path(chain_path).name}{Color.END}")
    print(f"{Color.BOLD}{'='*70}{Color.END}\n")
    
    print(f"  총 블록: {len(chain_data)}")
    print(f"  기여 블록: {len(contribution_blocks)}")
    
    if contribution_blocks:
        avg_score = sum(b["data"].get("score", 0) for b in contribution_blocks) / len(contribution_blocks)
        print(f"  평균 점수: {Color.YELLOW}{avg_score:.4f}{Color.END}")
        print()
        
        # 테이블 헤더
        print(f"  {Color.BOLD}{'블록':^6} {'점수':^8} {'등급':^15} {'파일명':^30}{Color.END}")
        print(f"  {'-'*66}")
        
        for block in contribution_blocks:
            data = block["data"]
            idx = block["index"]
            score = data.get("score", 0)
            label = data.get("label", "Unknown")
            title = data.get("title", "Unknown")[:28]
            
            print(f"  {idx:^6} {score:^8.4f} {format_label(label):^24} {title}")
        
        print()
    
    print(f"{Color.BOLD}{'='*70}{Color.END}\n")

def list_chain_files():
    """현재 디렉터리의 모든 체인 파일 나열"""
    chains = sorted(Path(".").glob("pham_chain_*.json"))
    
    if not chains:
        print(f"{Color.RED}❌ 체인 파일을 찾을 수 없습니다.{Color.END}")
        return
    
    print(f"\n{Color.BOLD}{'='*70}{Color.END}")
    print(f"{Color.CYAN}📦 사용 가능한 체인 파일 ({len(chains)}개){Color.END}")
    print(f"{Color.BOLD}{'='*70}{Color.END}\n")
    
    for chain_path in chains:
        try:
            with open(chain_path, "r") as f:
                chain_data = json.load(f)
            
            total_blocks = len(chain_data)
            file_size = chain_path.stat().st_size
            contribution_blocks = [b for b in chain_data if b["index"] > 0]
            
            if contribution_blocks:
                avg_score = sum(b["data"].get("score", 0) for b in contribution_blocks) / len(contribution_blocks)
                labels = [b["data"].get("label", "Unknown") for b in contribution_blocks]
                label_counts = {label: labels.count(label) for label in set(labels)}
                label_str = ", ".join([f"{format_label(k)}:{v}" for k, v in sorted(label_counts.items())])
            else:
                avg_score = 0.0
                label_str = "N/A"
            
            print(f"  {Color.BOLD}{chain_path.name}{Color.END}")
            print(f"    • 블록: {total_blocks} | 크기: {format_size(file_size)} | 평균: {avg_score:.4f}")
            print(f"    • 분포: {label_str}")
            print()
            
        except Exception as e:
            print(f"  {Color.RED}❌ {chain_path.name}: 읽기 실패 ({e}){Color.END}\n")
    
    print(f"{Color.BOLD}{'='*70}{Color.END}")
    print(f"\n💡 사용법:")
    print(f"   python3 view_pham_chain.py <파일명>        # 상세 보기")
    print(f"   python3 view_pham_chain.py --simple <파일> # 요약 보기")
    print(f"   python3 view_pham_chain.py --compact       # 전체 한눈에\n")

def verify_chain(chain_data):
    """체인 무결성 검증"""
    print(f"\n{Color.BOLD}🔍 체인 무결성 검증{Color.END}")
    print("─" * 70)
    
    errors = []
    
    if len(chain_data) == 0:
        errors.append("❌ 체인이 비어있습니다.")
    elif chain_data[0]["index"] != 0:
        errors.append(f"❌ Genesis 블록 인덱스 오류: {chain_data[0]['index']}")
    elif chain_data[0]["hash"] != "0":
        errors.append(f"❌ Genesis 블록 해시 오류: {chain_data[0]['hash']}")
    
    for i in range(1, len(chain_data)):
        prev = chain_data[i-1]
        curr = chain_data[i]
        
        if curr["previous_hash"] != prev["hash"]:
            errors.append(f"❌ 블록 {i}: 해시 체인 끊김")
        
        if curr["index"] != i:
            errors.append(f"❌ 블록 {i}: 인덱스 불일치 ({curr['index']})")
    
    if errors:
        for error in errors:
            print(f"  {Color.RED}{error}{Color.END}")
    else:
        print(f"  {Color.GREEN}✅ 체인 무결성 확인됨 ({len(chain_data)} 블록){Color.END}")
    
    print()

def view_chain_details(chain_path):
    """체인 파일의 상세 내용 표시"""
    if not Path(chain_path).exists():
        print(f"{Color.RED}❌ 파일을 찾을 수 없습니다: {chain_path}{Color.END}")
        return
    
    try:
        with open(chain_path, "r") as f:
            chain_data = json.load(f)
    except Exception as e:
        print(f"{Color.RED}❌ 파일 읽기 실패: {e}{Color.END}")
        return
    
    print(f"\n{Color.BOLD}{'='*70}{Color.END}")
    print(f"{Color.CYAN}📜 {chain_path}{Color.END}")
    print(f"{Color.BOLD}{'='*70}{Color.END}")
    
    verify_chain(chain_data)
    
    for block in chain_data:
        idx = block["index"]
        
        if idx == 0:
            print(f"\n{Color.BOLD}🌱 Block 0 (Genesis){Color.END}")
            print("─" * 70)
            print(f"  Timestamp: {datetime.fromtimestamp(block['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Hash: {block['hash']}")
            print()
        else:
            data = block["data"]
            print(f"\n{Color.BOLD}📦 Block {idx}: {data.get('title', 'Unknown')}{Color.END}")
            print("─" * 70)
            print(f"  작성자: {Color.CYAN}{data.get('author', 'Unknown')}{Color.END}")
            print(f"  시간: {data.get('timestamp', 'Unknown')}")
            print(f"  점수: {Color.YELLOW}{data.get('score', 0):.4f}{Color.END} ({format_label(data.get('label', 'Unknown'))})")
            print(f"  설명: {data.get('description', 'N/A')}")
            print()
            print(f"  해시:")
            print(f"    • 파일: {data.get('hash', 'N/A')[:64]}...")
            print(f"    • CID: {data.get('cid', 'N/A')}")
            print(f"    • 이전: {block.get('previous_hash', 'N/A')[:64]}...")
            print(f"    • 블록: {block.get('hash', 'N/A')[:64]}...")
            
            if "signals" in data:
                print()
                print(f"  신호:")
                signals = data["signals"]
                print_signal_bar("byte", signals.get("byte", 0))
                print_signal_bar("text", signals.get("text", 0))
                print_signal_bar("ast", signals.get("ast", 0))
                print_signal_bar("exec", signals.get("exec", 0))
            
            if "raw_bytes" in data:
                raw_size = len(data["raw_bytes"]) // 2
                print()
                print(f"  Raw 데이터: {format_size(raw_size)}")
            
            if "exec_output" in data and data["exec_output"]:
                exec_preview = data["exec_output"][:100].replace("\n", " ")
                if len(data["exec_output"]) > 100:
                    exec_preview += "..."
                print()
                print(f"  실행 출력: {exec_preview}")
            
            print()
    
    print(f"{Color.BOLD}{'='*70}{Color.END}\n")

def main():
    if len(sys.argv) < 2:
        list_chain_files()
    elif sys.argv[1] == "--compact":
        compact_view()
    elif sys.argv[1] == "--simple" and len(sys.argv) > 2:
        simple_view(sys.argv[2])
    else:
        view_chain_details(sys.argv[1])

if __name__ == "__main__":
    main()
