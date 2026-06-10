"""VisDrone2019-DET 데이터셋 다운로드 스크립트.

라이선스(CC BY-NC-SA 3.0)상 데이터는 저장소에 포함하지 않고, 사용자가 직접 내려받습니다.
공개 미러(Ultralytics assets release)에서 train/val/test-dev split을 받아 압축 해제합니다.

사용 예:
    python data/download_visdrone.py --out datasets/VisDrone
"""
import argparse
import zipfile
from pathlib import Path
from urllib.request import urlopen

# 공개 미러 (원 출처: http://www.aiskyeye.com/ , https://github.com/VisDrone/VisDrone-Dataset)
URLS = {
    "VisDrone2019-DET-train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "VisDrone2019-DET-val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    "VisDrone2019-DET-test-dev": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
}


def download(url: str, dst: Path, chunk: int = 1 << 20) -> None:
    print(f"  ↓ {url}")
    with urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(dst, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = done * 100 // total
                    print(f"\r    {pct:3d}%  ({done >> 20} / {total >> 20} MB)", end="")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/VisDrone", help="압축 해제 대상 폴더")
    ap.add_argument("--keep-zip", action="store_true", help="zip 파일 보존")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, url in URLS.items():
        target = out / name
        if target.exists():
            print(f"[skip] {name} (이미 존재)")
            continue
        zip_path = out / f"{name}.zip"
        print(f"[download] {name}")
        download(url, zip_path)
        print(f"[extract]  {name}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out)
        if not args.keep_zip:
            zip_path.unlink()

    print("\n완료. 다음 단계: python data/visdrone_to_yolo.py --root", out)


if __name__ == "__main__":
    main()
