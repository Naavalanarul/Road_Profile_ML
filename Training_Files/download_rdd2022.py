"""
download_rdd2022.py

The figshare RDD2022 archive (13 GB) is one zip holding a stored (uncompressed)
zip per country. This fetches only the selected country zips with HTTP range
requests -- skipping Norway (9.9 GB) and China_Drone -- and extracts them to
<out>/RDD2022/<Country>/...

    python download_rdd2022.py --out ../Dataset
"""

import argparse
import os
import struct
import subprocess
import zipfile

URL = "https://ndownloader.figshare.com/files/38030910"
COUNTRIES = ["India", "Japan", "Czech", "United_States", "China_MotorBike"]
PIECE = 16 * 1024 * 1024


def get_range(start, end):
    """Bytes [start, end] inclusive. Figshare redirects to a 10 s signed URL, so re-resolve each call."""
    # curl, not requests: requests stalled indefinitely on large ranges here.
    for _ in range(5):
        try:
            r = subprocess.run(["curl", "-sSL", "--fail", "--speed-limit", "10000", "--speed-time", "60",
                                "-r", f"{start}-{end}", URL], capture_output=True, timeout=1800)
            if r.returncode == 0 and len(r.stdout) == end - start + 1:
                return r.stdout
        except subprocess.TimeoutExpired:
            pass
    raise RuntimeError(f"range {start}-{end} failed")


def list_outer_zip():
    """Return {name: (data_start, size)} for the stored members of the outer zip."""
    head = subprocess.run(["curl", "-sSL", "-r", "0-0", "-D", "-", "-o", os.devnull, URL],
                          capture_output=True, text=True, timeout=60).stdout
    size = int(head.lower().split("content-range: bytes 0-0/")[-1].split()[0])
    tail = get_range(size - 65536, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    cd_size, cd_offset = struct.unpack("<II", tail[eocd + 12:eocd + 20])
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:  # zip64 end record
        loc = tail.rfind(b"PK\x06\x07")
        eocd64_off, = struct.unpack("<Q", tail[loc + 8:loc + 16])
        rec = get_range(eocd64_off, eocd64_off + 55)
        cd_size, cd_offset = struct.unpack("<QQ", rec[40:56])
    cd = get_range(cd_offset, cd_offset + cd_size - 1)

    members, p = {}, 0
    while cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        csize, = struct.unpack("<I", cd[p + 20:p + 24])
        nlen, xlen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        offset, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode()
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        q = 0
        while q + 4 <= len(extra):  # zip64 sizes/offsets for the >4 GB archive
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 0x0001:
                vals, k = extra[q + 4:q + 4 + hlen], 0
                if struct.unpack("<I", cd[p + 24:p + 28])[0] == 0xFFFFFFFF:
                    k += 8
                if csize == 0xFFFFFFFF:
                    csize, = struct.unpack("<Q", vals[k:k + 8]); k += 8
                if offset == 0xFFFFFFFF:
                    offset, = struct.unpack("<Q", vals[k:k + 8])
            q += 4 + hlen
        assert method == 0, f"{name} is compressed; expected stored"
        local = get_range(offset, offset + 29)
        lnlen, lxlen = struct.unpack("<HH", local[26:30])
        members[name] = (offset + 30 + lnlen + lxlen, csize)
        p += 46 + nlen + xlen + clen
    return members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../Dataset")
    ap.add_argument("--countries", nargs="+", default=COUNTRIES)
    args = ap.parse_args()
    zips_dir = os.path.join(args.out, "RDD2022", "zips")
    os.makedirs(zips_dir, exist_ok=True)

    members = list_outer_zip()
    for country in args.countries:
        start, size = members[f"RDD2022/{country}.zip"]
        path = os.path.join(zips_dir, f"{country}.zip")
        have = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"{country}: {size / 1e6:.0f} MB (have {have / 1e6:.0f})", flush=True)

        with open(path, "ab") as f:  # resumable
            pos = have
            while pos < size:
                end = min(pos + PIECE, size) - 1
                f.write(get_range(start + pos, start + end))
                pos = end + 1
                print(f"  {pos / 1e6:.0f}/{size / 1e6:.0f} MB", flush=True)

        with zipfile.ZipFile(path) as z:
            z.extractall(os.path.join(args.out, "RDD2022"))
        print(f"  extracted {country}", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
