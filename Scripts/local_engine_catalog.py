#!/usr/bin/env python3
"""Stage existing signed engines for an offline local app preview.

Only transport locations change. Publisher keys, signed manifests and archive
contents stay intact and are authenticated by TUFF's normal engine installer.
Never copies publisher private keys or modifies the source release directory.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlsplit


def stage(source: Path, output: Path) -> Path:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Use a separate preview output directory")
    catalog = json.loads((source / "engine-catalog.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    for release in catalog["releases"]:
        manifest_name = Path(urlsplit(release["manifestURL"]).path).name
        archive_name = Path(urlsplit(release["archiveURL"]).path).name
        manifest_path = source / manifest_name
        archive_path = source / archive_name
        if manifest_path.is_symlink() or archive_path.is_symlink():
            raise ValueError("Engine inputs must be regular files")
        signed = json.loads(manifest_path.read_text())
        payload = json.loads(base64.b64decode(signed["payload"], validate=True))
        if signed["publisherKeyID"] not in catalog["publisherKeys"]:
            raise ValueError("Engine publisher is absent from catalog")
        if payload["packageID"] != release["packageID"] or payload["version"] != release["version"]:
            raise ValueError("Engine identity differs from catalog")
        if archive_path.stat().st_size != release["archiveBytes"]:
            raise ValueError("Engine archive size differs from catalog")
        digest = hashlib.sha256()
        with archive_path.open("rb") as archive:
            for chunk in iter(lambda: archive.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != payload["archiveSHA256"]:
            raise ValueError("Engine archive digest differs from signed payload")
        for original in (manifest_path, archive_path):
            destination = output / original.name
            if destination.is_symlink():
                raise ValueError("Preview destination cannot be a symlink")
            shutil.copy2(original, destination)
        release["manifestURL"] = (output / manifest_name).as_uri()
        release["archiveURL"] = (output / archive_name).as_uri()
    catalog["localPackageRoot"] = output.as_uri() + "/"
    destination = output / "engine-catalog.json"
    if destination.is_symlink():
        raise ValueError("Preview catalog destination cannot be a symlink")
    destination.write_text(json.dumps(catalog, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(stage(args.source, args.output))
