#!/usr/bin/env python3
"""Update ReMem.pptx labels and append two rendered R-direction slides.

Uses only the Python standard library so the build remains self-contained.
"""

from __future__ import annotations

import copy
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


HERE = Path(__file__).resolve().parent
PPTX = HERE / "ReMem.pptx"
ARCHIVE = HERE / "archive" / "ReMem_pre_r_direction_6slides.pptx"
SLIDE_IMAGES = [
    HERE / "generated" / "r_direction_slide_7.png",
    HERE / "generated" / "r_direction_slide_8.png",
]

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    "ep": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties",
    "vt": "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes",
}
for prefix, uri in NS.items():
    if prefix not in {"rel", "ct", "ep", "vt"}:
        ET.register_namespace(prefix, uri)


def parse(path: Path) -> ET.ElementTree:
    return ET.parse(path)


def write(tree: ET.ElementTree, path: Path) -> None:
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def all_text_nodes(path: Path):
    tree = parse(path)
    nodes = tree.findall(".//a:t", NS)
    return tree, nodes


def replace_exact(nodes, old: str, new: str, limit: int | None = None) -> int:
    count = 0
    for node in nodes:
        if node.text == old and (limit is None or count < limit):
            node.text = new
            count += 1
    return count


def fix_existing_labels(root: Path) -> None:
    # Slide 3: explicitly describe the gap analysis.
    path = root / "ppt/slides/slide3.xml"
    tree, nodes = all_text_nodes(path)
    replace_exact(nodes, "차이 확인", "Representation Gap 분석", 1)
    write(tree, path)

    # Slide 4: the two existing tables are Gap and M-direction, not R vs M.
    path = root / "ppt/slides/slide4.xml"
    tree, nodes = all_text_nodes(path)
    replace_exact(nodes, "Reasning", "R/M Gap", 1)
    replace_exact(nodes, "Memorization", "M-direction", 1)
    write(tree, path)

    # Slide 5: clarify cross-model Gap results.
    path = root / "ppt/slides/slide5.xml"
    tree, nodes = all_text_nodes(path)
    replace_exact(nodes, "R ", "R/M Gap ", 1)
    replace_exact(nodes, "관련 ", "", 1)
    write(tree, path)

    # Slide 6: clarify cross-model M-direction results.
    path = root / "ppt/slides/slide6.xml"
    tree, nodes = all_text_nodes(path)
    replace_exact(nodes, "Memorization ", "Cross-model M-direction ", 1)
    replace_exact(nodes, "관련 ", "", 1)
    replace_exact(nodes, "Head/Neuron", "Components", 1)
    write(tree, path)


def slide_xml(image_rid: str = "rId2") -> bytes:
    p = NS["p"]
    a = NS["a"]
    r = NS["r"]
    sld = ET.Element(f"{{{p}}}sld")
    c_sld = ET.SubElement(sld, f"{{{p}}}cSld")
    tree = ET.SubElement(c_sld, f"{{{p}}}spTree")
    nv = ET.SubElement(tree, f"{{{p}}}nvGrpSpPr")
    ET.SubElement(nv, f"{{{p}}}cNvPr", {"id": "1", "name": ""})
    ET.SubElement(nv, f"{{{p}}}cNvGrpSpPr")
    ET.SubElement(nv, f"{{{p}}}nvPr")
    grp = ET.SubElement(tree, f"{{{p}}}grpSpPr")
    xfrm = ET.SubElement(grp, f"{{{a}}}xfrm")
    ET.SubElement(xfrm, f"{{{a}}}off", {"x": "0", "y": "0"})
    ET.SubElement(xfrm, f"{{{a}}}ext", {"cx": "0", "cy": "0"})
    ET.SubElement(xfrm, f"{{{a}}}chOff", {"x": "0", "y": "0"})
    ET.SubElement(xfrm, f"{{{a}}}chExt", {"cx": "0", "cy": "0"})
    pic = ET.SubElement(tree, f"{{{p}}}pic")
    nv_pic = ET.SubElement(pic, f"{{{p}}}nvPicPr")
    ET.SubElement(nv_pic, f"{{{p}}}cNvPr", {"id": "2", "name": "R-direction result slide"})
    c_nv = ET.SubElement(nv_pic, f"{{{p}}}cNvPicPr")
    ET.SubElement(c_nv, f"{{{a}}}picLocks", {"noChangeAspect": "1"})
    ET.SubElement(nv_pic, f"{{{p}}}nvPr")
    fill = ET.SubElement(pic, f"{{{p}}}blipFill")
    ET.SubElement(fill, f"{{{a}}}blip", {f"{{{r}}}embed": image_rid})
    stretch = ET.SubElement(fill, f"{{{a}}}stretch")
    ET.SubElement(stretch, f"{{{a}}}fillRect")
    sp_pr = ET.SubElement(pic, f"{{{p}}}spPr")
    px = ET.SubElement(sp_pr, f"{{{a}}}xfrm")
    ET.SubElement(px, f"{{{a}}}off", {"x": "0", "y": "0"})
    ET.SubElement(px, f"{{{a}}}ext", {"cx": "12192000", "cy": "6858000"})
    geom = ET.SubElement(sp_pr, f"{{{a}}}prstGeom", {"prst": "rect"})
    ET.SubElement(geom, f"{{{a}}}avLst")
    clr = ET.SubElement(sld, f"{{{p}}}clrMapOvr")
    ET.SubElement(clr, f"{{{a}}}masterClrMapping")
    return ET.tostring(sld, encoding="UTF-8", xml_declaration=True)


def rels_xml(image_name: str) -> bytes:
    rel_ns = NS["rel"]
    root = ET.Element(f"{{{rel_ns}}}Relationships")
    ET.SubElement(root, f"{{{rel_ns}}}Relationship", {
        "Id": "rId1",
        "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout",
        "Target": "../slideLayouts/slideLayout7.xml",
    })
    ET.SubElement(root, f"{{{rel_ns}}}Relationship", {
        "Id": "rId2",
        "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
        "Target": f"../media/{image_name}",
    })
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)


def append_slides(root: Path) -> None:
    for index, image in zip((7, 8), SLIDE_IMAGES):
        image_name = f"image{index + 9}.png"  # image16.png, image17.png
        shutil.copy2(image, root / "ppt/media" / image_name)
        (root / "ppt/slides" / f"slide{index}.xml").write_bytes(slide_xml())
        rel_path = root / "ppt/slides/_rels" / f"slide{index}.xml.rels"
        rel_path.write_bytes(rels_xml(image_name))

    # Content types.
    ct_path = root / "[Content_Types].xml"
    ct_tree = parse(ct_path)
    ct_root = ct_tree.getroot()
    existing = {x.attrib.get("PartName") for x in ct_root}
    for index in (7, 8):
        name = f"/ppt/slides/slide{index}.xml"
        if name not in existing:
            ET.SubElement(ct_root, f"{{{NS['ct']}}}Override", {
                "PartName": name,
                "ContentType": "application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
            })
    write(ct_tree, ct_path)

    # Presentation relationships.
    rel_path = root / "ppt/_rels/presentation.xml.rels"
    rel_tree = parse(rel_path)
    rel_root = rel_tree.getroot()
    for rid, index in (("rId46", 7), ("rId47", 8)):
        ET.SubElement(rel_root, f"{{{NS['rel']}}}Relationship", {
            "Id": rid,
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
            "Target": f"slides/slide{index}.xml",
        })
    write(rel_tree, rel_path)

    # Presentation slide order.
    pres_path = root / "ppt/presentation.xml"
    pres_tree = parse(pres_path)
    pres_root = pres_tree.getroot()
    slide_list = pres_root.find("p:sldIdLst", NS)
    assert slide_list is not None
    ET.SubElement(slide_list, f"{{{NS['p']}}}sldId", {"id": "264", f"{{{NS['r']}}}id": "rId46"})
    ET.SubElement(slide_list, f"{{{NS['p']}}}sldId", {"id": "265", f"{{{NS['r']}}}id": "rId47"})
    write(pres_tree, pres_path)

    # Extended properties: slide count only; titles metadata is optional.
    app_path = root / "docProps/app.xml"
    app_tree = parse(app_path)
    slides = app_tree.find("ep:Slides", NS)
    if slides is not None:
        slides.text = "8"
    write(app_tree, app_path)


def main() -> None:
    for path in [PPTX, *SLIDE_IMAGES]:
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        shutil.copy2(PPTX, ARCHIVE)
    source = ARCHIVE
    with tempfile.TemporaryDirectory(prefix="remem_pptx_") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(source) as zf:
            zf.extractall(root)
        fix_existing_labels(root)
        append_slides(root)
        out = PPTX.with_suffix(".pptx.tmp")
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(root).as_posix())
        with zipfile.ZipFile(out) as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"corrupt member: {bad}")
        out.replace(PPTX)
    print(f"updated: {PPTX}")


if __name__ == "__main__":
    main()
